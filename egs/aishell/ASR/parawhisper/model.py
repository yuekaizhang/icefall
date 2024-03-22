# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2023 ASLP@NWPU (authors: He Wang, Fan Yu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet) and
# FunASR(https://github.com/alibaba-damo-academy/FunASR)

from typing import Dict, List, Optional, Tuple
import re
import whisper
import torch
from cif import Cif
from label_smoothing import LabelSmoothingLoss
from whisper_tokens import load_new_tokens_dict_list

class ParaWhisper(torch.nn.Module):
    """ Paraformer: Fast and Accurate Parallel Transformer for
        Non-autoregressive End-to-End Speech Recognition
        see https://arxiv.org/pdf/2206.08317.pdf

    """

    def __init__(self,
                 whisper_model,
                 custom_token_path: str,
                 sampler: bool = True,
                 sampling_ratio: float = 0.75
                 ):
        super().__init__()
        self.cif_predictor = Cif(whisper_model.decoder.n_state)
        self.whisper_model = whisper_model
        self.tokenzier = CustomTokenizer(custom_token_path)
        self.whisper_model.decoder.token_embedding = nn.Embedding(self.tokenizer.vocab_size, self.whisper_model.decoder.n_state)
        
        self.decoder_criterion = LabelSmoothingLoss(
            ignore_index=self.tokenizer.pad, label_smoothing=0.1, reduction="sum"
        )
        # self.decoder_criterion = LabelSmoothingLoss(
        #     ignore_index=50256, label_smoothing=0.1, reduction="sum"
        # )
        # tokenizer = whisper.tokenizer.get_tokenizer(
        #     whisper_model.is_multilingual,
        #     num_languages=whisper_model.num_languages,
        #     language="zh",
        #     task="transcribe",
        # )
        # if not sampler:
        #     self.tokenizer = tokenizer
        # custom_dict, custom_index_set, suppress_index_list = load_new_tokens_dict_list(custom_token_path, tokenizer)
        # self.suppress_tokens_list = suppress_index_list
        # self.custom_dict = custom_dict


        self.sampler = sampler
        self.sampling_ratio = sampling_ratio

    @torch.jit.ignore(drop=True)
    def forward(
        self,
        feature: torch.Tensor,
        feature_len: torch.Tensor,
        prev_outputs_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        target_lengths: torch.Tensor,
        is_training: bool,
    ):
        """Frontend + Encoder + Predictor + Decoder + Calc loss
        """
        # ignore the first 3 tokens, which are always <|lang_id|>, <|transcibe|>, <|notimestampes|>
        ignore_prefix_size = 3
        with torch.set_grad_enabled(is_training):
            encoder_out = self.whisper_model.encoder(feature)
            # encoder_out_mask is feature_len // 2
            encoder_out_len = feature_len // 2
            encoder_out_mask = make_non_pad_mask(encoder_out_len, encoder_out.shape[1]).unsqueeze(1)  # (B, 1, T)
            acoustic_embd, token_num, _, _ = self.cif_predictor(
                encoder_out, mask=encoder_out_mask, target_label_length=target_lengths.to(encoder_out.device))

            # 2 decoder with sampler
            # TODO(Mddct): support mwer here
            acoustic_embd = self._sampler(
                encoder_out,
                prev_outputs_tokens,
                target_tokens,
                target_lengths,
                acoustic_embd,
            )
            target_lengths = target_lengths.to(encoder_out.device)
            loss_quantity = torch.nn.functional.l1_loss(
                token_num,
                target_lengths.to(token_num.dtype),
                reduction='sum',
            )
            # print("token_num", token_num, "target_lengths", target_lengths)
            loss_quantity = loss_quantity / target_lengths.sum().to(token_num.dtype)


            text_logits = self.whisper_model.decoder(acoustic_embd, encoder_out)
            # text_logits = suppress_tokens(text_logits, self.suppress_tokens_list, -10000)
            # text_logits = text_logits[:, ignore_prefix_size:, :]
            # target_tokens = target_tokens[:, ignore_prefix_size:]
            loss_decoder = self.decoder_criterion(text_logits, target_tokens.to(text_logits.device))

            loss = loss_decoder + 100 * loss_quantity
        assert loss.requires_grad == is_training

        return (loss, loss_decoder, loss_quantity)

    @torch.jit.ignore(drop=True)
    def _sampler(self, encoder_out, prev_outputs_tokens, ys_pad, ys_pad_lens,
                 pre_acoustic_embeds):
        device = encoder_out.device
        B, _ = ys_pad.size()

        tgt_mask = make_non_pad_mask(ys_pad_lens)
        # zero the ignore id
        #ys_pad = ys_pad * tgt_mask
        #ys_pad_embed = self.embed(ys_pad)  # [B, T, L]
        ys_pad = ys_pad.to(encoder_out.device)
        tgt_mask = tgt_mask.to(encoder_out.device)
        prev_outputs_tokens = prev_outputs_tokens.to(encoder_out.device)
        # ys_pad_embed = self.whisper_model.decoder.token_embedding(ys_pad)  # ??? why not ys_pad_in
        ys_pad_embed = self.whisper_model.decoder.token_embedding(prev_outputs_tokens)
        with torch.no_grad():
            # decoder_out, _, _ = self.decoder(encoder_out, encoder_out_mask,
            #                                  pre_acoustic_embeds, ys_pad_lens)
            decoder_out = self.whisper_model.decoder(pre_acoustic_embeds, encoder_out)
            # decoder_out = suppress_tokens(decoder_out, self.suppress_tokens_list, -10000)
            pred_tokens = decoder_out.argmax(-1)

            nonpad_positions = tgt_mask
            same_num = ((pred_tokens == ys_pad) * nonpad_positions).sum(1)
            input_mask = torch.ones_like(
                nonpad_positions,
                device=device,
                dtype=tgt_mask.dtype,
            )
            for li in range(B):
                target_num = (ys_pad_lens[li] -
                              same_num[li].sum()).float() * self.sampling_ratio
                target_num = target_num.long()
                if target_num > 0:
                    input_mask[li].scatter_(
                        dim=0,
                        index=torch.randperm(ys_pad_lens[li],
                                             device=device)[:target_num],
                        value=0,
                    )
            input_mask = torch.where(input_mask > 0, 1, 0)
            input_mask = input_mask * tgt_mask
            input_mask_expand = input_mask.unsqueeze(2)  # [B, T, 1]

        sematic_embeds = torch.where(input_mask_expand == 1,
                                     pre_acoustic_embeds, ys_pad_embed)
        # zero out the paddings
        return sematic_embeds * tgt_mask.unsqueeze(2)

    def decode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        oracle_target_label_length: torch.Tensor=None,
    ) -> Dict[str, torch.Tensor]:
        def remove_non_chinese(text):
            # Define a pattern for matching Chinese characters
            # This pattern matches characters in the range of Unicode Chinese characters
            pattern = re.compile(r'[^\u4e00-\u9fff]+')
            
            # Replace all non-Chinese characters with an empty string
            filtered_text = pattern.sub('', text)
            
            return filtered_text
            # encoder
        encoder_out = self.whisper_model.encoder(speech)
        encoder_out_len = speech_lengths // 2
        encoder_out_mask = make_non_pad_mask(encoder_out_len, encoder_out.shape[1]).unsqueeze(1)  # (B, 1, T)
        encoder_out_mask = encoder_out_mask.to(encoder_out.device)
        # cif predictor
        # convert encoder_out to torch.float32
        encoder_out = encoder_out.to(dtype=torch.float32)
        if oracle_target_label_length is not None:
            oracle_target_label_length = oracle_target_label_length.to(encoder_out.device)
        acoustic_embed, token_num, _, _,= self.cif_predictor(
            encoder_out, mask=encoder_out_mask, target_label_length=oracle_target_label_length)
        token_num = token_num.floor().to(speech_lengths.dtype)

        # decoder
        # decoder_out = self.whisper_model.decoder(encoder_out, encoder_out_mask,
        #                                  acoustic_embed, token_num)
        decoder_out = self.whisper_model.decoder(acoustic_embed, encoder_out)
        # decoder_out = suppress_tokens(decoder_out, self.suppress_tokens_list)

        pred = decoder_out.argmax(dim=-1)
        pred = pred.tolist()
        hyps = []
        for i, tokens in enumerate(pred):
            # keep tokens until first 50257 sos token
            # if 50257 in tokens and tokens.index(50257) > 4:
            #     pred_tokens = tokens[: tokens.index(50257)]
            # else:
            #     pred_tokens = tokens
            pred_tokens = tokens
            hyp = self.tokenizer.decode(pred_tokens)
            print(hyp, pred_tokens, acoustic_embed.shape, token_num)
            s = re.sub(r'<\|.*?\|>', '', hyp)
            # s = remove_non_chinese(s)
            hyps.append(s)
        return hyps

def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


# def suppress_tokens(logits, suppress_tokens_list, suppress_value=None) -> None:
#     if suppress_value is None:
#         suppress_value = float('-inf')
#     else:
#         suppress_value = float(suppress_value)
#     logits[:, :, suppress_tokens_list] = suppress_value
#     return logits
