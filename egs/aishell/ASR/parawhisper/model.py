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
from torch import nn
from cif import Cif
from label_smoothing import LabelSmoothingLoss
from whisper_tokens import load_new_tokens_dict_list
from custom_tokens import CustomTokenizer
from ctc import CTC

class ParaWhisper(torch.nn.Module):
    """ Paraformer: Fast and Accurate Parallel Transformer for
        Non-autoregressive End-to-End Speech Recognition
        see https://arxiv.org/pdf/2206.08317.pdf

    """

    def __init__(self,
                 whisper_model,
                 custom_token_path: str,
                 sampler: bool = True,
                 sampling_ratio: float = 0.75,
                 ctc: bool = False,
                 ):
        super().__init__()
        self.cif_predictor = Cif(whisper_model.dims.n_audio_state)
        self.whisper_model = whisper_model
        # self.tokenizer = CustomTokenizer(custom_token_path)
        # self.whisper_model.decoder.token_embedding = torch.nn.Embedding(self.whisper_model.dims.n_vocab, self.whisper_model.dims.n_text_state)
        
        # self.decoder_criterion = LabelSmoothingLoss(
        #     ignore_index=self.tokenizer.pad, label_smoothing=0.1, reduction="sum"
        # )
        self.pad_id = 50256
        self.decoder_criterion = LabelSmoothingLoss(
            ignore_index=self.pad_id, label_smoothing=0.1, reduction="sum"
        )
        tokenizer = whisper.tokenizer.get_tokenizer(
            whisper_model.is_multilingual,
            num_languages=whisper_model.num_languages,
            language="zh",
            task="transcribe",
        )
        if not sampler:
            self.tokenizer = tokenizer

        # custom_dict, custom_index_set, suppress_index_list = load_new_tokens_dict_list(custom_token_path, tokenizer)
        # self.suppress_tokens_list = suppress_index_list
        # self.custom_dict = custom_dict


        self.sampler = sampler
        self.sampling_ratio = sampling_ratio

        # if ctc:
        #     self.ctc = CTC(
        #         odim=whisper_model.dims.n_vocab,
        #         encoder_output_size=whisper_model.dims.n_audio_state,
        #         blank_id=50256,
        #     )

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


            text_logits = self.whisper_model.decoder.forward_nar(acoustic_embd, encoder_out)
            # text_logits = suppress_tokens(text_logits, self.suppress_tokens_list, -10000)
            # text_logits = text_logits[:, ignore_prefix_size:, :]
            # target_tokens = target_tokens[:, ignore_prefix_size:]
            loss_decoder = self.decoder_criterion(text_logits, target_tokens.to(text_logits.device))

            loss = loss_decoder + 100 * loss_quantity
        assert loss.requires_grad == is_training

        return (loss, loss_decoder, loss_quantity)

    def forward_nar_ar(
        self,
        feature: torch.Tensor,
        feature_len: torch.Tensor,
        prev_outputs_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        target_lengths: torch.Tensor,
        is_training: bool,
    ):
        """Frontend + Encoder + Predictor + Decoder + Calc loss

        loss: loss_quantity + loss_nar_decoder + loss_ar_decoder + loss_distill
        """
        with torch.set_grad_enabled(is_training):
            encoder_out = self.whisper_model.encoder(feature)
            prev_outputs_tokens = prev_outputs_tokens.to(encoder_out.device)
            text_ar_logits = self.whisper_model.decoder(prev_outputs_tokens, encoder_out)
            target_tokens = target_tokens.to(text_ar_logits.device)
            loss_ar_decoder = self.decoder_criterion(text_ar_logits, target_tokens)

            encoder_out_len = feature_len // 2
            encoder_out_mask = make_non_pad_mask(encoder_out_len, encoder_out.shape[1]).unsqueeze(1)  # (B, 1, T)
            acoustic_embd, token_num, _, _ = self.cif_predictor(
                encoder_out, mask=encoder_out_mask, target_label_length=target_lengths.to(encoder_out.device))
            target_lengths = target_lengths.to(encoder_out.device)
            loss_quantity = torch.nn.functional.l1_loss(
                token_num,
                target_lengths.to(token_num.dtype),
                reduction='sum',
            )
            loss_quantity = loss_quantity / target_lengths.sum().to(token_num.dtype)

            acoustic_embd = self._sampler(
                encoder_out,
                prev_outputs_tokens,
                target_tokens,
                target_lengths,
                acoustic_embd,
            )
            text_logits = self.whisper_model.decoder.forward_nar(acoustic_embd, encoder_out)
            loss_decoder = self.decoder_criterion(text_logits, target_tokens)

            # distillation loss between ar and nar decoder_logits, ar distribution is the target, and should be disabled the gradient
            temperature = 2.0
            if not is_training:
                temperature = 1.0
            student_distribution = nn.functional.log_softmax(text_logits / temperature, dim=-1)
            teacher_distribution = nn.functional.softmax(text_ar_logits / temperature, dim=-1)
            # disable the gradient of teacher_distribution
            teacher_distribution = teacher_distribution.detach()
            loss_distill = self.kl_divergence(teacher_distribution, student_distribution, target_tokens) * temperature**2

            loss = loss_decoder + 100 * loss_quantity + 1.0 * loss_ar_decoder + 1.0 * loss_distill
        assert loss.requires_grad == is_training
        return (loss, loss_decoder, loss_quantity, loss_ar_decoder, loss_distill)

    def forward_nar_ar_feature(
        self,
        feature: torch.Tensor,
        feature_len: torch.Tensor,
        prev_outputs_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        target_lengths: torch.Tensor,
        is_training: bool,
    ):
        """Frontend + Encoder + Predictor + Decoder + Calc loss

        loss: loss_quantity + loss_nar_decoder + loss_ar_decoder + loss_distill
        """
        with torch.set_grad_enabled(is_training):
            encoder_out = self.whisper_model.encoder(feature)
            prev_outputs_tokens = prev_outputs_tokens.to(encoder_out.device)
            text_ar_logits, text_ar_feats = self.whisper_model.decoder(prev_outputs_tokens, encoder_out, return_second_top_feature=True)
            target_tokens = target_tokens.to(text_ar_logits.device)
            loss_ar_decoder = self.decoder_criterion(text_ar_logits, target_tokens)

            encoder_out_len = feature_len // 2
            encoder_out_mask = make_non_pad_mask(encoder_out_len, encoder_out.shape[1]).unsqueeze(1)  # (B, 1, T)
            acoustic_embd, token_num, _, _ = self.cif_predictor(
                encoder_out, mask=encoder_out_mask, target_label_length=target_lengths.to(encoder_out.device))
            target_lengths = target_lengths.to(encoder_out.device)
            loss_quantity = torch.nn.functional.l1_loss(
                token_num,
                target_lengths.to(token_num.dtype),
                reduction='sum',
            )
            loss_quantity = loss_quantity / target_lengths.sum().to(token_num.dtype)

            acoustic_embd = self._sampler(
                encoder_out,
                prev_outputs_tokens,
                target_tokens,
                target_lengths,
                acoustic_embd,
            )
            text_logits, text_nar_feats = self.whisper_model.decoder.forward_nar(acoustic_embd, encoder_out, return_second_top_feature=True)
            loss_decoder = self.decoder_criterion(text_logits, target_tokens)

            # distillation loss between ar and nar second top feature, ar feats should be disabled the gradient, using smooth l1 loss
            loss_distill = self.smooth_l1_loss(text_nar_feats, text_ar_feats.detach(), target_tokens)

            loss = loss_decoder + 100 * loss_quantity + 1.0 * loss_ar_decoder + 1.0 * loss_distill
        assert loss.requires_grad == is_training
        return (loss, loss_decoder, loss_quantity, loss_ar_decoder, loss_distill)

    def forward_ctc_only(
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

        with torch.set_grad_enabled(is_training):
            encoder_out = self.whisper_model.encoder(feature)
            # encoder_out_mask is feature_len // 2
            encoder_out_len = feature_len // 2
            
            target_lengths = target_lengths.to(encoder_out.device)

            loss, _ = self.ctc(encoder_out, encoder_out_len, target_tokens, target_lengths)
            
        assert loss.requires_grad == is_training

        # make loss_decoder, loss_quantity to be 0 tensor
        loss_decoder = torch.zeros_like(loss)
        loss_quantity = torch.zeros_like(loss)

        return (loss, loss_decoder, loss_quantity)

    @torch.jit.ignore(drop=True)
    def _sampler(self, encoder_out, prev_outputs_tokens, ys_pad, ys_pad_lens,
                 pre_acoustic_embeds):
        device = encoder_out.device
        B, _ = ys_pad.size()

        tgt_mask = make_non_pad_mask(ys_pad_lens)
        if tgt_mask.shape[1] < ys_pad.shape[1]:
            tgt_mask = torch.cat([tgt_mask, torch.zeros(B, ys_pad.shape[1]-tgt_mask.shape[1], dtype=tgt_mask.dtype, device=tgt_mask.device)], dim=1)
        # zero the ignore id
        #ys_pad = ys_pad * tgt_mask
        #ys_pad_embed = self.embed(ys_pad)  # [B, T, L]
        tgt_mask = tgt_mask.to(encoder_out.device)
        # ys_pad_embed = self.whisper_model.decoder.token_embedding(ys_pad)  # ??? why not ys_pad_in
        ys_pad_embed = self.whisper_model.decoder.token_embedding(prev_outputs_tokens)
        with torch.no_grad():
            # decoder_out, _, _ = self.decoder(encoder_out, encoder_out_mask,
            #                                  pre_acoustic_embeds, ys_pad_lens)
            decoder_out = self.whisper_model.decoder.forward_nar(pre_acoustic_embeds, encoder_out)
            # decoder_out = suppress_tokens(decoder_out, self.suppress_tokens_list, -10000)
            pred_tokens = decoder_out.argmax(-1)

            nonpad_positions = tgt_mask
            same_num = ((pred_tokens == ys_pad) * nonpad_positions).sum(1)
            # same_num = ((pred_tokens == ys_pad)).sum(1)
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

        acoustic_embed, token_num, alphas, cif_peaks,= self.cif_predictor(
            encoder_out, mask=encoder_out_mask, target_label_length=oracle_target_label_length)
        # acoustic_embed, token_num, _, _,= self.cif_predictor(
        #     encoder_out, mask=encoder_out_mask)
        token_num = token_num.floor().to(speech_lengths.dtype)


        decoder_out = self.whisper_model.decoder.forward_nar(acoustic_embed, encoder_out)


        pred = decoder_out.argmax(dim=-1)
        pred = pred.tolist()
        hyps = []
        for i, tokens in enumerate(pred):
            # keep tokens until first 50257 sos token
            if 50257 in tokens and tokens.index(50257) > 4:
                pred_tokens = tokens[: tokens.index(50257)]
            else:
                pred_tokens = tokens
            # pred_tokens = tokens
            hyp = self.tokenizer.decode(pred_tokens)
            # print(hyp, pred_tokens, acoustic_embed.shape, token_num)
            s = re.sub(r'<\|.*?\|>', '', hyp)
            # s = remove_non_chinese(s)
            hyps.append(s)
        return hyps, pred

    def decode_cif_rescore(
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

        # acoustic_embed_oracle, token_num_oracle, alphas_oracle, cif_peaks_oracle,= self.cif_predictor(
        #     encoder_out, mask=encoder_out_mask, target_label_length=oracle_target_label_length)

        acoustic_embed, token_num, alphas, cif_peaks,= self.cif_predictor(
            encoder_out, mask=encoder_out_mask, target_label_length=None)
        acoustic_embed_plus, token_num_plus, _, _,= self.cif_predictor(
            encoder_out, mask=encoder_out_mask, target_label_length=token_num+1)
        acoustic_embed_minus, token_num_minus, _, _,= self.cif_predictor(
            encoder_out, mask=encoder_out_mask, target_label_length=token_num-1)

        print(acoustic_embed.shape, acoustic_embed_minus.shape, acoustic_embed_plus.shape, oracle_target_label_length)


        # acoustic_embed, token_num, _, _,= self.cif_predictor(
        #     encoder_out, mask=encoder_out_mask)
        token_num = token_num.floor().to(speech_lengths.dtype)


        decoder_out = self.whisper_model.decoder.forward_nar(acoustic_embed, encoder_out)
        pred = decoder_out.argmax(dim=-1)
        # get the scores of pred
        scores = torch.nn.functional.softmax(decoder_out, dim=-1)
        scores = scores.gather(2, pred.unsqueeze(2)).squeeze(2)
        # convert scores to a scalar
        scores = scores.sum(dim=1) / acoustic_embed.shape[1]
        scores = scores.tolist()    
        pred = pred.tolist()

        decoder_out = self.whisper_model.decoder.forward_nar(acoustic_embed_plus, encoder_out)
        pred_plus = decoder_out.argmax(dim=-1)
        # get the scores of pred
        scores_plus = torch.nn.functional.softmax(decoder_out, dim=-1)
        scores_plus = scores_plus.gather(2, pred_plus.unsqueeze(2)).squeeze(2)
        scores_plus = scores_plus.sum(dim=1) / acoustic_embed_plus.shape[1]
        scores_plus = scores_plus.tolist()    
        pred_plus = pred_plus.tolist()

        decoder_out = self.whisper_model.decoder.forward_nar(acoustic_embed_minus, encoder_out)
        pred_minus = decoder_out.argmax(dim=-1)
        # get the scores of pred
        scores_minus = torch.nn.functional.softmax(decoder_out, dim=-1)
        scores_minus = scores_minus.gather(2, pred_minus.unsqueeze(2)).squeeze(2)
        scores_minus = scores_minus.sum(dim=1) / acoustic_embed_minus.shape[1]
        scores_minus = scores_minus.tolist()
        pred_minus = pred_minus.tolist()

        print(scores, scores_minus, scores_plus, 2333333333333333333333333333333333333333333333333)


        # according to scores, select the best one from pred, pred_plus, pred_minus
        pred = [pred[i] if scores[i] >= scores_plus[i] and scores[i] >= scores_minus[i] else pred_plus[i] if scores_plus[i] >= scores[i] and scores_plus[i] >= scores_minus[i] else pred_minus[i] for i in range(len(pred))]


        hyps = []
        for i, tokens in enumerate(pred):
            # keep tokens until first 50257 sos token
            # if 50257 in tokens and tokens.index(50257) > 4:
            #     pred_tokens = tokens[: tokens.index(50257)]
            # else:
            #     pred_tokens = tokens
            pred_tokens = tokens
            hyp = self.tokenizer.decode(pred_tokens)
            # print(hyp, pred_tokens, acoustic_embed.shape, token_num)
            s = re.sub(r'<\|.*?\|>', '', hyp)
            # s = remove_non_chinese(s)
            hyps.append(s)
        return hyps, pred
    
    def ctc_decode(self, speech: torch.Tensor, speech_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoder_out = self.whisper_model.encoder(speech)
        encoder_out_len = speech_lengths // 2
        encoder_out_len = encoder_out_len.to(encoder_out.device)
        encoder_out = encoder_out.to(dtype=torch.float32)
        ctc_probs = self.ctc.log_softmax(encoder_out)
        pred_tokens = ctc_greedy_search(ctc_probs, encoder_out_len, blank_id=50256, eot_id=self.tokenizer.eot)
        hyps = []
        for tokens in pred_tokens:
            hyp = self.tokenizer.decode(tokens)
            print(hyp, tokens)
            hyps.append(hyp)
        return hyps, pred_tokens
   
    def kl_divergence(self, target_distribution, log_predicted_distribution, labels):
        kl_loss = nn.KLDivLoss(reduction="none")
        divergence = kl_loss(log_predicted_distribution, target_distribution)
        # ignore padded tokens from divergence, i.e. where labels are not set to -100
        padding_mask = labels != self.pad_id
        padding_mask = padding_mask.unsqueeze(-1)
        divergence = divergence * padding_mask
        # take the average over the mini-batch
        divergence = divergence.sum() / padding_mask.sum()
        return divergence
    
    def smooth_l1_loss(self, inputs, target, labels):
        criterion = nn.SmoothL1Loss(reduction="none")
        loss = criterion(inputs, target)
        padding_mask = labels != self.pad_id
        padding_mask = padding_mask.unsqueeze(-1)
        loss = loss * padding_mask
        # take the average over the mini-batch
        loss = loss.sum() / padding_mask.sum()
        return loss

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

def ctc_greedy_search(ctc_probs: torch.Tensor,
                      ctc_lens: torch.Tensor,
                      blank_id: int = 50256,
                      eot_id: int = 50257) -> List[List[int]]:
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.size(1)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    hyps = [remove_duplicates_and_blank(hyp, blank_id, eot_id) for hyp in hyps]
    return hyps

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
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
    return mask


def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int,
                                eos: int,) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id and hyp[cur] != eos:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp