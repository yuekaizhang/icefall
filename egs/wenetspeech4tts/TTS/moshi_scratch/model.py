import os
from typing import Optional

import torch
import torchaudio
from compute_neural_codec_and_prepare_text_tokens import AudioTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from transformers.trainer_pt_utils import LabelSmoother

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class SPEECH_LLM(nn.Module):
    """
    The Speech-to-Text model. It consists of an encoder, a language model and an encoder projector.
    The encoder is used to extract speech features from the input speech signal.
    The encoder projector is used to project the encoder outputs to the same dimension as the language model.
    The language model is used to generate the text from the speech features.
    Args:
        encoder (:obj:`nn.Module`): The encoder module.
        llm (:obj:`nn.Module`): The language model module.
        encoder_projector (:obj:`nn.Module`): The encoder projector module.
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        llm: nn.Module = None,
        encoder_projector: nn.Module = None,
        model_path: str = "/workspace/Qwen2.5-0.5B-Instruct",
        freeze_llm: bool = False,
        post_adapter: bool = False,
    ):
        super().__init__()

        self.llm = Qwen2ForCausalLM(config=self.get_config())

        self.num_codebooks = 8
        self.codebook_size = 1024

        self.audio_bos_token_id = self.codebook_size
        self.audio_eos_token_id = self.codebook_size + 1
        self.audio_pad_token_id = self.codebook_size + 2

        self.audio_vocab_size = self.codebook_size + 3
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(self.audio_vocab_size, self.llm.config.hidden_size)
                for _ in range(self.num_codebooks)
            ]
        )
        self.audio_lm_heads = nn.ModuleList(
            [
                nn.Linear(self.llm.config.hidden_size, self.audio_vocab_size)
                for _ in range(self.num_codebooks)
            ]
        )

        self.loss_fct = CrossEntropyLoss()
        self.audio_accuracy_metric = MulticlassAccuracy(
            self.audio_vocab_size,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=IGNORE_TOKEN_ID,
        )

        self.audio_tokenizer = AudioTokenizer()

    def get_config(self):
        return Qwen2Config(
            vocab_size=5000,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=2048,
            max_position_embeddings=4096,
        )

    def save_audio(self, audio_codes, path):
        audio_code = audio_codes.unsqueeze(0)
        audio_code = audio_code.to(torch.int64)
        samples_org = self.audio_tokenizer.decode([(audio_code, None)])
        torchaudio.save(path, samples_org[0].cpu(), 24000)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        audio_codes_lens: Optional[torch.LongTensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
        text_loss_scale: float = 1.0,
        debug: bool = True,
    ):
        """
        input_ids: (batch_size, sequence_length)
        audio_codes: (batch_size, num_codebooks, sequence_length)
        text_labels: (batch_size, sequence_length)
        audio_labels: (batch_size, num_codebooks, sequence_length)
        """
        # change the -23 pad token to the eos token id
        # save audios according to audio_codes_lens
        # print(audio_codes.shape)
        # for i in range(audio_codes.shape[0]):
        #     audio_code = audio_codes[i][:, : audio_codes_lens[i]]
        #     #print(audio_code.shape)
        #     self.save_audio(audio_code, f"real_audio_ground_{i}.wav")
        audio_codes = audio_codes.to(torch.int64)
        audio_codes = audio_codes.masked_fill(
            audio_codes == -23, self.audio_pad_token_id
        )

        # add bos token to the audio codes
        audio_codes_input = torch.cat(
            [
                torch.full(
                    (audio_codes.shape[0], audio_codes.shape[1], 1),
                    self.audio_bos_token_id,
                    dtype=audio_codes.dtype,
                    device=audio_codes.device,
                ),
                audio_codes,
            ],
            dim=-1,
        )
        # add eos token to the audio labels
        audio_labels = torch.cat(
            [
                audio_codes,
                torch.full(
                    (audio_codes.shape[0], audio_codes.shape[1], 1),
                    self.audio_pad_token_id,
                    dtype=audio_codes.dtype,
                    device=audio_codes.device,
                ),
            ],
            dim=-1,
        )
        for i in range(len(audio_codes_lens)):
            audio_labels[i, :, audio_codes_lens[i]] = self.audio_eos_token_id

        # build the delay pattern audio codes and audio labels
        # delayed shape is (batch_size, num_codebooks, sequence_length+num_codebooks-1)
        delayed_audio_inputs = torch.full(
            (
                audio_codes.shape[0],
                audio_codes.shape[1],
                audio_codes_input.shape[2] + self.num_codebooks - 1,
            ),
            self.audio_pad_token_id,
            dtype=audio_codes.dtype,
            device=audio_codes.device,
        )
        delayed_audio_labels = delayed_audio_inputs.clone()
        for i in range(self.num_codebooks):
            delayed_audio_inputs[
                :, i, i : i + audio_codes_input.shape[2]
            ] = audio_codes_input[:, i]
            delayed_audio_labels[:, i, i : i + audio_labels.shape[2]] = audio_labels[
                :, i
            ]

        inputs_text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_audio_embeds = sum(
            [
                self.embed_tokens[codebook](delayed_audio_inputs[:, codebook])
                for codebook in range(self.num_codebooks)
            ]
        )
        # construct inputs_audio_attention_mask from audio_codes_lens, now we add the bos token and self.num_codebooks-1 padding tokens
        audio_codes_lens_pad = audio_codes_lens + 1 + self.num_codebooks - 1
        # inputs_audio_attention_mask has shape (batch_size, sequence_length)
        inputs_audio_attention_mask = torch.arange(
            inputs_audio_embeds.shape[1], device=inputs_audio_embeds.device
        ).expand(inputs_audio_embeds.shape[0], -1) < audio_codes_lens_pad.unsqueeze(
            1
        ).to(
            inputs_audio_embeds.device
        )

        # concatenate the text and audio embeddings
        inputs_embeds = torch.cat([inputs_text_embeds, inputs_audio_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, inputs_audio_attention_mask], dim=1)

        model_outputs = self.llm(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            output_hidden_states=True,
        )
        last_hidden_state = model_outputs.hidden_states[-1].clone()

        audio_logits = [
            self.audio_lm_heads[codebook](last_hidden_state).float()
            for codebook in range(self.num_codebooks)
        ]

        # stack the audio logits
        # audio_logits shape (batch_size, sequence_length, vocab_size, num_codebooks)
        audio_logits = torch.stack(audio_logits, dim=-1)
        # get the audio part only
        audio_logits = audio_logits[:, inputs_text_embeds.shape[1] :]

        # get the real audio from audio_logits
        # real_audio_index = audio_logits.argmax(dim=-2)
        # # real_audio_index shape (batch_size, sequence_length, num_codebooks)
        # real_audio_index = real_audio_index.transpose(1, 2)
        # real_audio_code = torch.full(
        #     (
        #         real_audio_index.shape[0],
        #         self.num_codebooks,
        #         real_audio_index.shape[2] - self.num_codebooks + 1 - 1,
        #     ),
        #     self.audio_pad_token_id,
        #     dtype=real_audio_index.dtype,
        #     device=real_audio_index.device,
        # )
        # for i in range(self.num_codebooks):
        #     real_audio_code[:, i, :] = real_audio_index[
        #         :, i, i : i + real_audio_code.shape[2]
        #     ]
        # now save the real audio
        # print(233333333333333333333333333)
        # for i in range(real_audio_code.shape[0]):
        #     # get the first index of eos token, then slice the real_audio_code
        #     eos_index = (real_audio_code[i] == self.audio_eos_token_id).nonzero(as_tuple=True)[1][0]
        #     real_audio_code_save = real_audio_code[i, :, :eos_index - 1]
        #     print(real_audio_code_save.shape, real_audio_code[i].shape, eos_index)
        #     self.save_audio(real_audio_code_save, f"real_audio_checkpoint_{i}.wav")

        #     # real_audio_index_save = real_audio_index[i, :, 7:- 7]
        #     # print(real_audio_index_save.shape, real_audio_index[i].shape, eos_index)
        #     # self.save_audio(real_audio_index_save, f"real_audio_index_checkpoint_{i}.wav")
        # exit(0)

        total_loss = []
        top_k_acc_list = []

        # replace the delay audio labels' pad token with the IGNORE_TOKEN_ID
        delayed_audio_labels = delayed_audio_labels.masked_fill(
            delayed_audio_labels == self.audio_pad_token_id, IGNORE_TOKEN_ID
        )

        for i in range(self.num_codebooks):
            logits = audio_logits[:, :, :, i]
            labels = delayed_audio_labels[:, i]

            logits = logits.contiguous().view(-1, self.audio_vocab_size)
            labels = labels.contiguous().view(-1)

            loss = self.loss_fct(logits, labels)
            total_loss.append(loss)
            if debug:
                top_k_acc = self.audio_accuracy_metric(logits.detach(), labels).item()
                top_k_acc_list.append(top_k_acc)

        total_loss_value = sum(total_loss)
        return total_loss_value, total_loss, top_k_acc_list

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        prompt_audio_codes: Optional[torch.Tensor] = None,
        audio_path_list: Optional[str] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        max_length: int = 1024,
    ):
        """
        input_ids: (batch_size, sequence_length)
        audio_codes: (batch_size, num_codebooks, sequence_length)
        """
        # change the -23 pad token to the eos token id
        if prompt_audio_codes is None:
            audio_codes_input = torch.full(
                (input_ids.shape[0], self.num_codebooks, 1),
                self.audio_bos_token_id,
                dtype=torch.int64,
                device=input_ids.device,
            )
        else:
            audio_codes = prompt_audio_codes.to(torch.int64)
            audio_codes = audio_codes.masked_fill(
                audio_codes == -23, self.audio_pad_token_id
            )

            # add bos token to the audio codes
            audio_codes_input = torch.cat(
                [
                    torch.full(
                        (audio_codes.shape[0], audio_codes.shape[1], 1),
                        self.audio_bos_token_id,
                        dtype=audio_codes.dtype,
                        device=audio_codes.device,
                    ),
                    audio_codes,
                ],
                dim=-1,
            )

        # build the delay pattern audio codes and audio labels
        # delayed shape is (batch_size, num_codebooks, sequence_length+num_codebooks-1)
        delayed_audio_inputs = torch.full(
            (
                audio_codes_input.shape[0],
                audio_codes_input.shape[1],
                audio_codes_input.shape[2] + self.num_codebooks - 1,
            ),
            self.audio_pad_token_id,
            dtype=audio_codes_input.dtype,
            device=audio_codes_input.device,
        )
        # delayed_audio_labels = delayed_audio_inputs.clone()
        for i in range(self.num_codebooks):
            delayed_audio_inputs[
                :, i, i : i + audio_codes_input.shape[2]
            ] = audio_codes_input[:, i]

        audio_codes_input_complete, audio_codes_input_partial = (
            delayed_audio_inputs[..., : audio_codes_input.shape[2]],
            delayed_audio_inputs[..., audio_codes_input.shape[2] :],
        )

        inputs_text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_audio_embeds = sum(
            [
                self.embed_tokens[codebook](audio_codes_input_complete[:, codebook])
                for codebook in range(self.num_codebooks)
            ]
        )
        # construct inputs_audio_attention_mask from audio_codes_lens, now we add the bos token and self.num_codebooks-1 padding tokens
        # audio_codes_lens_pad = audio_codes_lens + 1 + self.num_codebooks - 1
        # # inputs_audio_attention_mask has shape (batch_size, sequence_length)
        # inputs_audio_attention_mask = torch.arange(
        #     inputs_audio_embeds.shape[1], device=inputs_audio_embeds.device
        # ).expand(inputs_audio_embeds.shape[0], -1) < audio_codes_lens_pad.unsqueeze(1).to(
        #     inputs_audio_embeds.device
        # )

        # concatenate the text and audio embeddings
        # ！ WAR
        inputs_embeds = torch.cat(
            [inputs_text_embeds, inputs_audio_embeds[:, : -(self.num_codebooks - 1)]],
            dim=1,
        )
        # attention_mask = torch.cat(
        #     [attention_mask, inputs_audio_attention_mask], dim=1
        # )
        # TOOD: add attention_mask to the model
        cache = None
        generated_audio_tokens = []
        for i in range(max_length):
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=cache,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = outputs.past_key_values

            last_hidden_state = outputs.hidden_states[-1].clone()

            audio_logits = [
                self.audio_lm_heads[codebook](last_hidden_state).float()
                for codebook in range(self.num_codebooks)
            ]

            # get the last token logits
            audio_logits = torch.stack(audio_logits, dim=-1)
            audio_logits = audio_logits[:, -1]

            # audio_logits shape (batch_size, vocab_size, num_codebooks)
            audio_logits = (
                audio_logits.transpose(1, 2)
                .contiguous()
                .view(-1, self.audio_vocab_size)
            )
            assert audio_logits.shape[1] == self.audio_vocab_size
            token_ids = topk_sampling(
                audio_logits, top_k=-1, top_p=1.0, temperature=1.0
            )
            print(i, token_ids)
            if i < self.num_codebooks - 1:
                assert self.audio_eos_token_id not in token_ids
            token_ids = token_ids.view(-1, self.num_codebooks, 1)
            # token_ids shape (batch_size, num_codebooks, 1)
            # replace token_ids with groud truth
            if i < self.num_codebooks - 1:
                token_ids[:, i + 1 :, 0] = audio_codes_input_partial[:, i + 1 :, i]
            generated_audio_tokens.append(token_ids)

            # check if the eos token is generated
            inputs_embeds = sum(
                [
                    self.embed_tokens[codebook](token_ids[:, codebook])
                    for codebook in range(self.num_codebooks)
                ]
            )
            assert token_ids.shape[0] == 1
            if token_ids[0, 0] == self.audio_eos_token_id:
                break

        for i in range(self.num_codebooks - 2):
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=cache,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = outputs.past_key_values

            last_hidden_state = outputs.hidden_states[-1].clone()

            audio_logits = [
                self.audio_lm_heads[codebook](last_hidden_state).float()
                for codebook in range(self.num_codebooks)
            ]

            # get the last token logits
            audio_logits = torch.stack(audio_logits, dim=-1)
            audio_logits = audio_logits[:, -1]

            # audio_logits shape (batch_size, vocab_size, num_codebooks)
            audio_logits = (
                audio_logits.transpose(1, 2)
                .contiguous()
                .view(-1, self.audio_vocab_size)
            )
            assert audio_logits.shape[1] == self.audio_vocab_size
            token_ids = topk_sampling(
                audio_logits, top_k=-1, top_p=1.0, temperature=1.0
            )
            print(i, token_ids)
            token_ids = token_ids.view(-1, self.num_codebooks, 1)
            token_ids[:, : i + 1, :] = self.audio_pad_token_id
            token_ids[:, i + 1, :] = self.audio_eos_token_id
            # token_ids shape (batch_size, num_codebooks, 1)
            # replace token_ids with groud truth
            generated_audio_tokens.append(token_ids)

            # check if the eos token is generated
            inputs_embeds = sum(
                [
                    self.embed_tokens[codebook](token_ids[:, codebook])
                    for codebook in range(self.num_codebooks)
                ]
            )
        generated_audio_tokens = torch.cat(generated_audio_tokens, dim=-1)

        # real_audio_index shape (batch_size, sequence_length, num_codebooks)
        real_audio_index = generated_audio_tokens
        real_audio_code = torch.full(
            (
                real_audio_index.shape[0],
                self.num_codebooks,
                real_audio_index.shape[2] - self.num_codebooks + 1,
            ),
            self.audio_pad_token_id,
            dtype=real_audio_index.dtype,
            device=real_audio_index.device,
        )
        for i in range(self.num_codebooks):
            real_audio_code[:, i, :] = real_audio_index[
                :, i, i : i + real_audio_code.shape[2]
            ]
        # now save the real audio
        for i in range(real_audio_code.shape[0]):
            # get the first index of eos token, then slice the real_audio_code
            # if self.audio_eos_token_id in real_audio_code[i]:
            #     eos_index = (real_audio_code[i] == self.audio_eos_token_id).nonzero(
            #         as_tuple=True
            #     )[1][0]
            #     real_audio_code_save = real_audio_code[i, :, : eos_index - 1]
            #     print(real_audio_code_save.shape, real_audio_code[i].shape, eos_index)
            # else:
            real_audio_code_save = real_audio_code[i]
            self.save_audio(real_audio_code_save, audio_path_list[i])

        return


def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.
    Copied from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/utils/metric.py
    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()


def topk_sampling(
    logits,
    top_k=10,
    top_p=1.0,
    temperature=1.0,
):
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits_filtered = top_k_top_p_filtering(
        logits.clone(), top_k=top_k, top_p=top_p, min_tokens_to_keep=2
    )
    # Sample
    probs = torch.nn.functional.softmax(logits_filtered, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1)

    return tokens


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits
