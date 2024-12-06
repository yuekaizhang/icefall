import os
from typing import Optional

import torch
import torchaudio
from compute_neural_codec_and_prepare_text_tokens import AudioTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
    """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
    the mask is set to -1, and otherwise setting to the value detailed in the mask."""
    seq_len = input_ids.shape[-1]
    decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
    input_ids = torch.where(
        decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask
    )
    return input_ids


def build_delay_pattern_mask(
    input_ids: torch.LongTensor,
    bos_token_id: int,
    pad_token_id: int,
    max_length: int = 8,
):
    """Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
    one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
    are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,
    seq_len)`:
    - [B, -1, -1, -1, P, P, P, P]
    - [B, B, -1, -1, -1, P, P, P]
    - [B, B, B, -1, -1, -1, P, P]
    - [B, B, B, B, -1, -1, -1, P]
    where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
    a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
    mask is set to the value in the prompt:
    - [B, a, b, -1, P, P, P, P]
    - [B, B, c, d, -1, P, P, P]
    - [B, B, B, e, f, -1, P, P]
    - [B, B, B, B, g, h, -1, P]
    where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
    tokens in our prediction.
    Modified from https://github.com/huggingface/parler-tts/blob/main/parler_tts/modeling_parler_tts.py#L213.
    """
    # pad the input_ids with num_codebooks - 1 padding tokens

    bsz, num_codebooks, seq_len = input_ids.shape
    input_ids_shifted = (
        torch.ones(
            (bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device
        )
        * -1
    )

    # fill the shifted ids with the prompt entries, offset by the codebook idx
    for codebook in range(num_codebooks):
        # mono channel - loop over the codebooks one-by-one
        # + 1 to account for the bos token for the first codebook
        input_ids_shifted[
            :, codebook, codebook + 1 : seq_len + codebook + 1
        ] = input_ids[:, codebook]

    # construct a pattern mask that indicates the positions of padding tokens for each codebook
    # first fill the upper triangular part (the EOS padding)
    eos_delay_pattern = torch.triu(
        torch.ones((num_codebooks, max_length), dtype=torch.bool),
        diagonal=max_length - num_codebooks,
    )
    # then fill the lower triangular part (the BOS padding)
    bos_delay_pattern = torch.tril(
        torch.ones((num_codebooks, max_length), dtype=torch.bool)
    )
    bos_mask = ~(bos_delay_pattern).to(input_ids.device)
    eos_mask = ~(eos_delay_pattern).to(input_ids.device)
    mask = ~(bos_delay_pattern + eos_delay_pattern).to(input_ids.device)

    input_ids_shifted = (
        mask * input_ids_shifted + ~bos_mask * bos_token_id + ~eos_mask * pad_token_id
    )
    pattern_mask = input_ids_shifted

    new_seq_len = (
        seq_len + 1 + 1 + num_codebooks - 1
    )  # +1 for the bos token, +1 for the pad token
    input_ids = input_ids_shifted[..., :new_seq_len]
    # fill the -1 with the pad token id
    input_ids = torch.where(input_ids == -1, pad_token_id, input_ids)
    return input_ids, pattern_mask


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

        self.llm = AutoModelForCausalLM.from_pretrained(model_path)
        if freeze_llm:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm.config.pad_token_id = 151643

        self.num_codebooks = 8
        self.codebook_size = 1024
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size + 1, self.llm.config.hidden_size)
                for _ in range(self.num_codebooks)
            ]
        )
        self.audio_lm_heads = nn.ModuleList(
            [
                nn.Linear(self.llm.config.hidden_size, self.codebook_size + 1)
                for _ in range(self.num_codebooks)
            ]
        )

        self.loss_fct = CrossEntropyLoss()
        self.audio_accuracy_metric = MulticlassAccuracy(
            self.codebook_size + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=IGNORE_TOKEN_ID,
        )
        if post_adapter:
            self.post_adapter = nn.ModuleList(
                [
                    MLP(self.llm.config.hidden_size, 2048, self.llm.config.hidden_size)
                    for _ in range(2)
                ]
            )
        else:
            self.post_adapter = None

        self.audio_tokenizer = AudioTokenizer()

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
        audio_codes_origin = audio_codes.clone()
        print("The original audio codes are", audio_codes[0].shape, audio_codes[0])
        # WAR: 16 is a heuristic value
        audio_codes = build_delay_pattern_mask(
            audio_codes,
            bos_token_id=self.codebook_size,
            pad_token_id=self.codebook_size,
            max_length=audio_codes.shape[-1] + 16,
        )[0]
        print("The audio codes are", audio_codes[0].shape, audio_codes[0])
        # change -23 in the original code to self.codebook_size
        audio_codes = audio_codes.masked_fill(audio_codes == -23, self.codebook_size)

        # mask input prompt
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        text_labels = input_ids.clone()
        audio_codes_list = []

        for i in range(len(input_ids)):
            assistant_index = torch.where(input_ids[i] == assistant_id)[0]
            assert assistant_index > 0
            text_labels[i, : assistant_index + 1] = IGNORE_TOKEN_ID
            num_shift_token = assistant_index + 1

            # now we could left pad the audio codes to make sure the real codes start from the tokens after the assistant token
            assert audio_codes[i].shape[0] == self.num_codebooks
            audio_code = torch.cat(
                [
                    torch.full(
                        (audio_codes[i].shape[0], num_shift_token),
                        self.codebook_size,
                        dtype=audio_codes.dtype,
                        device=audio_codes.device,
                    ),
                    audio_codes[i],
                ],
                dim=-1,
            )
            audio_codes_list.append(audio_code)

        # now we could right pad the audio_codes_list to get the audio_codes
        max_seq_len = max([audio_code.shape[-1] for audio_code in audio_codes_list])
        audio_codes = torch.stack(
            [
                torch.cat(
                    [
                        audio_code,
                        torch.full(
                            (self.num_codebooks, max_seq_len - audio_code.shape[-1]),
                            self.codebook_size,
                            dtype=audio_codes.dtype,
                            device=audio_codes.device,
                        ),
                    ],
                    dim=-1,
                )
                for audio_code in audio_codes_list
            ]
        )

        audio_labels = audio_codes.clone()
        audio_labels = audio_labels.masked_fill(
            audio_labels == self.codebook_size, IGNORE_TOKEN_ID
        )

        seq_len = audio_codes.shape[-1]
        # pad the input_ids to the maximum length of the audio codes with pad token
        input_ids = torch.cat(
            [
                input_ids,
                torch.full(
                    (input_ids.shape[0], seq_len - input_ids.shape[1]),
                    self.llm.config.pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.full(
                    (attention_mask.shape[0], seq_len - attention_mask.shape[1]),
                    False,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        text_labels = torch.cat(
            [
                text_labels,
                torch.full(
                    (text_labels.shape[0], seq_len - text_labels.shape[1]),
                    self.llm.config.pad_token_id,
                    dtype=text_labels.dtype,
                    device=text_labels.device,
                ),
            ],
            dim=1,
        )

        text_labels[text_labels == self.llm.config.pad_token_id] = IGNORE_TOKEN_ID

        inputs_text_embeds = self.llm.get_input_embeddings()(input_ids)

        audio_inputs_embeds = sum(
            [
                self.embed_tokens[codebook](audio_codes[:, codebook])
                for codebook in range(self.num_codebooks)
            ]
        )

        inputs_embeds = inputs_text_embeds + audio_inputs_embeds
        inputs_embeds /= self.num_codebooks + 1
        # inputs_embeds = self.embed_tokens[0](audio_codes[:, 0])

        model_outputs = self.llm(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=text_labels,
            return_dict=True,
            output_hidden_states=True,
        )
        model_outputs_text = model_outputs
        # model_outputs_text = self.llm(
        #     attention_mask=attention_mask,
        #     inputs_embeds=inputs_text_embeds,
        #     labels=text_labels,
        #     return_dict=True,
        #     output_hidden_states=True,
        # )
        if debug:
            with torch.no_grad():
                preds = torch.argmax(model_outputs_text.logits, -1)
                acc = compute_accuracy(
                    preds.detach()[:, :-1],
                    text_labels.detach()[:, 1:],
                    ignore_label=IGNORE_TOKEN_ID,
                )

                pad_mask = torch.where(text_labels[0] != IGNORE_TOKEN_ID)
                pred_text = preds[0][pad_mask]
                target_text = text_labels[0][pad_mask]
                print(f"the text is {self.tokenizer.decode(pred_text)}")
                print(f"the label is {self.tokenizer.decode(target_text)}")

        last_hidden_state = model_outputs.hidden_states[-1].clone()
        if self.post_adapter is not None:
            last_hidden_state = self.post_adapter[0](last_hidden_state)
            last_hidden_state = self.post_adapter[1](last_hidden_state)
        audio_logits = [
            self.audio_lm_heads[codebook](last_hidden_state).float()
            for codebook in range(self.num_codebooks)
        ]

        # get pred audio codes from audio_logits
        audio_logits_shifted = [
            audio_logit[..., :-1, :].contiguous() for audio_logit in audio_logits
        ]
        audio_codes_pred = [
            torch.argmax(audio_logit, dim=-1) for audio_logit in audio_logits_shifted
        ]
        audio_codes_pred = torch.stack(audio_codes_pred, dim=1)
        print(
            "The audio codes pred are",
            audio_codes_pred[0].shape,
            audio_codes_pred[0][:, :10],
        )
        audio_codes_real = torch.zeros_like(audio_codes_pred)
        for i in range(self.num_codebooks):
            seq_len = audio_codes_pred.shape[-1] - 8
            audio_codes_real[:, i, :seq_len] = audio_codes_pred[:, i, i : i + seq_len]
        audio_codes_real = audio_codes_real[:, :, :seq_len]
        audio_codes_first_level_ground = audio_codes_real.clone()
        for i in range(len(audio_codes_real)):
            audio_codes_first_level_ground[
                i, 0, : audio_codes_lens[i]
            ] = audio_codes_origin[i, 0, : audio_codes_lens[i]]
            self.save_audio(
                audio_codes_real[i][: audio_codes_lens[i]], f"audio_codes_real_{i}.wav"
            )
            self.save_audio(
                audio_codes_pred[i][: audio_codes_lens[i]], f"audio_codes_pred_{i}.wav"
            )
            self.save_audio(
                audio_codes_first_level_ground[i][: audio_codes_lens[i]],
                f"audio_codes_first_level_ground_{i}.wav",
            )
        exit(0)

        # last hidden state shape (batch_size, sequence_length, hidden_size)
        # audio_logits shape (batch_size, sequence_length, codebook_size)
        # audio_labels shape (batch_size, sequence_length, num_codebooks)
        audio_labels = audio_labels.transpose(1, 2)
        total_loss = [model_outputs_text.loss]
        top_k_acc_list = [acc]

        for i in range(self.num_codebooks):
            shift_logits = audio_logits[i][..., :-1, :].contiguous()
            shift_labels = audio_labels[..., 1:, i].contiguous()

            shift_logits = shift_logits.view(-1, self.codebook_size + 1)
            shift_labels = shift_labels.view(-1)

            loss = self.loss_fct(shift_logits, shift_labels)
            total_loss.append(loss)

            if debug:
                top_k_acc = self.audio_accuracy_metric(
                    shift_logits.detach(), shift_labels
                ).item()
                top_k_acc_list.append(top_k_acc)

        total_loss_value = text_loss_scale * total_loss[0] + sum(total_loss[1:])
        return total_loss_value, total_loss, top_k_acc_list


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
