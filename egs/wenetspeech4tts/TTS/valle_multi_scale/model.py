import os
from typing import Optional

import torch
import torchaudio
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

        self.num_codebooks = 8
        self.codebook_size = 1024

        self.audio_bos_token_id = self.codebook_size
        self.audio_eos_token_id = self.codebook_size + 1
        self.audio_pad_token_id = self.codebook_size + 2

        self.audio_vocab_size = self.codebook_size + 3

        self.temporal_llm = Qwen2ForCausalLM(config=self.get_config())
        self.depth_llm = Qwen2ForCausalLM(config=self.get_config(depth=True))
        # get the hidden size of the llm
        self.hidden_size = self.temporal_llm.config.hidden_size
        self.dep_hidden_size = self.depth_llm.config.hidden_size

        self.depth_linear_in = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.dep_hidden_size)
                for _ in range(self.num_codebooks)
            ]
        )
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(self.audio_vocab_size, self.hidden_size)
                for _ in range(self.num_codebooks)
            ]
        )
        self.dep_embed_tokens = nn.ModuleList(
            [
                nn.Embedding(self.audio_vocab_size, self.dep_hidden_size)
                for _ in range(self.num_codebooks)
            ]
        )
        self.audio_lm_heads = nn.ModuleList(
            [
                nn.Linear(self.dep_hidden_size, self.audio_vocab_size)
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

    def get_config(self, depth=False):
        if depth:
            return Qwen2Config(
                vocab_size=self.audio_vocab_size,
                hidden_size=256,
                num_hidden_layers=6,
                num_attention_heads=4,
                num_key_value_heads=4,
                intermediate_size=512,
                max_position_embeddings=self.num_codebooks,
            )
        return Qwen2Config(
            vocab_size=15000,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=2048,
            max_position_embeddings=4096,
        )

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

        depth_audio_codes_input = torch.cat(
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
            depth_audio_codes_input[i, :, audio_codes_lens[i]] = self.audio_bos_token_id

        depth_audio_codes_input = torch.cat(
            [
                torch.full(
                    (audio_codes.shape[0], 1, depth_audio_codes_input.shape[2]),
                    self.audio_bos_token_id,
                    dtype=audio_codes.dtype,
                    device=audio_codes.device,
                ),
                depth_audio_codes_input[
                    :, :-1
                ],  # we don't need the last level token as input
            ],
            dim=1,
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

        inputs_text_embeds = self.temporal_llm.get_input_embeddings()(input_ids)
        inputs_audio_embeds = sum(
            [
                self.embed_tokens[codebook](audio_codes_input[:, codebook])
                for codebook in range(self.num_codebooks)
            ]
        )
        # construct inputs_audio_attention_mask from audio_codes_lens, now we add the bos token and self.num_codebooks-1 padding tokens
        audio_codes_lens_pad = audio_codes_lens + 1
        # inputs_audio_attention_mask has shape (batch_size, sequence_length)
        inputs_audio_attention_mask = torch.arange(
            inputs_audio_embeds.shape[1], device=inputs_audio_embeds.device
        ).expand(inputs_audio_embeds.shape[0], -1) < audio_codes_lens_pad.unsqueeze(
            1
        ).to(
            inputs_audio_embeds.device
        )

        # separate postion_ids for text and audio, then concatenate them
        # position_ids = torch.arange(
        #     inputs_text_embeds.shape[1], device=inputs_text_embeds.device
        # ).unsqueeze(0)
        # position_ids_audio = torch.arange(
        #     inputs_audio_embeds.shape[1], device=inputs_audio_embeds.device
        # ).unsqueeze(0)
        # position_ids = torch.cat([position_ids, position_ids_audio], dim=1)
        # concatenate the text and audio embeddings
        inputs_embeds = torch.cat([inputs_text_embeds, inputs_audio_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, inputs_audio_attention_mask], dim=1)

        model_outputs = self.temporal_llm(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=None,
            return_dict=True,
            output_hidden_states=True,
        )
        last_hidden_state = model_outputs.hidden_states[-1].clone()

        # get the audio part only
        last_hidden_state = last_hidden_state[:, inputs_text_embeds.shape[1] :]

        dep_hidden_states = [
            self.depth_linear_in[i](last_hidden_state)
            for i in range(self.num_codebooks)
        ]  # shape [(B, T, dep_hidden_size)] * num_codebooks
        # there are [BxT, num_codebooks] input_ids --> [BxT, num_codebooks, dep_hidden_size]
        for i in range(self.num_codebooks):
            dep_input_embeds = self.dep_embed_tokens[i](
                depth_audio_codes_input[:, i]
            )  # B, T, dep_hidden_size
            dep_hidden_states[i] += dep_input_embeds
        # dep_hidden_input shape (BxT, num_codebooks, dep_hidden_size)
        dep_hidden_input = torch.stack(
            dep_hidden_states, dim=2
        )  # B, T, num_codebooks, dep_hidden_size
        assert dep_hidden_input.shape[2] == self.num_codebooks
        dep_hidden_input = dep_hidden_input.view(
            -1, self.num_codebooks, self.dep_hidden_size
        )

        depth_model_outputs = self.depth_llm(
            inputs_embeds=dep_hidden_input,
            return_dict=True,
            output_hidden_states=True,
        )  # shape (BxT, num_codebooks, dep_hidden_size)

        # now separately compute the audio logits
        audio_logits = [
            self.audio_lm_heads[codebook](
                depth_model_outputs.hidden_states[-1][:, codebook]
            )
            for codebook in range(self.num_codebooks)
        ]  # shape [(BxT, audio_vocab_size)] * num_codebooks

        # stack the audio logits
        # audio_logits shape (batch_sizexsequence_length, vocab_size, num_codebooks)
        audio_logits = torch.stack(audio_logits, dim=-1)
        batch_size = audio_codes_input.shape[0]
        audio_logits = audio_logits.view(
            batch_size, -1, self.audio_vocab_size, self.num_codebooks
        )
        # print("audio_logits shape: ", audio_logits.shape)
        # print("audio_labels shape: ", audio_labels.shape)

        total_loss = []
        top_k_acc_list = []

        # replace the delay audio labels' pad token with the IGNORE_TOKEN_ID
        audio_labels = audio_labels.masked_fill(
            audio_labels == self.audio_pad_token_id, IGNORE_TOKEN_ID
        )
        # print("audio_labels shape: ", audio_labels.shape)
        assert audio_labels.shape[1] == self.num_codebooks
        audio_labels[:, 1:, -1] = IGNORE_TOKEN_ID
        # print("audio_labels shape: ", audio_labels.shape, 23333333333)

        for i in range(self.num_codebooks):
            logits = audio_logits[:, :, :, i]
            labels = audio_labels[:, i]
            # print("labels shape: ", labels.shape)

            logits = logits.contiguous().view(-1, self.audio_vocab_size)
            labels = labels.contiguous().view(-1)
            # print("logits shape: ", logits.shape)
            # print("labels shape: ", labels.shape)

            loss = self.loss_fct(logits, labels)
            total_loss.append(loss)
            if debug:
                top_k_acc = self.audio_accuracy_metric(logits.detach(), labels).item()
                top_k_acc_list.append(top_k_acc)
        scale = 1.0
        total_loss_value = scale * total_loss[0] + sum(total_loss[1:])
        return total_loss_value, total_loss, top_k_acc_list

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        prompt_audio_codes: Optional[torch.Tensor] = None,
        audio_path_list: Optional[str] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        max_length: int = 1024,
        top_p: float = 0.9,
        top_k: int = -1,
    ):
        """
        input_ids: (batch_size, sequence_length)
        audio_codes: (batch_size, num_codebooks, sequence_length)
        """
        batch_size = input_ids.shape[0]
        assert batch_size == 1, "The batch size should be 1"
        # change the default -23 pad token in lhotse to the eos token id
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

        inputs_text_embeds = self.temporal_llm.get_input_embeddings()(input_ids)
        inputs_audio_embeds = sum(
            [
                self.embed_tokens[codebook](audio_codes_input[:, codebook])
                for codebook in range(self.num_codebooks)
            ]
        )
        # TODO: consider batch_size > 1, we need to add masks
        # position_ids = torch.arange(
        #     inputs_text_embeds.shape[1], device=inputs_text_embeds.device
        # ).unsqueeze(0)
        # position_ids_audio = torch.arange(
        #     inputs_audio_embeds.shape[1], device=inputs_audio_embeds.device
        # ).unsqueeze(0)
        # position_ids = torch.cat([position_ids, position_ids_audio], dim=1)

        inputs_embeds = torch.cat(
            [inputs_text_embeds, inputs_audio_embeds],
            dim=1,
        )

        kv_cache, preceding_tokens, token_ids_first_codebook = None, None, None
        generated_audio_tokens = []
        # last_step_codes B,Num_codebooks, 1
        # last_step_codes = audio_codes_input[:, :, -1:]

        for _ in range(max_length):
            outputs = self.temporal_llm(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=kv_cache,
                position_ids=None,
                output_hidden_states=True,
                return_dict=True,
            )
            kv_cache = outputs.past_key_values

            last_hidden_state = outputs.hidden_states[-1].clone()
            last_hidden_state = last_hidden_state[:, -1:]  # shape (B, 1, hidden_size)

            dep_hidden_states = [
                self.depth_linear_in[i](last_hidden_state)
                for i in range(self.num_codebooks)
            ]
            # start from the audio bos token B,num_codebooks,1
            last_ids = torch.full(
                (batch_size, 1),
                self.audio_bos_token_id,
                dtype=torch.long,
                device=last_hidden_state.device,
            )
            next_ids = []
            depth_kv_cache = None
            for i in range(self.num_codebooks):
                dep_input_embeds = self.dep_embed_tokens[i](last_ids)
                dep_input_embeds += dep_hidden_states[i]

                depth_model_outputs = self.depth_llm(
                    inputs_embeds=dep_input_embeds,
                    return_dict=True,
                    output_hidden_states=True,
                    use_cache=True,
                    past_key_values=depth_kv_cache,
                )
                audio_logits = self.audio_lm_heads[i](
                    depth_model_outputs.hidden_states[-1]
                )  # shape (B, 1, audio_vocab_size)
                audio_logits = audio_logits.squeeze(1)
                if i >= 0:
                    # get preceding tokens from generated_audio_tokens [(B,num_codebooks)] * decoding_steps
                    # preceding_tokens # shape (B, decoding_steps) for specific codebook
                    if generated_audio_tokens:
                        preceding_tokens = torch.stack(
                            [item[:, i] for item in generated_audio_tokens], dim=1
                        )
                        # get the last 10 columns for the preceding tokens
                        preceding_tokens = preceding_tokens[:, -10:]
                    else:
                        preceding_tokens = None
                    last_ids = topk_sampling(
                        audio_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=1.0,
                        repetition_aware_sampling=True,
                        preceding_tokens=preceding_tokens,
                    )
                else:
                    pass
                    # preceding_tokens = None
                    # greedy decoding
                    # last_ids = torch.argmax(audio_logits, dim=-1).unsqueeze(1)
                    # last_ids = topk_sampling(
                    #     audio_logits,
                    #     top_k=top_k,
                    #     top_p=top_p,
                    #     temperature=1.0,
                    #     repetition_aware_sampling=True,
                    #     preceding_tokens=preceding_tokens,
                    # )
                assert last_ids.shape[0] == 1, "WAR: The batch size should be 1"
                if last_ids[0, 0] == self.audio_eos_token_id:
                    assert i == 0, "Only the first codebook can generate the eos token"
                    break
                next_ids.append(last_ids)
                depth_kv_cache = depth_model_outputs.past_key_values

            if next_ids:
                next_ids = torch.cat(next_ids, dim=-1)  # shape (B, num_codebooks)
                generated_audio_tokens.append(next_ids)
                inputs_embeds = sum(
                    [
                        self.embed_tokens[codebook](next_ids[:, codebook].unsqueeze(-1))
                        for codebook in range(self.num_codebooks)
                    ]
                )
            else:
                break
        generated_audio_tokens = torch.stack(
            generated_audio_tokens, dim=-1
        )  # shape (B, num_codebooks, max_length)
        return generated_audio_tokens


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
    top_k=-1,
    top_p=1.0,
    temperature=1.0,
    repetition_aware_sampling=False,
    preceding_tokens=None,
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
    if repetition_aware_sampling and preceding_tokens is not None:
        window_size = 10
        threshold = 0.1
        # we first generate the target code ct′
        # by nucleus sampling with a pre-defined top-p value v. Then, we
        # calculate the repetition ratio r of token ct′
        # in the preceding code sequence with a window size K.
        # If the ratio r exceeds a pre-defined repetition threshold ratio tn, we replace the target code ct′
        # by
        # random sampling from p(ct′
        # |x, c<t·G,0; θAR). make sure the token is not repeated.
        # https://arxiv.org/abs/2406.05370
        # y: B, T
        # token: B, 1
        assert preceding_tokens is not None
        if preceding_tokens.shape[1] > window_size:
            preceding_tokens = preceding_tokens[:, -window_size:]
        if preceding_tokens.shape[1] > 0:
            for i, item in enumerate(preceding_tokens):
                # check if the repeat ratio exceeds the threshold
                if (item == tokens[i]).sum() / window_size > threshold:
                    # replace the target code ct′ by random sampling
                    probs = torch.nn.functional.softmax(logits[i], dim=-1)
                    token_new = torch.multinomial(probs, num_samples=1)
                    tokens[i] = token_new
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
