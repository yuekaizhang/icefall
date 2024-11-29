from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
assert IGNORE_TOKEN_ID == -100


class EncoderProjector(nn.Module):
    """
    The encoder projector module. It is used to project the encoder outputs to the same dimension as the language model.
    Modified from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/models/projector.py.
    Args:
        encoder_dim (:obj:`int`): The dimension of the encoder outputs.
        llm_dim (:obj:`int`): The dimension of the language model.
        downsample_rate (:obj:`int`, `optional`, defaults to 5): The downsample rate to use.
    """

    def __init__(self, encoder_dim, llm_dim, downsample_rate=5):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * self.downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):

        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.downsample_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.downsample_rate, feat_dim * self.downsample_rate
        )

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
    ):
        super().__init__()
        # self.encoder = encoder
        # self.llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.llm = AutoModelForCausalLM.from_pretrained(
            "/workspace/Qwen2.5-0.5B-Instruct"
        )
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
        # self.encoder_projector = encoder_projector
        self.loss_fct = CrossEntropyLoss()

    def _merge_input_ids_with_speech_features(
        self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None
    ):
        """
        Merge the speech features with the input_ids and attention_mask. This is done by replacing the speech tokens
        with the speech features and padding the input_ids to the maximum length of the speech features.
        Modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py#L277.
        Args:
            speech_features (:obj:`torch.Tensor`): The speech features to merge with the input_ids.
            inputs_embeds (:obj:`torch.Tensor`): The embeddings of the input_ids.
            input_ids (:obj:`torch.Tensor`): The input ids to merge.
            attention_mask (:obj:`torch.Tensor`): The attention mask to merge.
            labels (:obj:`torch.Tensor`, `optional`): The labels to merge.
        Returns:
            :obj:`Tuple(torch.Tensor)`: The merged embeddings, attention mask, labels and position ids.
        """
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.llm.config.pad_token_id)
        )
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == self.llm.config.default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_speech_tokens.max() * (speech_len - 1)
        ) + sequence_length
        batch_indices, non_speech_indices = torch.where(
            input_ids != self.llm.config.default_speech_token_id
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged speech-text sequence.
        # `special_speech_token_mask` identifies speech tokens. Each speech token will be replaced by `nb_text_tokens_per_speechs - 1` text tokens.
        # `torch.cumsum` computes how each speech token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        )
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                IGNORE_TOKEN_ID,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_speech_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<speech>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the speech features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_speech_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_speech_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_speech_indices
            ]

        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[
            :, None
        ].to(target_device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= speech_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(
            input_ids == self.llm.config.pad_token_id
        )
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_codes: Optional[torch.Tensor] = None,
        text_labels: Optional[torch.LongTensor] = None,
        audio_labels: Optional[torch.LongTensor] = None,
    ):
        """
        input_ids: (batch_size, sequence_length)
        audio_codes: (batch_size, num_codebooks, sequence_length)
        text_labels: (batch_size, sequence_length)
        audio_labels: (batch_size, num_codebooks, sequence_length)
        """
        audio_codes = build_delay_pattern_mask(
            audio_codes,
            bos_token_id=self.codebook_size,
            pad_token_id=self.codebook_size,
            max_length=audio_codes.shape[-1] + 16,
        )[0]
        if audio_labels:
            audio_labels = build_delay_pattern_mask(
                audio_labels,
                bos_token_id=self.codebook_size,
                pad_token_id=self.codebook_size,
                max_length=audio_labels.shape[-1] + 16,
            )[0]
        else:
            audio_labels = audio_codes.clone()

        # now we could left pad the audio codes and audio labels to let text tokens be predicted before audio tokens
        shift_text_tokens = 2
        audio_codes = torch.cat(
            [
                torch.full(
                    (audio_codes.shape[0], audio_codes.shape[1], shift_text_tokens),
                    self.codebook_size,
                    dtype=audio_codes.dtype,
                    device=audio_codes.device,
                ),
                audio_codes,
            ],
            dim=-1,
        )
        audio_labels = torch.cat(
            [
                torch.full(
                    (audio_labels.shape[0], audio_labels.shape[1], shift_text_tokens),
                    self.codebook_size,
                    dtype=audio_labels.dtype,
                    device=audio_labels.device,
                ),
                audio_labels,
            ],
            dim=-1,
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
        if text_labels:
            # pad the text labels to the maximum length of the audio labels with pad token
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
        else:
            text_labels = input_ids.clone()

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        audio_inputs_embeds = sum(
            [
                self.embed_tokens[codebook](audio_codes[:, codebook])
                for codebook in range(self.codebook_size)
            ]
        )
        inputs_embeds += audio_inputs_embeds

        model_outputs = self.llm(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=text_labels,
        )

        last_hidden_state = model_outputs.last_hidden_state
        audio_logits = [
            self.audio_lm_heads[codebook](last_hidden_state).float()
            for codebook in range(self.num_codebooks)
        ]

        # last hidden state shape (batch_size, sequence_length, hidden_size)
        # audio_logits shape (batch_size, sequence_length, codebook_size)
        # audio_labels shape (batch_size, sequence_length, num_codebooks)
        total_loss = 10 * model_outputs.loss
        for i in range(self.num_codebooks):
            shift_logits = audio_logits[i][..., :-1, :].contiguous()
            shift_labels = audio_labels[..., 1:, i].contiguous()

            shift_logits = shift_logits.view(-1, self.codebook_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.masked_fill(
                shift_labels == self.codebook_size, -100
            )

            loss = self.loss_fct(shift_logits, shift_labels)
            total_loss += loss

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                text_labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )

        return total_loss, acc

    def decode(
        self,
        fbank: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):

        encoder_outs = self.encoder(fbank)
        speech_features = self.encoder_projector(encoder_outs)
        speech_features = speech_features.to(torch.float16)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        (
            inputs_embeds,
            attention_mask,
            _,
            position_ids,
        ) = self._merge_input_ids_with_speech_features(
            speech_features, inputs_embeds, input_ids, attention_mask
        )
        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 1),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )

        return generated_ids


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
