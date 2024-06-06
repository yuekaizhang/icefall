from torch import nn
import torch
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class EncoderProjector(nn.Module):

    def __init__(self, encoder_dim, llm_dim):
        super().__init__()
        self.linear1 = nn.Linear(encoder_dim, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):   
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class SPEECH_LLM(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        for name, param in encoder.named_parameters(): 
            param.requires_grad = False
        self.encoder.eval()
        self.llm = llm
        for name, param in llm.named_parameters(): 
            param.requires_grad = False
        self.llm.eval()
        self.encoder_projector = encoder_projector
        self.encoder_outputs_downsample_rate = 4

    def _merge_input_ids_with_speech_features(self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None):
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.llm.config.pad_token_id))
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == self.llm.config.default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_speech_tokens.max() * (speech_len - 1)) + sequence_length
        batch_indices, non_speech_indices = torch.where(input_ids != self.llm.config.default_speech_token_id)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged speech-text sequence.
        # `special_speech_token_mask` identifies speech tokens. Each speech token will be replaced by `nb_text_tokens_per_speechs - 1` text tokens.
        # `torch.cumsum` computes how each speech token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), IGNORE_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device
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
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_speech_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_speech_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_speech_indices]

        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[:, None].to(target_device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= speech_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.llm.config.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(self,
                fbank: torch.Tensor = None,
                input_ids: torch.LongTensor = None,
                attention_mask: torch.Tensor = None,
                labels: torch.LongTensor = None,
                ):
        encoder_outs = self.encoder(fbank)
        # downsample encoder_outs by 4
        encoder_outs = encoder_outs[:, ::self.encoder_outputs_downsample_rate]

        speech_features = self.encoder_projector(encoder_outs)
        
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        #print("input_ids", input_ids, input_ids.shape)
        #print("labels", labels, labels.shape)
        inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_speech_features(
            speech_features, inputs_embeds, input_ids, attention_mask, labels
        )
        #print("labels", labels, labels.shape)
        #print("speech_features", speech_features.shape)

        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, position_ids=position_ids)

        return model_outputs


    def decode(self,
               fbank: torch.Tensor = None,
               input_ids: torch.LongTensor = None,
               attention_mask: torch.Tensor = None,
               **kwargs
               ):

        encoder_outs = self.encoder(fbank)
        # downsample encoder_outs by 4
        encoder_outs = encoder_outs[:, ::self.encoder_outputs_downsample_rate]

        speech_features = self.encoder_projector(encoder_outs)
        speech_features = speech_features.to(torch.float16)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, _, position_ids = self._merge_input_ids_with_speech_features(
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
            pad_token_id=self.llm.config.pad_token_id
        )
        # print(generated_ids, input_ids)
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
        # ]
        return generated_ids