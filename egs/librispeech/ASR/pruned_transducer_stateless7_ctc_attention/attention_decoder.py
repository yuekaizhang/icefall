# Copyright    2022  Nvidia        (authors: Yuekai Zhang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from typing import List

from transformer import TransformerDecoder, TransformerDecoderLayer, PositionalEncoding, add_sos, add_eos, generate_square_subsequent_mask, decoder_padding_mask
from label_smoothing import LabelSmoothingLoss

from scaling import (
    ScaledLinear,
    ScaledEmbedding,
)

class AttentionDecoder(nn.Module):
    """This class warps TransformerDecoder and other related classes into the AttentionDecoder Class.
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int = 512,
        sos_id: int = 0,
        eos_id: int = 0,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()

        self.sos_id = sos_id
        self.eos_id = eos_id

        self.decoder_num_class = (
            vocab_size
        )  # bpe model already has sos/eos symbol

        self.decoder_embed = ScaledEmbedding(
            num_embeddings=self.decoder_num_class, embedding_dim=decoder_dim
        )
        self.decoder_pos = PositionalEncoding(decoder_dim, dropout)

        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            aux_layers=[],
        )

        self.decoder_output_layer = ScaledLinear(
            decoder_dim, self.decoder_num_class, bias=True
        )

        self.decoder_criterion = LabelSmoothingLoss()


    @torch.jit.export
    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        token_ids: List[List[int]],
        sos_id: int = 0,
        eos_id: int = 0,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          memory:
            It's the output of the encoder of shape (S, N, C)
          memory_key_padding_mask:
            The padding mask from the encoder of shape (N, S).
          token_ids:
            A list-of-list IDs. Each sublist contains IDs for an utterance.
            The IDs can be either phone IDs or word piece IDs.
          sos_id:
            sos token id
          eos_id:
            eos token id
          warmup:
            a floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up". It is used
            to turn modules on sequentially.
        Returns:
          A scalar, the **sum** of label smoothing loss over utterances
          in the batch without any normalization.
        """
        ys_in = add_sos(token_ids, sos_id=sos_id)
        ys_in = [torch.tensor(y) for y in ys_in]
        ys_in_pad = pad_sequence(
            ys_in, batch_first=True, padding_value=float(eos_id)
        )

        ys_out = add_eos(token_ids, eos_id=eos_id)
        ys_out = [torch.tensor(y) for y in ys_out]
        ys_out_pad = pad_sequence(
            ys_out, batch_first=True, padding_value=float(-1)
        )

        device = memory.device
        ys_in_pad = ys_in_pad.to(device)
        ys_out_pad = ys_out_pad.to(device)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(
            device
        )

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad, ignore_id=eos_id)
        # TODO: Use length information to create the decoder padding mask
        # We set the first column to False since the first column in ys_in_pad
        # contains sos_id, which is the same as eos_id in our current setting.
        tgt_key_padding_mask[:, 0] = False

        tgt = self.decoder_embed(ys_in_pad)  # (N, T) -> (N, T, C)
        tgt = self.decoder_pos(tgt)
        tgt = tgt.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        pred_pad = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            warmup=warmup,
        )  # (T, N, C)
        pred_pad = pred_pad.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        pred_pad = self.decoder_output_layer(pred_pad)  # (N, T, C)

        decoder_loss = self.decoder_criterion(pred_pad, ys_out_pad)

        return decoder_loss

    @torch.jit.export
    def decoder_nll(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        token_ids: List[torch.Tensor],
        sos_id: int,
        eos_id: int,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
          memory:
            It's the output of the encoder of shape (S, N, C).
          memory_key_padding_mask:
            The padding mask from the encoder of shape (N, S).
          token_ids:
            A list-of-list IDs (e.g., word piece IDs).
            Each sublist represents an utterance.
          sos_id:
            The token ID for SOS.
          eos_id:
            The token ID for EOS.
          warmup:
            a floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up". It is used
            to turn modules on sequentially.
            
        Returns:
          A 2-D tensor of shape (len(token_ids), max_token_length)
          representing the cross entropy loss (i.e., negative log-likelihood).
        """
        # The common part between this function and decoder_forward could be
        # extracted as a separate function.
        if isinstance(token_ids[0], torch.Tensor):
            # This branch is executed by torchscript in C++.
            # See https://github.com/k2-fsa/k2/pull/870
            # https://github.com/k2-fsa/k2/blob/3c1c18400060415b141ccea0115fd4bf0ad6234e/k2/torch/bin/attention_rescore.cu#L286
            token_ids = [tolist(t) for t in token_ids]

        ys_in = add_sos(token_ids, sos_id=sos_id)
        ys_in = [torch.tensor(y) for y in ys_in]
        ys_in_pad = pad_sequence(
            ys_in, batch_first=True, padding_value=float(eos_id)
        )

        ys_out = add_eos(token_ids, eos_id=eos_id)
        ys_out = [torch.tensor(y) for y in ys_out]
        ys_out_pad = pad_sequence(
            ys_out, batch_first=True, padding_value=float(-1)
        )

        device = memory.device
        ys_in_pad = ys_in_pad.to(device, dtype=torch.int64)
        ys_out_pad = ys_out_pad.to(device, dtype=torch.int64)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(
            device
        )

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad, ignore_id=eos_id)
        # TODO: Use length information to create the decoder padding mask
        # We set the first column to False since the first column in ys_in_pad
        # contains sos_id, which is the same as eos_id in our current setting.
        tgt_key_padding_mask[:, 0] = False

        tgt = self.decoder_embed(ys_in_pad)  # (N, T) -> (N, T, C)
        tgt = self.decoder_pos(tgt)
        tgt = tgt.permute(1, 0, 2)  # (N, T, С) -> (T, N, C)
        pred_pad = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            warmup=warmup,
        )  # (T, B, F)
        pred_pad = pred_pad.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        pred_pad = self.decoder_output_layer(pred_pad)  # (N, T, C)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            pred_pad.view(-1, self.decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=-1,
            reduction="none",
        )

        nll = nll.view(pred_pad.shape[0], -1)

        return nll