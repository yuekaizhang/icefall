# Copyright (c) 2023 ASLP@NWPU (authors: He Wang, Fan Yu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License. Modified from
# FunASR(https://github.com/alibaba-damo-academy/FunASR)

from typing import Optional

import torch
from torch import nn
from typing import Tuple

class Cif(nn.Module):

    def __init__(
        self,
        idim=1280,
        l_order=1,
        r_order=1,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0.0,
        tail_threshold=0.45,
        residual=False,
        cnn_groups=1,
    ):
        super().__init__()

        self.pad = nn.ConstantPad1d((l_order, r_order), 0.0)
        self.cif_conv1d = nn.Conv1d(
            idim,
            idim,
            l_order + r_order + 1,
            stride=1,
            groups=idim if cnn_groups == 0 else cnn_groups)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold
        self.residual = residual

    def forward(
        self,
        hidden,
        target_label: Optional[torch.Tensor] = None,
        mask: torch.Tensor = torch.tensor(0),
        ignore_id: int = -1,
        mask_chunk_predictor: Optional[torch.Tensor] = None,
        target_label_length: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        memory = self.cif_conv1d(queries)
        if self.residual:
            output = memory + context
        else:
            output = memory
        output = self.dropout(output)
        output = output.transpose(1, 2)
        output = torch.relu(output)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor -
                                          self.noise_threshold)
        if mask is not None:
            mask = mask.transpose(-1, -2)
            alphas = alphas * mask
            mask = mask.squeeze(-1)
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)
        if target_length is not None:
            alphas *= (target_length / token_num)[:, None] \
                .repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            hidden, alphas, token_num = self.tail_process_fn(hidden,
                                                             alphas,
                                                             token_num,
                                                             mask=mask)

        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)

        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        return acoustic_embeds, token_num, alphas, cif_peak

    def tail_process_fn(
        self,
        hidden: torch.Tensor,
        alphas: torch.Tensor,
        token_num: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, d = hidden.size()
        if mask is not None:
            zeros_t = torch.zeros((b, 1),
                                  dtype=torch.float32,
                                  device=alphas.device)
            mask = mask.to(zeros_t.dtype)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * self.tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold_tensor = torch.tensor([self.tail_threshold],
                                                 dtype=alphas.dtype).to(
                                                     alphas.device)
            tail_threshold_tensor = torch.reshape(tail_threshold_tensor,
                                                  (1, 1))
            alphas = torch.cat([alphas, tail_threshold_tensor], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor

def cif(hidden: torch.Tensor, alphas: torch.Tensor, threshold: float):
    batch_size, len_time, hidden_size = hidden.size()

    # loop varss
    integrate = torch.zeros([batch_size], device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
    # intermediate vars along time
    list_fires = []
    list_frames = []

    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([batch_size],
                                             device=hidden.device) - integrate

        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place,
            integrate - torch.ones([batch_size], device=hidden.device),
            integrate)
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur

        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(fire_place[:, None].repeat(1, hidden_size),
                            remainds[:, None] * hidden[:, t, :], frame)

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    len_labels = torch.round(alphas.sum(-1)).int()
    max_label_len = len_labels.max()
    for b in range(batch_size):
        fire = fires[b, :]
        l = torch.index_select(frames[b, :, :], 0,
                               torch.nonzero(fire >= threshold).squeeze())
        pad_l = torch.zeros([int(max_label_len - l.size(0)), hidden_size],
                            device=hidden.device)
        list_ls.append(torch.cat([l, pad_l], 0))
    return torch.stack(list_ls, 0), fires