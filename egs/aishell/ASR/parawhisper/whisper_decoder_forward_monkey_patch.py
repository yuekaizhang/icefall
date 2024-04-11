import torch
import torch.nn.functional as F
import whisper
from torch import Tensor
from torch import nn
from typing import Dict, Iterable, Optional
from whisper.model import ResidualAttentionBlock, LayerNorm
import numpy as np


def forward_nar(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, return_second_top_feature=False):
    """
    x : torch.LongTensor, shape = (batch_size, <= n_ctx)
        the text tokens
    xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
        the encoded audio features to be attended on
    """
    offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
    # x = (
    #     self.token_embedding(x)
    #     + self.positional_embedding[offset : offset + x.shape[-1]]
    # )
    x = (
        x
        + self.positional_embedding[offset : offset + x.shape[1]]
    )
    x = x.to(xa.dtype)

    for block in self.blocks:
        # x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = block(x, xa, kv_cache=kv_cache)

    x = self.ln(x)
    logits = (
        x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
    ).float()
    if return_second_top_feature:
        return logits, x
    else:
        return logits

def forward_nar_causal(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, return_second_top_feature=False):
    """
    x : torch.LongTensor, shape = (batch_size, <= n_ctx)
        the text tokens
    xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
        the encoded audio features to be attended on
    """
    offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
    # x = (
    #     self.token_embedding(x)
    #     + self.positional_embedding[offset : offset + x.shape[-1]]
    # )
    x = (
        x
        + self.positional_embedding[offset : offset + x.shape[1]]
    )
    x = x.to(xa.dtype)

    for block in self.blocks:
        x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        # x = block(x, xa, kv_cache=kv_cache)

    x = self.ln(x)
    logits = (
        x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
    ).float()
    if return_second_top_feature:
        return logits, x
    else:
        return logits

def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None, return_second_top_feature=False):
    """
    x : torch.LongTensor, shape = (batch_size, <= n_ctx)
        the text tokens
    xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
        the encoded audio features to be attended on
    """
    offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
    x = (
        self.token_embedding(x)
        + self.positional_embedding[offset : offset + x.shape[-1]]
    )
    x = (
        x
        + self.positional_embedding[offset : offset + x.shape[1]]
    )
    x = x.to(xa.dtype)

    for block in self.blocks:
        x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

    x = self.ln(x)
    logits = (
        x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
    ).float()
    if return_second_top_feature:
        return logits, x
    else:
        return logits

def replace_whisper_decoder_forward():
    """
    This function monkey patches the forward method of the whisper encoder.
    To be called before the model is loaded, it changes whisper to process audio with any length < 30s.
    """
    whisper.model.TextDecoder.forward_nar = forward_nar
    whisper.model.TextDecoder.forward = forward
    whisper.model.TextDecoder.forward_nar_causal = forward_nar_causal
