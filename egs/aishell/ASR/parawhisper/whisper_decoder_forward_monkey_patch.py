import torch
import torch.nn.functional as F
import whisper
from torch import Tensor
from torch import nn
from typing import Optional

class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        # mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        # self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
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

        return logits

def replace_whisper_decoder_forward():
    """
    This function monkey patches the forward method of the whisper encoder.
    To be called before the model is loaded, it changes whisper to process audio with any length < 30s.
    """
    whisper.model.TextDecoder = TextDecoder
