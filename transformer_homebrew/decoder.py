"""Decoder-module for the Transformer-architecture."""

import torch

from attention_heads import MultiHeadAttention
from positional_encoding import positional_encoding
from residual import Residual


class _DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        dim_mdl: int = 512,
        N_heads: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Determine the dimension for Q and K, which are equal in this case:
        dim_qk = max(dim_mdl // N_heads, 1)

        # Define the self-attention-layer for the target y:
        self.attention_y = Residual(
            MultiHeadAttention(N_heads=N_heads, dim_in=dim_mdl, dim_q=dim_qk, dim_k=dim_qk),
            dim=dim_mdl,
            dropout=dropout,
        )

        # Define the attention layer with respect to the memory:
        self.attention_memory = Residual(
            MultiHeadAttention(N_heads=N_heads, dim_in=dim_mdl, dim_q=dim_qk, dim_k=dim_qk),
            dim=dim_mdl,
            dropout=dropout,
        )

        # Set up the FF-module based on the attention-results:
        self.feed_forward = Residual(
            torch.nn.Sequential(
                torch.nn.Linear(dim_mdl, dim_ff),
                torch.nn.ReLU(),
                torch.nn.Linear(dim_ff, dim_mdl)
            ),
            dim=dim_mdl,
            dropout=dropout,
        )

    def forward(self, y: torch.Tensor, memory: torch.Tensor):
        # First, the target gets passed through the self-attention-layers.
        # The result of this then gets passed through the memory-attention-layer
        # and finally through the feed-forward-layer.
        # (Yes, I don't like assigning variables, when I think I can avoid it with
        # only minor losses to readability.)
        return self.feed_forward(
            self.attention_memory(
                self.attention_y(y, y, y),
                memory,
                memory,
            )
        )


class Decoder(torch.nn.Module):
    def __init__(
        self,
        N_decoder_layer: int = 4,
        dim_mdl: int = 512,
        N_heads: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.3
    ):
        super().__init__()

        self.decoders = torch.nn.ModuleList(
            [_DecoderLayer(dim_mdl=dim_mdl, N_heads=N_heads, dim_ff=dim_ff, dropout=dropout) for _ in range(N_decoder_layer)]
        )

        self.linear = torch.nn.Linear(dim_mdl, dim_mdl)

    def forward(self, y: torch.Tensor, memory: torch.Tensor):
        y += positional_encoding(y.size(1), y.size(2))
        for decoder in self.decoders:
            y = decoder(y, memory)

        return torch.softmax(self.linear(y), dim=-1)
