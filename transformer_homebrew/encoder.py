"""Encoder module for the Transformer-architecture."""

import torch

from attention_heads import MultiHeadAttention
from positional_encoding import positional_encoding
from residual import Residual


class _EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        dim_mdl: int = 512,
        N_heads: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Determine the dimension of Q and K, which are equal in this case:
        dim_qk = max(dim_mdl // N_heads, 1)

        # Set up the attention module:
        self.attention = Residual(
            MultiHeadAttention(
                N_heads=N_heads,
                dim_in=dim_mdl,
                dim_q=dim_qk,
                dim_k=dim_qk
            ),
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

    def forward(self, x: torch.Tensor):
        return self.feed_forward(
            self.attention(x, x, x)
        )


class Encoder(torch.nn.Module):
    def __init__(
        self,
        N_encoder_layer: int = 4,
        dim_mdl: int = 512,
        N_heads: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoders = torch.nn.ModuleList(
            [
                _EncoderLayer(dim_mdl=dim_mdl, N_heads=N_heads, dim_ff=dim_ff, dropout=dropout) 
                for _ in range(N_encoder_layer)
            ]
        )

    def forward(self, x: torch.Tensor):
        x += positional_encoding(x.size(1), x.size(2))
        for encoder in self.encoders:
            x = encoder(x)
    
        return x
