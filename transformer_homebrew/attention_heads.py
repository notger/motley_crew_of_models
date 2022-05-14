"""Module to define the attention heads."""

import torch


class _AttentionHead(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_q: int,
        dim_k: int,
    ):
        super().__init__()

        self.Ql = torch.nn.Linear(dim_in, dim_q)
        self.Kl = torch.nn.Linear(dim_in, dim_k)
        self.Vl = torch.nn.Linear(dim_in, dim_k)

    @staticmethod
    def scaled_inner_product(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        QKT = torch.bmm(Q, K.transpose(1, 2))
        scaler = Q.size(-1) ** 0.5
        return torch.bmm(torch.softmax(QKT / scaler, dim=-1), V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        return self.scaled_inner_product(self.Ql(Q), self.Kl(K), self.Vl(V))


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        N_heads: int,
        dim_in: int,
        dim_q: int,
        dim_k: int,
    ):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            _AttentionHead(dim_in, dim_q, dim_k) for _ in range(N_heads)
        ])
        self.linear = torch.nn.Linear(N_heads * dim_k, dim_in)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        return self.linear(
            torch.concat(
                [head(Q, K, V) for head in self.heads], dim=-1
            )
        )
