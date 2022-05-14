"""Define the class to calculate the residual.

As this will be used in the encoder and the decoder, we want to keep this
part rather generic. (See esp. the forward-step. I feel it could be done
more elegantly, but fail to see how at a glance.)
"""

import torch


class Residual(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        dim: int,
        dropout: float,
    ):
        super().__init__()
        self.layer = layer
        self.norm = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor):
        # Please note that we expect the args to contain tensors which have
        # been calculated by other modules and might come alone or in the Q, K, V ordering.
        # The output would then be Res = Norm(Q + dropout((Q, K, V)))
        return self.norm(
            tensors[0] + self.dropout(self.layer(*tensors))
        )