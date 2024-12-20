# Later found a good example implemenation here; useful for debugging and benchmarking:
# https://github.com/pytorch/examples/blob/main/vae/main.py

import torch
from torch.nn.functional import leaky_relu


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, num_features: int, num_embeddings: int):
        super(VariationalAutoEncoder, self).__init__()

        self.enc1 = torch.nn.Linear(num_features, 2 * num_features)
        self.enc2 = torch.nn.Linear(2 * num_features, num_embeddings)
        self.dec1 = torch.nn.Linear(num_embeddings, 2 * num_features)
        self.dec2 = torch.nn.Linear(2 * num_features, num_features)

        self.mean_layer = torch.nn.Linear(num_embeddings, num_embeddings)
        self.var_layer = torch.nn.Linear(num_embeddings, num_embeddings)

        self.bn_enc1 = torch.nn.BatchNorm1d(2 * num_features)
        self.bn_enc2 = torch.nn.BatchNorm1d(num_embeddings)
        self.bn_dec1 = torch.nn.BatchNorm1d(2 * num_features)
        self.bn_dec2 = torch.nn.BatchNorm1d(num_features)

    def encode(self, x):
        x1 = leaky_relu(self.bn_enc1(self.enc1(x)))
        x2 = leaky_relu(self.bn_enc2(self.enc2(x1)))
        return self.mean_layer(x2), self.var_layer(x2)

    def reparameterise(self, mean, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, x):
        x1 = leaky_relu(self.bn_dec1(self.dec1(x)))
        return leaky_relu(self.bn_dec2(self.dec2(x1)))

    def forward(self, x):
        mean, var = self.encode(x)
        e = self.reparameterise(mean, var)
        return self.decode(e), e
