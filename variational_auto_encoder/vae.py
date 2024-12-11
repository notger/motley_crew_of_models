import torch
from torch.nn.functional import leaky_relu


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, num_features: int, num_embeddings: int):
        super(VariationalAutoEncoder, self).__init__()

        self.enc1 = torch.nn.Linear(num_features, 2 * num_features)
        self.enc2 = torch.nn.Linear(2 * num_features, num_embeddings)
        self.dec1 = torch.nn.Linear(num_embeddings, 2 * num_features)
        self.dec2 = torch.nn.Linear(2 * num_features, num_features)

        self.bn_enc1 = torch.nn.BatchNorm1d(2 * num_features)
        self.bn_enc2 = torch.nn.BatchNorm1d(num_embeddings)
        self.bn_dec1 = torch.nn.BatchNorm1d(2 * num_features)
        self.bn_dec2 = torch.nn.BatchNorm1d(num_features)

    def encode(self, x):
        x1 = leaky_relu(self.bn_enc1(self.enc1(x)))
        return leaky_relu(self.bn_enc2(self.enc2(x1)))

    def decode(self, x):
        x1 = leaky_relu(self.bn_dec1(self.dec1(x)))
        return leaky_relu(self.bn_dec2(self.dec2(x1)))

    def forward(self, x):
        e = self.encode(x)
        return self.decode(e), e
