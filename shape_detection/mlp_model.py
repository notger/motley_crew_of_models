"""Simple MLP-model to test the basic setup and training flow."""

import torch
import pytorch_lightning as pl


class ShapeDetectorModelMLP(pl.LightningModule):

    def __init__(self, N_x=50, N_y=50, N_c=3, N_target=10):
        super(ShapeDetectorModelMLP, self).__init__()
        self.fc1 = torch.nn.Linear(N_x * N_y * N_c, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.out = torch.nn.Linear(128, N_target)
        self.lr = 0.001
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size,-1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.out(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr = self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        return self.loss(logits, y)

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        print(x.shape)
        logits = self.forward(x)
        return self.loss(logits, y)
