"""Simple CNN model to verify the basic idea of being able to identify shapes."""

import torch
import pytorch_lightning as pl


class ShapeDetectorModelCNN(pl.LightningModule):

    def __init__(self, N_c=3, N_target=10, learning_rate=0.001):
        super(ShapeDetectorModelCNN, self).__init__()

        # Hyperparameters:
        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()

        # Basic architecture, reminiscent of LeNet with a bit of batch normalisation:
        self.conv1 = torch.nn.Conv2d(in_channels=N_c, out_channels=20, kernel_size=(5, 5))
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.linear3 = torch.nn.Linear(in_features=450 * 9, out_features=500)
        self.relu3 = torch.nn.ReLU()

        self.linear4 = torch.nn.Linear(in_features=500, out_features=N_target)
        self.logsoftmax4 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        z1 = self.maxpool1(self.relu1(self.conv1(x)))
        z2 = self.maxpool2(self.relu2(self.conv2(z1)))
        z3 = self.relu3(self.linear3(torch.flatten(z2, 1)))
        return self.logsoftmax4(self.linear4(z3))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr = self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        return self.loss(self.forward(x), y)

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        return self.loss(self.forward(x), y)