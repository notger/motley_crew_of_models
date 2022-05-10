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
        self.conv1 = torch.nn.Conv2d(in_channels=N_c, out_channels=40, kernel_size=(5, 5), padding=2, dilation=2)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(40)

        self.conv2 = torch.nn.Conv2d(in_channels=40, out_channels=50, kernel_size=(5, 5), padding=2, dilation=2)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.bn2 = torch.nn.BatchNorm2d(50)

        self.conv3 = torch.nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(5, 5), padding=2, dilation=2)
        self.relu3 = torch.nn.ReLU()
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn3 = torch.nn.BatchNorm2d(50)

        self.linear4 = torch.nn.Linear(in_features=16200, out_features=500)
        self.relu4 = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(p=0.5)

        self.linear5 = torch.nn.Linear(in_features=500, out_features=N_target)
        self.logsoftmax5 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        z1 = self.bn1(self.maxpool1(self.relu1(self.conv1(x))))
        z2 = self.bn2(self.maxpool2(self.relu2(self.conv2(z1))))
        z3 = self.bn3(self.maxpool3(self.relu3(self.conv3(z2))))
        z4 = self.dropout(self.relu4(self.linear4(torch.flatten(z3, 1))))
        return self.logsoftmax5(self.linear5(z4))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr = self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.forward(x), y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.forward(x), y)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self.forward(x), y)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss
