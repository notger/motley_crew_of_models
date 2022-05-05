"""Defines a loader for the training and validation.

The thing to look out for is that there is no fixed loader, 
but we will create all the data on the fly.
"""

import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


class ShapeDataLoader(pl.LightningDataModule):
    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
      
        self.train_data = datasets.MNIST('', train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST('', train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=4, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, num_workers=2, batch_size=32, shuffle=False)
