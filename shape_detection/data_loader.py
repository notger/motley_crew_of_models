"""Defines a loader for the training and validation.

The thing to look out for is that there is no fixed loader, 
but we will create all the data on the fly.
"""

from typing import Optional

import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, IterableDataset

from shape_generator import ShapeGenerator, Colouring


class ShapeIterableDataset(IterableDataset):
    def __init__(self, N_x=256, N_y=256, colouring=None, batch_size=10):
        super(ShapeIterableDataset).__init__()
        self.shape_generator = ShapeGenerator(N_x, N_y)
        self.colouring = colouring
        self.transforms = transforms.ToTensor()
        self.batch_size = batch_size

    def __iter__(self):
        dataset = []
        for _ in range(self.batch_size):
            im, label = self.shape_generator.generate_random(colouring=self.colouring)
            dataset.append([self.transforms(im), label])
        return iter(dataset)


class ShapeIterableDataLoader(pl.LightningDataModule):
    def __init__(self, N_x=50, N_y=50, batch_size=10, colouring=Colouring.SINGLE_COLOUR):
        super(ShapeIterableDataLoader, self).__init__()
        self.N_x = N_x
        self.N_y = N_y
        self.batch_size = batch_size
        self.colouring = colouring

    def prepare_data(self) -> None:
        return super().prepare_data()

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        return DataLoader(
            ShapeIterableDataset(self.N_x, self.N_y, colouring=self.colouring, batch_size=self.batch_size)
        )
    
    def train_dataloader(self):
        return DataLoader(
            ShapeIterableDataset(self.N_x, self.N_y, colouring=self.colouring, batch_size=self.batch_size),
        )


class MNISTDataLoader(pl.LightningDataModule):
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
