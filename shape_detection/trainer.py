"""Pytorch-Lightning trainer."""

import pytorch_lightning as pl

from data_loader import ShapeDataLoader
from mlp_model import ShapeDetectorModelMLP

if __name__ == '__main__':
    # Set up model and data:
    clf = ShapeDetectorModelMLP()
    mnist = ShapeDataLoader()

    # Create Trainer Object
    trainer = pl.Trainer(gpus=0, accelerator='dp', max_epochs=5)
    trainer.fit(model=clf, datamodule=mnist)
