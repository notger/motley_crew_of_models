"""Pytorch-Lightning trainer."""

import pytorch_lightning as pl

from data_loader import MNISTDataLoader, ShapeIterableDataLoader
from mlp_model import ShapeDetectorModelMLP

from shape_generator import ShapeTypes, Colouring

if __name__ == '__main__':
    # Set up model and data:
    #data = MNISTDataLoader() 
    #clf = ShapeDetectorModelMLP(N_x=28, N_y=28, N_c=1, N_target=10)

    data = ShapeIterableDataLoader(N_x=50, N_y=50, batch_size=100, colouring=Colouring.SINGLE_COLOUR)
    clf = ShapeDetectorModelMLP(N_x=50, N_y=50, N_c=3, N_target=len(ShapeTypes))

    # Create Trainer Object
    trainer = pl.Trainer(gpus=0, accelerator='dp', max_epochs=50)
    trainer.fit(model=clf, datamodule=data)
