"""Pytorch-Lightning trainer.

The trainer currently trains the CNN on the randomly generated data.
If you want to debug, there is also an MLP-version (very bad) and an
MLP which works on the MNIST-data (very boring). Check mlp_model.py for
the model and data_loader.py for the data-loaders.
"""

import pytorch_lightning as pl

from data_loader import ShapeIterableDataLoader
from cnn_model import ShapeDetectorModelCNN

from shape_generator import ShapeTypes, Colouring

if __name__ == '__main__':
    # Set up model and data:
    data_module = ShapeIterableDataLoader(N_x=50, N_y=50, batch_size=100, colouring=Colouring.SINGLE_COLOUR)
    shape_cnn = ShapeDetectorModelCNN(N_c=3, N_target=len(ShapeTypes), learning_rate=0.001)

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        verbose=True,
        save_top_k=1,
        filename='shape_identification_best',
        monitor=['train_loss', 'val_loss'],
        mode='min',
    )

    early_stopping_callback = pl.callbacks.EarlyStopping('val_loss', 0.0001, patience=5, verbose=True, mode='min')

    logger = pl.loggers.TensorBoardLogger('lightning_logs', name='shape_identification')

    trainer = pl.Trainer(
        checkpoint_callback=model_checkpoint_callback,
        callbacks=[early_stopping_callback],
        check_val_every_n_epoch=5,
        logger=logger,
        gpus=0,
        accelerator='dp', 
        max_epochs=100,
    )

    trainer.fit(model=shape_cnn, datamodule=data_module)
