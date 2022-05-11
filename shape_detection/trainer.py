"""Pytorch-Lightning trainer.

The trainer currently trains the CNN on the randomly generated data.
If you want to debug, there is also an MLP-version (very bad) and an
MLP which works on the MNIST-data (very boring). Check mlp_model.py for
the model and data_loader.py for the data-loaders.
"""

import pytorch_lightning as pl

from data_loader import ShapeIterableDataset, ShapeIterableDataLoader
from cnn_model import ShapeDetectorModelCNN

from shape_generator import ShapeTypes, Colouring

if __name__ == '__main__':
    # Hyperparameters:
    N_x, N_y, N_c, N_target = 50, 50, 3, len(ShapeTypes)
    colouring = Colouring.RANDOM_PIXELS
    batch_size = 1000
    learning_rate = 0.00001
    optimise_learning_rate = False  # If set to True, the learning-rate-setting will be ignored, obviously.
    max_epochs = 300
    load_model = False
    load_model_path = 'checkpoints/best.ckpt'

    # Set up model and data:
    data_module = ShapeIterableDataLoader(N_x=N_x, N_y=N_y, batch_size=batch_size, colouring=colouring)
    if load_model:
        shape_cnn = ShapeDetectorModelCNN.load_from_checkpoint(
            checkpoint_path='checkpoints/best.ckpt',
            N_c=N_c, N_target=N_target,
        )
    else:
        shape_cnn = ShapeDetectorModelCNN(N_c=N_c, N_target=N_target, learning_rate=learning_rate)
    
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        verbose=True,
        save_last=True,
        save_top_k=1,
        #filename='shape_identification_best',
        filename='{epoch}-{step}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
    )

    #early_stopping_callback = pl.callbacks.EarlyStopping('val_loss', 0.0001, patience=5, verbose=True, mode='min')

    callbacks = [model_checkpoint_callback]

    logger = pl.loggers.TensorBoardLogger('lightning_logs', name='shape_identification', version='full_train')

    trainer = pl.Trainer(
        #checkpoint_callback=model_checkpoint_callback,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        logger=logger,
        gpus=1,
        max_epochs=max_epochs,
    )

    if optimise_learning_rate:
        lr_finder = trainer.tuner.lr_find(shape_cnn, data_module, max_lr = 0.01)
        shape_cnn.learning_rate = lr_finder.suggestion()
        print(f'Learning rate is optimised to {lr_finder.suggestion()}.')

    trainer.fit(model=shape_cnn, datamodule=data_module)
