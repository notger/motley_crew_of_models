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
    learning_rate = 0.0001
    optimise_learning_rate = False  # If set to True, the learning-rate-setting will be ignored, obviously.
    max_epochs = 300
    analyse_last_model_trained = False

    # Set up model and data:
    data_module = ShapeIterableDataLoader(N_x=N_x, N_y=N_y, batch_size=batch_size, colouring=colouring)
    shape_cnn = ShapeDetectorModelCNN(N_c=N_c, N_target=N_target, learning_rate=learning_rate)

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        verbose=True,
        save_last=True,
        save_top_k=1,
        #filename='shape_identification_best',
        filename='{epoch}-{step}-{val_loss:.4f}',
        monitor=['train_loss', 'val_loss'],
        mode='min',
    )

    #early_stopping_callback = pl.callbacks.EarlyStopping('val_loss', 0.0001, patience=5, verbose=True, mode='min')

    logger = pl.loggers.TensorBoardLogger('lightning_logs', name='shape_identification', version='full_train')

    trainer = pl.Trainer(
        checkpoint_callback=model_checkpoint_callback,
        #callbacks=[early_stopping_callback],
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

    # After training, let's do some analysis:
    if analyse_last_model_trained:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        test_data = ShapeIterableDataset(N_x, N_y, colouring, batch_size=batch_size)

        # Generate labels and predictions:
        predictions = []
        labels = []
        shape_cnn.freeze()
        for im_tensor, label in test_data:
            yp = shape_cnn(im_tensor.unsqueeze(dim=0))
            predictions.append(yp.numpy().argmax())
            labels.append(label)
        shape_cnn.unfreeze()

        # Plot the results:
        sns.set(rc={"figure.figsize":(15, 10)})
        ax = sns.heatmap(
            confusion_matrix(labels, predictions, normalize='true'), 
            annot=True
        )
        _ = ax.set(
            xlabel='predicted as', 
            ylabel='true label', 
            xticklabels=[s.name for s in ShapeTypes],
            yticklabels=[s.name for s in ShapeTypes]
        )
        plt.show()
