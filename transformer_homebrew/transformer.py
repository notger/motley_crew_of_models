"""High level transformer architecture.

Contains the high-level class you want to import should you
ever foolishly decide that you want to use this code on a real problem.
"""

import torch
import pytorch_lightning as pl

from encoder import Encoder
from decoder import Decoder


class Transformer(pl.LightningModule):
    def __init__(
        self,
        N_encoder_layer: int = 4,
        N_decoder_layer: int = 4,
        dim_mdl: int = 512,
        N_heads: int = 4,
        dim_ff: int = 2048,
        learning_rate: float = 0.0001,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Set some hyperparameters:
        self.lr = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()

        # Generate the architecture:
        self.encoder = Encoder(
            N_encoder_layer=N_encoder_layer,
            dim_mdl=dim_mdl,
            N_heads=N_heads,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.decoder = Decoder(
            N_decoder_layer=N_decoder_layer,
            dim_mdl=dim_mdl,
            N_heads=N_heads,
            dim_ff=dim_ff,
            dropout=dropout,
        )

    def feedforward(self, x: torch.Tensor, y: torch.Tensor):
        """Feedforward step.

        Params:
            x: Source sequence
            y: Target sequence
        """
        return self.decoder(y, self.encoder(x))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr = self.lr)

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
