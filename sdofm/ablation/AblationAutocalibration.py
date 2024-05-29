
import lightning.pytorch as pl
import torch
import torch.nn as nn
from typing import Optional

from ..BaseModule import BaseModule
from ..models import (
    Autocalibration13,
)

from ..finetuning.Autocalibration import HeteroscedasticLoss

class AblationAutocalibration(BaseModule):
    def __init__(
        self,
        # Backbone parameters
        img_size=512,
        channels=9,
        # Head parameters
        output_dim: int = 1,
        loss: str = "mse",
        # all else
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.head = Autocalibration13(
            [channels, img_size, img_size], channels #output_dim
        )

        # set loss function
        match loss:
            case "mse":
                self.loss_function = nn.MSELoss()
            case "heteroscedastic":
                self.loss_function = HeteroscedasticLoss()
            case _:
                raise NotImplementedError(f"Loss function {loss} not implemented")

    def training_step(self, batch, batch_idx):
        degraded_img, degrad_factor, orig_img = batch
        # x = self.encoder(degraded_img)
        # # print("Autocal training: encoder out dim", x.shape)
        # # x_hat = self.autoencoder.unpatchify(x_hat)
        # x = self.decoder(x)
        # print("Autocal training: decoder out dim", x.shape)
        y_hat = self.head(degraded_img) # does not support multiframe/temporal, [:,:,0,:,:], returns (mean, log_var)x(batch)x(output dim)
        loss = self.loss_function(y_hat[0, :, :], degrad_factor)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        degraded_img, degrad_factor, orig_img = batch
        y_hat = self.head(degraded_img) # does not support multiframe/temporal, [:,:,0,:,:], returns (mean, log_var)x(batch)x(output dim)
        loss = self.loss_function(y_hat[0, :, :], degrad_factor)
        self.log("val_loss", loss)
