import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..BaseModule import BaseModule
from ..models import MaskedAutoencoderViT3D


class MAE(BaseModule):
    def __init__(
        self,
        # MAE specific
        img_size=224,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        # pass to BaseModule
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.encoder_decoder = MaskedAutoencoderViT3D(
            img_size,
            patch_size,
            num_frames,
            tubelet_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            mlp_ratio,
            norm_layer,
            norm_pix_loss,
        )

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        x_hat = self.encoder_decoder(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.encoder_decoder(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)
