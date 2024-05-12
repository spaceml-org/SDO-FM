import time

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..BaseModule import BaseModule
from ..models import MaskedAutoencoderViT3D, PrithviEncoder
from ..benchmarks.reconstruction import get_batch_metrics


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

        self.autoencoder = MaskedAutoencoderViT3D(
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
        # self.autoencoder = PrithviEncoder(self.mae)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch
        loss, x_hat, mask = self.autoencoder(x)
        x_hat = self.autoencoder.unpatchify(x_hat)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss, x_hat, mask = self.autoencoder(x)
        x_hat = self.autoencoder.unpatchify(x_hat)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)

    def on_validation_epoch_end(self):
        x = next(iter(self.val_dataloader()))
        x = x.to(self.device)
        _, x_hat, _ = self.autoencoder(x)
        x_hat = self.autoencoder.unpatchify(x_hat)
        channels = ["131A","1600A","1700A","171A","193A","211A","304A","335A","94A"]    

        batch_metrics = get_batch_metrics(x, x_hat, channels)

        for k, v in batch_metrics.items():
            self.log(f"val_{k}", v, sync_dist=True)
