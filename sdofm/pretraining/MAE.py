import time

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..BaseModule import BaseModule
from ..models import MaskedAutoencoderViT3D, PrithviEncoder
from ..benchmarks import reconstruction as bench_recon
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from sdofm.constants import ALL_WAVELENGTHS

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
        norm_layer='LayerNorm',
        norm_pix_loss=False,
        # pass to BaseModule
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # self.validation_step_outputs = {'x': [], 'x_hat': []}
        self.validation_metrics = []

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss, x_hat, mask = self.autoencoder(x)
        x_hat = self.autoencoder.unpatchify(x_hat)
        loss = F.mse_loss(x_hat, x)
        for i in range(x.shape[0]):
            for frame in range(x.shape[2]):
                self.validation_metrics.append(bench_recon.get_metrics(x[i,:,frame,:,:], x_hat[i,:,frame,:,:], ALL_WAVELENGTHS))
        self.log("val_loss", loss)

    def forward(self, x):
        loss, x_hat, mask = self.autoencoder(x)
        x_hat = self.autoencoder.unpatchify(x_hat)
        return loss, x_hat, mask

    def on_validation_epoch_end(self):
        # retrieve the validation outputs (images and reconstructions)
        merged_metrics = bench_recon.merge_metrics(self.validation_metrics)
        batch_metrics = bench_recon.mean_metrics(merged_metrics)
       
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            from pandas import DataFrame
            # this only occurs on rank zero only 
            df = DataFrame(batch_metrics)
            df['metric'] = df.index
            cols = df.columns.tolist()
            self.logger.log_table(key='val_reconstruction', dataframe=df[cols[-1:] + cols[:-1]], step=self.validation_step)
            for k, v in batch_metrics.items():
            # sync_dist as this tries to include all
                for i,j in v.items():
                    self.log(f"val_{k}_{i}", j)
        else:
            for k in batch_metrics.keys():
                batch_metrics[k]['channel'] = k
            for k, v in batch_metrics.items():
                # sync_dist as this tries to include all
                self.log_dict(v, sync_dist=True) # This doesn't work?
        
        self.validation_metrics.clear()
