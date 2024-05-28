import time

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..BaseModule import BaseModule
from ..models import PrithviEncoder, SolarAwareMaskedAutoencoderViT3D
from ..benchmarks.reconstruction import get_batch_metrics
from ..benchmarks import reconstruction as bench_recon
from sdofm.constants import ALL_WAVELENGTHS

class SAMAE(BaseModule):
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
        # masking
        masking_type="random",  # 'random' or 'solar_aware'
        active_region_mu_degs=15.73,
        active_region_std_degs=6.14,
        active_region_scale=1.0,
        active_region_abs_lon_max_degs=60,
        active_region_abs_lat_max_degs=60,
        #
        checkpoint_path=None,
        # pass to BaseModule
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.autoencoder = SolarAwareMaskedAutoencoderViT3D(
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
            masking_type,  # 'random' or 'solar_aware'
            active_region_mu_degs,
            active_region_std_degs,
            active_region_scale,
            active_region_abs_lon_max_degs,
            active_region_abs_lat_max_degs,
        )
        if checkpoint_path is not None:
            state_dict = torch.load(
                checkpoint_path, map_location=self.autoencoder.device
            )

            #            if num_frames != 3:
            #                del state_dict["pos_embed"]
            #                del state_dict["decoder_pos_embed"]
            #
            #            if in_chans != 6:
            #                del state_dict["patch_embed.proj.weight"]
            #                del state_dict["decoder_pred.weight"]
            #                del state_dict["decoder_pred.bias"]

            self.autoencoder.load_state_dict(state_dict, strict=False)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch
        loss, x_hat, mask = self.autoencoder(x)
        x_hat = self.autoencoder.unpatchify(x_hat)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss, x_hat, mask = self.autoencoder(x)
        x_hat = self.autoencoder.unpatchify(x_hat)
        self.validation_metrics.append(bench_recon.get_metrics(x[0,:,0,:,:], x_hat[0,:,0,:,:], ALL_WAVELENGTHS)) # shouldn't be hardcoded to all wavelengths and frames dim is not considered
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, sync_dist=True)

    def forward(self, x):
        loss, x_hat, mask = self.autoencoder(x)
        x_hat = self.autoencoder.unpatchify(x_hat)
        return loss, x_hat, mask

    def predict_step(self, batch):  # loss, x_hat, mask
        return self(batch)

    def on_validation_epoch_end(self):
        # retrieve the validation outputs (images and reconstructions)
        # TODO: reconstruction should apply where num_frames > 1
        # x, x_hat = torch.stack(self.validation_step_outputs['x'])[:,0,:,0,:,:], torch.stack(self.validation_step_outputs['x_hat'])[:,0,:,0,:,:]

        # TODO: these shouldn't be hardcoded
        # channels = ["131A","1600A","1700A","171A","193A","211A","304A","335A","94A"]

        # generate metrics 
        # batch_metrics = get_batch_metrics(x, x_hat, channels)
        merged_metrics = bench_recon.merge_metrics(self.validation_metrics)
        batch_metrics = bench_recon.mean_metrics(merged_metrics)
       
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            from pandas import DataFrame
            import wandb
            # this only occurs on rank zero only 
            df = DataFrame(batch_metrics)
            df['metric'] = df.index
            cols = df.columns.tolist()
            self.logger.log_table(key='val_reconstruction', dataframe=df[cols[-1:] + cols[:-1]], step=self.validation_step)
            for k, v in batch_metrics.items():
            # sync_dist as this tries to include all
                for i,j in v.items():
                    self.log(f"val_{k}_{i}", j)

            # table = wandb.Table(columns=["ID", "Image"])
            # model_artifact = wandb.Artifact("model", type="model")
            # model_artifact.add_reference(f"gs://sdofm-checkpoints/{wandb.run.id}-{wandb.run.name}/model-step{wandb.run.step}.ckpt")
        else:
            print(batch_metrics)
            for k in batch_metrics.keys():
                batch_metrics[k]['channel'] = k
            for k, v in batch_metrics.items():
                # sync_dist as this tries to include all
                self.log_dict(v, sync_dist=True) # This doesn't work?
        
        # reset
        # self.validation_step_outputs['x'].clear()
        # self.validation_step_outputs['x_hat'].clear()
        self.validation_metrics.clear()
