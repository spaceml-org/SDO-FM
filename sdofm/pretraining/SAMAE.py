import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from sdofm.constants import ALL_WAVELENGTHS

from ..BaseModule import BaseModule
from ..benchmarks import reconstruction as bench_recon
from ..models import SolarAwareMaskedAutoencoderViT3D


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
        norm_layer="LayerNorm",
        norm_pix_loss=False,
        masking_ratio=0.75,
        # masking
        masking_type="random",  # 'random' or 'solar_aware'
        active_region_mu_degs=15.73,
        active_region_std_degs=6.14,
        active_region_scale=1.0,
        active_region_abs_lon_max_degs=60,
        active_region_abs_lat_max_degs=60,
        # pass to BaseModule
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.validation_metrics = []
        self.masking_ratio = masking_ratio

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
        # if checkpoint_path is not None:
        #     state_dict = torch.load(
        #         checkpoint_path, map_location=self.autoencoder.device
        #     )

        #     #            if num_frames != 3:
        #     #                del state_dict["pos_embed"]
        #     #                del state_dict["decoder_pos_embed"]
        #     #
        #     #            if in_chans != 6:
        #     #                del state_dict["patch_embed.proj.weight"]
        #     #                del state_dict["decoder_pred.weight"]
        #     #                del state_dict["decoder_pred.bias"]

        #     self.autoencoder.load_state_dict(state_dict, strict=False)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat = self.autoencoder.unpatchify(x_hat)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat = self.autoencoder.unpatchify(x_hat)
        self.validation_metrics.append(
            bench_recon.get_metrics(
                x[0, :, 0, :, :], x_hat[0, :, 0, :, :], ALL_WAVELENGTHS
            )
        )  # shouldn't be hardcoded to all wavelengths and frames dim is not considered
        loss = F.mse_loss(x_hat, x)
        for i in range(x.shape[0]):
            for frame in range(x.shape[2]):
                self.validation_metrics.append(
                    bench_recon.get_metrics(
                        x[i, :, frame, :, :], x_hat[i, :, frame, :, :], ALL_WAVELENGTHS
                    )
                )
        self.log("val_loss", loss)

    def forward(self, x):
        loss, x_hat, mask = self.autoencoder(x, mask_ratio=self.masking_ratio)
        x_hat = self.autoencoder.unpatchify(x_hat)
        return loss, x_hat, mask

    def predict_step(self, batch):  # loss, x_hat, mask
        return self(batch)

    def on_validation_epoch_end(self):
        # retrieve the validation outputs (images and reconstructions)

        merged_metrics = bench_recon.merge_metrics(self.validation_metrics)
        batch_metrics = bench_recon.mean_metrics(merged_metrics)

        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
            from pandas import DataFrame

            # this only occurs on rank zero only
            df = DataFrame(batch_metrics)
            df["mean"] = df.mean(numeric_only=True, axis=1)
            df["metric"] = df.index
            cols = df.columns.tolist()
            self.logger.log_table(
                key="val_reconstruction",
                dataframe=df[cols[-1:] + cols[:-1]],
                step=self.validation_step,
            )
            for k, v in batch_metrics.items():
                # sync_dist as this tries to include all
                for i, j in v.items():
                    self.log(f"val_{k}_{i}", j)
        else:
            for k in batch_metrics.keys():
                batch_metrics[k]["channel"] = k
            for k, v in batch_metrics.items():
                # sync_dist as this tries to include all
                self.log_dict(v, sync_dist=True)  # This doesn't work?

        self.validation_metrics.clear()
