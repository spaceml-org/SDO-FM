import lightning.pytorch as pl
import torch.nn.functional as F
import torch
from ..BaseModule import BaseModule
from ..models import MaskedAutoencoderViT3D
from ..benchmarks import reconstruction as bench_recon
from sdofm.constants import ALL_WAVELENGTHS

from sdofm.models import UNet


class BrightSpots(BaseModule):
    def __init__(
        self,
        # backbone specific
        n_channels: int=12,
        n_classes: int= 1,
        bilinear: bool=True,
        use_embeddings_block: bool=True,
        size_factor:int= 4,
        # pass to BaseModule
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_channels = n_channels

        self.model = UNet(
            n_channels=n_channels, n_classes=n_classes, bilinear=bilinear, use_embeddings_block=use_embeddings_block, size_factor=size_factor
        )

    def training_step(self, batch, batch_idx):
        image_stack, bright_spots = batch['image_stack'], batch['bright_spots']
        y_hat = self.model.forward(image_stack).repeat_interleave(self.n_channels, 1)
        loss = torch.sqrt(F.mse_loss(y_hat, bright_spots))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image_stack, bright_spots = batch['image_stack'], batch['bright_spots']
        x_hat = self.model.forward(image_stack).repeat_interleave(self.n_channels, 1)
        loss = torch.sqrt(F.mse_loss(x_hat, bright_spots))
        self.log("val_loss", loss, sync_dist=True)

