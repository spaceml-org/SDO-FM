# Adapted from:https://github.com/vale-salvatelli/sdo-autocal_pub/blob/master/src/sdo/pipelines/autocalibration_pipeline.py

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..BaseModule import BaseModule
from ..models import (
    Autocalibration13,
    ConvTransformerTokensToEmbeddingNeck,
    MaskedAutoencoderViT3D,
    PrithviEncoder,
)


def heteroscedastic_loss(output, gt_output, reduction):
    """
    Args:
        output: NN output values, tensor of shape 2, batch_size, n_channels.
        where the first dimension contains the mean values and the second
        dimension contains the log_var
        gt_output: groundtruth values. tensor of shape batch_size, n_channels
        reduction: if mean, the loss is averaged across the third dimension,
        if summ the loss is summed across the third dimension, if None any
        aggregation is performed

    Returns:
        tensor of size n_channels if reduction is None or tensor of size 0
        if reduction is mean or sum

    """
    precision = torch.exp(-output[1])
    batch_size = output[0].shape[0]
    loss = (
        torch.sum(precision * (gt_output - output[0]) ** 2.0 + output[1], 0)
        / batch_size
    )
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction is None:
        return loss
    else:
        raise ValueError("Aggregation can only be None, mean or sum.")


class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic loss
    """

    def __init__(self, reduction="mean"):
        super(HeteroscedasticLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        return heteroscedastic_loss(output, target, reduction=self.reduction)


class Autocalibration(BaseModule):
    def __init__(
        self,
        # Backbone parameters
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
        # Neck parameters
        num_neck_filters: int = 32,
        # Head parameters
        output_dim: int = 1,
        loss: str = "mse",
        freeze_encoder: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # BACKBONE
        self.mae = MaskedAutoencoderViT3D(
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

        self.encoder = PrithviEncoder(self.mae)

        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        num_tokens = img_size // patch_size

        # NECK
        self.decoder = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=embed_dim,
            output_embed_dim=num_neck_filters,
            Hp=num_tokens,
            Wp=num_tokens,
            drop_cls_token=True,
        )

        # HEAD
        self.head = Autocalibration13(
            [num_neck_filters, num_tokens, num_tokens], output_dim
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
        dimmed_img, dim_factor, orig_img = batch
        x = self.encoder(dimmed_img)
        # x_hat = self.autoencoder.unpatchify(x_hat)
        y_hat = self.head(self.decoder(x))
        loss = self.loss_function(y_hat, dim_factor)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        dimmed_img, dim_factor, orig_img = batch
        x = self.encoder(dimmed_img)
        # x_hat = self.autoencoder.unpatchify(x_hat)
        y_hat = self.head(self.decoder(x))
        loss = self.loss_function(y_hat, dim_factor)
        self.log("val_loss", loss)