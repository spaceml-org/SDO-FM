# Adapted from https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE

import sys

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchvision
from torch.nn import HuberLoss

from ..BaseModule import BaseModule
from ..models import (
    Autocalibration13Head,
    ConvTransformerTokensToEmbeddingNeck,
    HybridIrradianceModel,
    PrithviEncoder,
)


class VirtualEVE_bUNet(BaseModule):

    def __init__(
        self,
        # Backbone parameters
        # img_size: int = 512,
        # patch_size: int = 16,
        # embed_dim: int = 128,
        # num_frames: int = 5,
        # Neck parameters
        num_neck_filters: int = 32,
        # Head parameters
        # d_input=None,
        cnn_model: str = "efficientnet_b3",
        lr_linear: float = 0.01,
        lr_cnn: float = 0.0001,
        cnn_dp: float = 0.75,
        epochs_linear: int = 50,
        d_output=None,
        eve_norm=None,
        # for finetuning
        backbone: object = None,
        freeze_encoder: bool = True,
        # all else
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.encoder = backbone

        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        # HEAD
        self.head = HybridIrradianceModel(
            # virtual eve
            d_input=num_neck_filters,
            d_output=d_output,
            eve_norm=eve_norm,
            # from config
            cnn_model=cnn_model,
            lr_linear=lr_linear,
            lr_cnn=lr_cnn,
            cnn_dp=cnn_dp,
            epochs_linear=epochs_linear,
            # general - might need to be ported over correctly
            # optimiser=self.cfg.model.opt.optimiser,
            # lr=self.cfg.model.opt.learning_rate,
            # weight_decay=self.cfg.model.opt.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        image_stack, eve = batch
        x = self.encoder.forward_encode(image_stack)  # imgs[:, :9, :, :, :]
        embeddings = self.encoder.forward_from_embeddings(x).reshape(-1)
        y_hat = self.head(embeddings)
        loss = self.head.loss_func(y_hat, eve[:, :38])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image_stack, eve = batch
        x = self.encoder.forward_encode(image_stack)  # imgs[:, :9, :, :, :]
        embeddings = self.encoder.forward_from_embeddings(x).reshape(-1)
        y_hat = self.head(embeddings)
        loss = self.head.loss_func(y_hat, eve[:, :38])
        self.log("val_loss", loss)
        return loss
