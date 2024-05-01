# Modified from: https://github.com/isaaccorley/prithvi-pytorch/blob/main/prithvi_pytorch/model.py
# This is the main model file for the Prithvi model.

import os
from typing import Optional

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

from .. import utils
from . import ConvTransformerTokensToEmbeddingNeck, MaskedAutoencoderViT3D

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]
MEAN = [
    "775.2290211032589",
    "1080.992780391705",
    "1228.5855250417867",
    "2497.2022620507532",
    "2204.2139147975554",
    "1610.8324823273745",
]
STD = [
    "1281.526139861424",
    "1270.0297974547493",
    "1399.4802505642526",
    "1368.3446143747644",
    "1291.6764008585435",
    "1154.505683480695",
]


class PrithviEncoder(nn.Module):
    def __init__(
        self,
        mae,
    ):
        super().__init__()
        # cfg.model.mae.num_frames = num_frames
        # cfg.model.mae.in_chans = in_chans
        # cfg.model_args.img_size = img_size

        # self.embed_dim = embed_dim
        # self.depth = cfg.model_args.depth
        # self.num_frames = num_frames
        # self.in_chans = in_chans
        # self.img_size = img_size
        # self.patch_size = cfg.model_args.patch_size
        # encoder = MaskedAutoencoderViT3D(
        #     img_size,
        #     patch_size,
        #     num_frames,
        #     tubelet_size,
        #     in_chans,
        #     embed_dim,
        #     depth,
        #     num_heads,
        #     decoder_embed_dim,
        #     decoder_depth,
        #     decoder_num_heads,
        #     mlp_ratio,
        #     norm_layer,
        #     norm_pix_loss,
        # )
        self.encoder = mae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add a temporal dimension if num_frames = 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        x, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0)

        # Squeeze temporal dim if t=1
        x = x.squeeze(dim=2)
        return x

    def forward_features(
        self,
        x: torch.Tensor,
        n: list[int],
        mask_ratio: float = 0.0,
        reshape: bool = True,
        norm=False,
    ):
        # add a temporal dimension if num_frames = 1
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b c () h w")

        x = self.encoder.get_intermediate_layers(
            x, n=n, mask_ratio=mask_ratio, reshape=reshape, norm=norm
        )
        return x


class PrithviViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cfg_path: str,
        ckpt_path: Optional[str] = None,
        in_chans: int = 6,
        img_size: int = 224,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PrithviEncoder(
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            num_frames=1,
            in_chans=in_chans,
            img_size=img_size,
        )
        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = nn.Linear(
            in_features=self.encoder.embed_dim, out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x[:, 0]  # cls token
        return self.head(x)


class PrithviUnet(Unet):
    def __init__(
        self,
        num_classes: int,
        cfg_path: str,
        ckpt_path: Optional[str] = None,
        in_chans: int = 6,
        img_size: int = 224,
        n: list[int] = [2, 5, 8, 11],
        norm: bool = True,
        decoder_channels: list[int] = [256, 128, 64, 32],
        freeze_encoder: bool = False,
    ):
        super().__init__(encoder_weights=None)
        assert len(n) == 4, "Num intermediate blocks must be 5"
        self.n = n
        self.num_classes = num_classes
        self.norm = norm
        self.encoder = PrithviEncoder(
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            num_frames=1,
            in_chans=in_chans,
            img_size=img_size,
        )
        assert all(
            [i < self.encoder.depth for i in n]
        ), "intermediate block index must be less than the ViT depth"

        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self._depth = 4
        self._in_channels = in_chans
        self._out_channels = [in_chans] + [self.encoder.embed_dim] * len(self.n)
        self.upsample = nn.ModuleList(
            [nn.UpsamplingBilinear2d(scale_factor=s) for s in self.scale_factors]
        )

        self.decoder = UnetDecoder(
            encoder_channels=self._out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )
        self.classification_head = None
        self.name = "u-prithvi"
        self.initialize()

    def forward(self, x):
        features = self.encoder.forward_features(
            x, n=self.n, mask_ratio=0.0, reshape=True, norm=self.norm
        )
        features = [up(f) for up, f in zip(self.upsample, features)]
        features = [x] + list(features)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks

    @property
    def scale_factors(self):
        num_tokens = self.encoder.img_size // self.encoder.patch_size
        sizes = [
            self.encoder.img_size // i
            for i in [2 ** (i + 1) for i in range(len(self.n))]
        ]
        return [s // num_tokens for s in sizes]


class PrithviEncoderDecoder(nn.Module):
    def __init__(
        self,
        # MAE
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
        # other
        num_classes: int = 3,
        # cfg_path: str,
        # ckpt_path: Optional[str] = None,
        # in_chans: int = 6,
        # img_size: int = 224,
        freeze_encoder: bool = False,
        num_neck_filters: int = 32,
    ):
        super().__init__()

        self.num_classes = num_classes

        # BACKBONE
        self.encoder = PrithviEncoder(
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
        self.head = nn.Conv2d(
            in_channels=num_neck_filters,
            out_channels=num_classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x
