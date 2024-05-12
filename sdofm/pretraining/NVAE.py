# Adapted from https://github.com/FrontierDevelopmentLab/sdolatent/blob/main/scripts/train_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import utils
from ..BaseModule import BaseModule
from ..models import NVIDIAAutoEncoder
from ..models.nvae.utils import (
    get_arch_cells,
    kl_balancer,
    kl_balancer_coeff,
    kl_coeff,
    reconstruction_loss,
)
from ..benchmarks.reconstruction import get_batch_metrics


class NVAE(BaseModule):
    def __init__(
        self,
        # NVAE specific
        use_se=True,
        res_dist=True,
        num_x_bits=8,
        num_latent_scales=3,
        num_groups_per_scale=1,
        num_latent_per_group=1,
        ada_groups=True,
        min_groups_per_scale=1,
        num_channels_enc=30,
        num_channels_dec=30,
        num_preprocess_blocks=2,
        num_preprocess_cells=2,
        num_cell_per_cond_enc=2,
        num_postprocess_blocks=2,
        num_postprocess_cells=2,
        num_cell_per_cond_dec=2,
        num_mixture_dec=1,
        num_nf=2,
        kl_anneal_portion=0.3,
        kl_const_portion=0.0001,
        kl_const_coeff=0.0001,
        # additional opt
        weight_decay_norm_anneal=True,
        weight_decay_norm_init=1.0,
        weight_decay_norm=1e-2,
        # masking
        hmi_mask=None,
        mask_with_hmi_threshold=0.001,
        # pass to BaseModule
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Set up for model arg dictionary
        self.model_args = utils.AttributeDict()
        self.model_args.dataset = "sdoml"  # WARN: fixed
        self.model_args.use_se = use_se
        self.model_args.res_dist = res_dist
        self.model_args.num_x_bits = num_x_bits
        self.model_args.num_latent_scales = num_latent_scales
        self.model_args.num_groups_per_scale = num_groups_per_scale
        self.model_args.num_latent_per_group = num_latent_per_group
        self.model_args.ada_groups = ada_groups
        self.model_args.min_groups_per_scale = min_groups_per_scale
        self.model_args.num_channels_enc = num_channels_enc
        self.model_args.num_channels_dec = num_channels_dec
        self.model_args.num_preprocess_blocks = num_preprocess_blocks
        self.model_args.num_preprocess_cells = num_preprocess_cells
        self.model_args.num_cell_per_cond_enc = num_cell_per_cond_enc
        self.model_args.num_postprocess_blocks = num_postprocess_blocks
        self.model_args.num_postprocess_cells = num_postprocess_cells
        self.model_args.num_cell_per_cond_dec = num_cell_per_cond_dec
        self.model_args.num_mixture_dec = num_mixture_dec
        self.model_args.num_nf = num_nf
        self.model_args.kl_anneal_portion = kl_anneal_portion
        self.model_args.kl_const_portion = kl_const_portion
        self.model_args.kl_const_coeff = kl_const_coeff

        # iters
        # self.num_total_training_iter = self.num_training_batches * self.max_epochs
        # self.num_total_testing_iter = self.num_test_batches * self.max_epochs
        # self.num_total_validation_iter = self.num_val_batches * self.max_epochs

        # opt
        self.weight_decay_norm_anneal = weight_decay_norm_anneal
        self.weight_decay_norm_init = weight_decay_norm_init
        self.weight_decay_norm = weight_decay_norm

        # mask
        if hmi_mask is not None:
            self.hmi_mask = hmi_mask
            self.hmi_mask_value = mask_with_hmi_threshold

        arch_instance = get_arch_cells("res_mbconv")
        self.autoencoder = NVIDIAAutoEncoder(self.model_args, None, arch_instance)

        z, z_shapes = self.autoencoder.encode(torch.randn(1, 12, 512, 512))

        # device agnostic lightning
        self.register_buffer("z_dim", torch.Tensor(z.nelement()))
        self.register_buffer("z_shapes", torch.Tensor(z_shapes))
        self.register_buffer(
            "alpha_i",
            kl_balancer_coeff(
                num_scales=self.autoencoder.num_latent_scales,
                groups_per_scale=self.autoencoder.groups_per_scale,
                fun="square",
            ),
        )

    def encode(self, x):
        z, _ = self.autoencoder.encode(x)
        return z

    def decode(self, z):
        logits = self.autoencoder.decode(z, self.z_shapes)
        output = self.autoencoder.decoder_output(logits)
        x = output.sample()
        if self.hmi_mask is not None:
            x = utils.apply_hmi_mask(x, self.hmi_mask, self.hmi_mask_value)
        return x

    def sample(self, n=1, t=1.0):
        logits = self.autoencoder.sample(n, t)
        output = self.autoencoder.decoder_output(logits)
        x = output.sample()
        if self.hmi_mask is not None:
            x = utils.apply_hmi_mask(x, self.hmi_mask, self.hmi_mask_value)
        return x

    # def forward(self, x):
    #     logits, log_q, log_p, kl_all, kl_diag = self.model(x)
    #     output = self.model.decoder_output(logits)
    #     x = output.sample()
    #     if self.hmi_mask is not None:
    #         x = utils.apply_hmi_mask(x, self.hmi_mask, self.hmi_mask_value)
    #     return x

    def loss(self, x, num_total_iter):
        logits, log_q, log_p, kl_all, kl_diag = self.autoencoder(x)

        output = self.autoencoder.decoder_output(logits)
        if self.hmi_mask is not None:
            mu = output.dist.mu
            mu = utils.apply_hmi_mask(mu, self.hmi_mask, self.hmi_mask_value)
            output.dist.mu = mu

        _kl_coeff = kl_coeff(
            self.global_step,  # from pytorch lightning
            self.model_args.kl_anneal_portion * num_total_iter,
            self.model_args.kl_const_portion * num_total_iter,
            self.model_args.kl_const_coeff,
        )  # .to(self.device)

        # print(torch.max(x), torch.min(x))
        # print(output, x.shape, self.autoencoder.crop_output)
        x = output.sample()
        # print(torch.max(x), torch.min(x))
        # print(x)
        recon_loss = reconstruction_loss(output, x, crop=self.autoencoder.crop_output)
        balanced_kl, kl_coeffs, kl_vals = kl_balancer(
            kl_all, _kl_coeff, kl_balance=True, alpha_i=self.alpha_i
        )

        nelbo_batch = recon_loss + balanced_kl
        loss = torch.mean(nelbo_batch)

        norm_loss = self.autoencoder.spectral_norm_parallel()
        bn_loss = self.autoencoder.batchnorm_loss()

        if self.weight_decay_norm_anneal:
            assert (
                self.weight_decay_norm_init > 0 and self.weight_decay_norm > 0
            ), "init and final wdn should be positive."
            wdn_coeff = (1.0 - _kl_coeff) * np.log(
                self.weight_decay_norm_init
            ) + _kl_coeff * np.log(self.weight_decay_norm)
            wdn_coeff = np.exp(wdn_coeff)
        else:
            wdn_coeff = self.weight_decay_norm
        loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        return loss

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        loss = self.loss(
            batch, self.trainer.num_training_batches * self.trainer.max_epochs
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(
            batch, self.trainer.num_val_batches[0] * self.trainer.max_epochs
        )
        self.log("val_loss", loss)
        return loss
    
    def on_validation_epoch_end(self):
        x = next(iter(self.val_dataloader()))
        x = x.to(self.device)
        _, x_hat, _ = self.autoencoder(x)
        x_hat = self.autoencoder.unpatchify(x_hat)
        channels = ["131A","1600A","1700A","171A","193A","211A","304A","335A","94A"]    

        batch_metrics = get_batch_metrics(x, x_hat, channels)

        for k, v in batch_metrics.items():
            self.log(f"val_{k}", v, sync_dist=True)
