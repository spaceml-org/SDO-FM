# Adapted from https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE

import sys
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchvision
from torch.nn import HuberLoss

from ..BaseModule import BaseModule


def unnormalize(y, eve_norm):
    eve_norm = torch.tensor(eve_norm).float()
    norm_mean = eve_norm[0]
    norm_stdev = eve_norm[1]
    y = y * norm_stdev[None] + norm_mean[None]  # .to(y) .to(y)
    return y


class CNNIrradianceModel(BaseModule):

    def __init__(self, d_input, d_output, eve_norm, model="efficientnet_b0", dp=0.75):
        super().__init__()
        # self.eve_norm = eve_norm

        if True:
            if model == "efficientnet_b0":
                model = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
            elif model == "efficientnet_b1":
                model = torchvision.models.efficientnet_b1(weights="IMAGENET1K_V1")
            elif model == "efficientnet_b2":
                model = torchvision.models.efficientnet_b2(weights="IMAGENET1K_V1")
            elif model == "efficientnet_b3":
                model = torchvision.models.efficientnet_b3(weights="IMAGENET1K_V1")
            elif model == "efficientnet_b4":
                model = torchvision.models.efficientnet_b4(weights="IMAGENET1K_V1")
            elif model == "efficientnet_b5":
                model = torchvision.models.efficientnet_b5(weights="IMAGENET1K_V1")
            elif model == "efficientnet_b6":
                model = torchvision.models.efficientnet_b6(weights="IMAGENET1K_V1")
            elif model == "efficientnet_b7":
                model = torchvision.models.efficientnet_b7(weights="IMAGENET1K_V1")

        conv1_out = model.features[0][0].out_channels
        model.features[0][0] = nn.Conv2d(
            d_input,
            conv1_out,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )

        lin_in = model.classifier[1].in_features
        classifier = nn.Sequential(
            nn.Dropout(p=dp, inplace=True),
            nn.Linear(in_features=lin_in, out_features=d_output, bias=True),
        )
        model.classifier = classifier

        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.p = dp

        self.model = model
        self.loss_func = HuberLoss()

    def forward(self, x):
        x = self.model(x)
        return x


class LinearIrradianceModel(BaseModule):

    def __init__(self, d_input, d_output, eve_norm):
        super().__init__()
        # self.eve_norm = eve_norm
        self.n_channels = d_input
        self.outSize = d_output

        self.model = nn.Linear(2 * self.n_channels, self.outSize)
        self.loss_func = HuberLoss()  # consider MSE

    def forward(self, x):
        mean_irradiance = torch.mean(x, dim=(2, 3))
        std_irradiance = torch.std(x, dim=(2, 3))
        x = self.model(torch.cat((mean_irradiance, std_irradiance), dim=1))
        return x


class HybridIrradianceModel(BaseModule):

    def __init__(
        self,
        d_input,
        d_output,
        eve_norm,
        *args,
        cnn_model="efficientnet_b3",
        lr_linear=0.01,
        lr_cnn=0.0001,
        cnn_dp=0.75,
        ln_params=None,  # used in lambda function out of scope?
        epochs_linear=None,  # out of scope
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        # self.eve_norm = torch.Tensor(self.eve_norm).float() #eve_norm
        self.register_buffer("eve_norm", torch.Tensor(eve_norm).float())
        self.n_channels = d_input
        self.outSize = d_output
        # self.ln_params = ln_params # unused
        self.lr_linear = lr_linear
        self.lr_cnn = lr_cnn
        self.train_mode = "linear"

        self.ln_model = LinearIrradianceModel(d_input, d_output, eve_norm)
        self.cnn_model = CNNIrradianceModel(
            d_input, d_output, eve_norm, model=cnn_model, dp=cnn_dp
        )
        self.loss_func = HuberLoss()

    def forward(self, x):
        return self.ln_model.forward(x) + self.cnn_lambda * self.cnn_model.forward(x)

    def forward_unnormalize(self, x):
        return self.unnormalize(self.forward(x))

    def unnormalize(self, y):
        # eve_norm = torch.tensor(self.eve_norm).float()
        norm_mean = self.eve_norm[0]
        norm_stdev = self.eve_norm[1]
        y = y * norm_stdev[None] + norm_mean[None]  # .to(y) .to(y)
        return y

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)

        # print("t: trying un")
        y = self.unnormalize(y)
        y_pred = self.unnormalize(y_pred)
        # print("t: success un")

        epsilon = sys.float_info.epsilon
        # computing relative absolute error
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()

        self.log_everything("train", loss, av_rae, float(self.cnn_lambda))
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)

        y = self.unnormalize(y)
        y_pred = self.unnormalize(y_pred)

        epsilon = sys.float_info.epsilon
        # computing relative absolute error
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor(
            [
                torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1]
                for i in range(y.shape[0])
            ]
        ).mean()
        # compute mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log_everything(
            "val", loss, av_rae, float(self.cnn_lambda), av_rae_wl, mae, cc
        )

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_func(y_pred, y)

        y = self.unnormalize(y)
        y_pred = self.unnormalize(y_pred)

        epsilon = sys.float_info.epsilon
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor(
            [
                torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1]
                for i in range(y.shape[0])
            ]
        ).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log_everything(
            "test", loss, av_rae, float(self.cnn_lambda), av_rae_wl, mae, cc
        )

        return loss

    def log_everything(
        self, mode, loss, av_rae, train_lambda_cnn, av_rae_wl=None, mae=None, cc=None
    ):
        # the .to() calls are a fix for https://github.com/Lightning-AI/pytorch-lightning/issues/18803

        self.log(
            f"{mode}_loss",
            loss,
            # on_epoch=True,
            prog_bar=True,
            # logger=True,
            # sync_dist=True,
        )
        self.log(
            f"{mode}_RAE",
            av_rae,
            # on_epoch=True,
            # logger=True,
            # sync_dist=True,
        )
        # self.log(
        #         f"{mode}_lambda_cnn",
        #         self.cnn_lambda,
        #         on_epoch=True,
        #         logger=True,
        #         sync_dist=True,
        #     )
        if mode != "train":
            pass
            # self.log(
            #     f"{mode}_MAE",
            #     mae,
            #     on_epoch=True,
            #     # prog_bar=True,
            #     logger=True,
            #     sync_dist=True
            # )
            # self.log(
            #     f"{mode}_RAE",
            #     av_rae,
            #     on_epoch=True,
            #     # prog_bar=True,
            #     logger=True,
            #     sync_dist=True,
            # )
            # [
            #     self.log(
            #         f"{mode}_RAE_{i}",
            #         err,
            #         on_epoch=True,
            #         # prog_bar=True,
            #         logger=True,
            #         sync_dist=True,
            #     )
            #     for i, err in enumerate(av_rae_wl)
            # ]
            # self.log(
            #     f"{mode}_correlation_coefficient",
            #     cc,
            #     on_epoch=True,
            #     # prog_bar=True,
            #     logger=True,
            #     sync_dist=True,
            # )

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.ln_model.parameters()},
                {"params": self.cnn_model.parameters(), "lr": self.lr_cnn},
            ],
            lr=self.lr_linear,
        )

    def set_train_mode(self, mode):
        if mode == "linear":
            self.train_mode = "linear"
            self.cnn_lambda = 0.0
            self.cnn_model.freeze()
            self.ln_model.unfreeze()
        elif mode == "cnn":
            self.train_mode = "cnn"
            self.cnn_lambda = 0.01
            self.cnn_model.unfreeze()
            self.ln_model.freeze()
        else:
            raise NotImplemented(f"Mode not supported: {mode}")
