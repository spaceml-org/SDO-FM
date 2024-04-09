# Adapted from https://github.com/vale-salvatelli/sdo-autocal_pub/blob/master/src/sdo/pipelines/training_pipeline.py

import math

import numpy as np
import torch

from .SDOML import SDOMLDataModule, SDOMLDataset


def create_noise_image(
    num_channels, scaled_height, scaled_width, base_val=0.5, percent_jitter=0.20
):
    """
    Generate a random image composed of just noise. Note that the values
    given as defaults for base_val and percent_jitter are quick and dirty
    experimentally chosen values to test how the deep net works with
    random noise images.
    """
    rand_max = torch.rand(1)
    results = rand_max * torch.rand(
        num_channels, scaled_height, scaled_width, dtype=torch.float32
    )

    low = base_val * (1.0 - percent_jitter)
    high = base_val * (1.0 + percent_jitter)
    value = (high - low) * np.random.rand() + low
    results[:, math.floor(scaled_height / 2), math.floor(scaled_width / 2)] = value

    return results


class DimmedSDOMLDataset(SDOMLDataset):
    def __init__(
        self,
        *args,
        threshold_black: bool = False,
        noise_image: bool = False,
        flip_test_images: bool = False,
        threshold_black_value: float = 0.0,
        min_alpha: float = 0.0,
        max_alpha: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.threshold_black = threshold_black
        self.noise_image = noise_image
        self.flip_test_images = flip_test_images
        self.threshold_black_value = threshold_black_value
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        # This Dataset is only for use on AIA
        self.num_channels = len(self.wavelengths)

    def __getitem__(self, idx):
        orig_img = super().__getitem__(idx)

        if self.threshold_black:
            orig_img[orig_img <= self.threshold_black_value] = 0

        # If this is being used as the test dataset, in some cases we might want to
        # flip the images to ensure no stray pixels are sitting around influencing
        # the results between the training and testing datasets.
        if self.flip_test_images:
            orig_img = torch.flip(orig_img, [2])

        if self.noise_image:
            dimmed_img = create_noise_image(
                self.num_channels, self.scaled_height, self.scaled_width
            )
        else:
            dimmed_img = orig_img.clone()

        SAFETY = 1000
        dim_factor = torch.zeros(self.num_channels)
        while any(dim_factor < self.min_alpha) and SAFETY > 0:
            dim_factor = self.max_alpha * torch.rand(self.num_channels)
            SAFETY -= 1

        for c in range(self.num_channels):
            dimmed_img[c] = dimmed_img[c] * dim_factor[c]

        # Note: For efficiency reasons, don't send each item to the GPU;
        # rather, later, send the entire batch to the GPU.
        return dimmed_img, dim_factor, orig_img


class DimmedSDOMLDataModule(SDOMLDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        self.train_ds = DimmedSDOMLDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.train_months,
            normalizations=self.normalizations,
        )

        self.valid_ds = DimmedSDOMLDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.val_months,
            normalizations=self.normalizations,
        )

        self.test_ds = DimmedSDOMLDataset(
            self.aligndata,
            self.hmi_data,
            self.aia_data,
            self.eve_data,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.test_months,
            normalizations=self.normalizations,
        )
