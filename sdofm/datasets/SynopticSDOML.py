# Adapted from https://github.com/vale-salvatelli/sdo-autocal_pub/blob/master/src/sdo/pipelines/training_pipeline.py

import math

import numpy as np
import torch
from overrides import override

from .SDOML import SDOMLDataModule, SDOMLDataset


class SynopticSDOMLDataset(SDOMLDataset):
    def __init__(
        self,
        *args,
        band_width_pixels,  # band to select in pixels (not degrees!)
        band_crop_pixels,  # number of pixels to crop above and below band
        band_only,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.band_width_pixels = band_width_pixels
        self.band_crop_pixels = band_crop_pixels
        self.band_only = band_only
        # print(kwargs)

    # def __getitem__(self, idx):
    #     orig_img = torch.Tensor(super().__getitem__(idx))

    #     if self.threshold_black:
    #         orig_img[orig_img <= self.threshold_black_value] = 0

    #     # If this is being used as the test dataset, in some cases we might want to
    #     # flip the images to ensure no stray pixels are sitting around influencing
    #     # the results between the training and testing datasets.
    #     if self.flip_test_images:
    #         orig_img = torch.flip(orig_img, [2])

    #     if self.noise_image:
    #         dimmed_img = create_noise_image(
    #             self.num_channels, self.scaled_height, self.scaled_width
    #         )
    #     else:
    #         dimmed_img = orig_img.clone()

    #     SAFETY = 1000
    #     dim_factor = torch.zeros(self.num_channels)
    #     while any(dim_factor < self.min_alpha) and SAFETY > 0:
    #         dim_factor = self.max_alpha * torch.rand(self.num_channels)
    #         SAFETY -= 1

    #     for c in range(self.num_channels):
    #         dimmed_img[c] = dimmed_img[c] * dim_factor[c]

    #     # Note: For efficiency reasons, don't send each item to the GPU;
    #     # rather, later, send the entire batch to the GPU.
    #     return dimmed_img, dim_factor, orig_img

    def get_hmi_band(self, idx):
        """Get HMI image for a given index.
        Returns a numpy array of shape (num_channels, num_frames, height, width).
        """
        hmi_image_dict = {}
        for component in self.components:
            hmi_image_dict[component] = []
            for frame in range(self.num_frames):

                idx_row_element = self.aligndata.iloc[idx + frame]
                idx_component = idx_row_element[f"idx_{self.components[0]}"]
                year = str(idx_row_element.name.year)

                img = self.hmi_data[year][component][idx_component, :, :]

                if self.mask is not None:
                    img = img * self.mask

                hmi_image_dict[component].append(img)

                if self.normalizations:
                    hmi_image_dict[component][-1] -= self.normalizations["HMI"][
                        component
                    ]["mean"]
                    hmi_image_dict[component][-1] /= self.normalizations["HMI"][
                        component
                    ]["std"]

        hmi_image = np.array(list(hmi_image_dict.values()))

        img_height = hmi_image.shape[2]
        img_width = hmi_image.shape[3]

        hmi_band = hmi_image[
            :,
            :,
            int(img_width / 2)
            - 1
            - (self.band_width_pixels - 1) : int(img_width / 2)
            + (self.band_width_pixels - 1),
            self.band_crop_pixels : img_height - self.band_crop_pixels,
        ]

        return hmi_band

    def get_hmi_synpotic(self, idx):
        band = self.get_hmi_band(idx)
        return band


class SynopticSDOMLDataModule(SDOMLDataModule):
    def __init__(
        self,
        *args,
        band_width_pixels=1,
        band_crop_pixels=54,
        band_only=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.band_width_pixels = band_width_pixels
        self.band_crop_pixels = band_crop_pixels
        self.band_only = band_only

    def setup(self, stage=None):
        self.train_ds = SynopticSDOMLDataset(
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
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            min_date=self.min_date,
            max_date=self.max_date,
            # Synoptic
            band_width_pixels=self.band_width_pixels,
            band_crop_pixels=self.band_crop_pixels,
            band_only=self.band_only,
        )

        self.valid_ds = SynopticSDOMLDataset(
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
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            min_date=self.min_date,
            max_date=self.max_date,
            # Synoptic
            band_width_pixels=self.band_width_pixels,
            band_crop_pixels=self.band_crop_pixels,
            band_only=self.band_only,
        )

        self.test_ds = SynopticSDOMLDataset(
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
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            min_date=self.min_date,
            max_date=self.max_date,
            # Synoptic
            band_width_pixels=self.band_width_pixels,
            band_crop_pixels=self.band_crop_pixels,
            band_only=self.band_only,
        )
