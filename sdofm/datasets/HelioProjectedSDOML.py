# Adapted from Daniel Gass' work

import math

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from overrides import override
from sunpy.coordinates import frames
from sunpy.map import Map

from .SDOML import SDOMLDataModule, SDOMLDataset


class HelioProjectedSDOMLDataset(SDOMLDataset):
    def __init__(
        self,
        *args,
        img_size: int = 512,
        long_range: float = 90,  # Goes from 0 - 90. 0 = zero coverage, 90 = full coverage
        lat_range: float = 90,
        fast_mode: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.long_range = long_range
        self.lat_range = lat_range
        self.fast_mode = fast_mode
        self.img_size = img_size

        latitude = np.arange(-lat_range, lat_range, 2 * lat_range / img_size)
        longitude = np.arange(-long_range, long_range, 2 * long_range / img_size)
        grids = np.meshgrid(latitude, longitude)
        xs = grids[0] * u.deg
        ys = grids[1] * u.deg

        # Loads the heliographic stonyhurst coordinate frame and passes constructed coordinate values.
        self.coords = SkyCoord(xs, ys, frame=frames.HeliographicStonyhurst)

        # # This Dataset is only for use on AIA
        # self.num_channels = len(self.wavelengths)

        # This is slightly incorrect but much faster
        self.pixel_lookup_cache = None

    def __getitem__(self, idx):
        # uncomment this for debugging
        # return super().__getitem__(idx)

        if self.get_header:
            imgs, headers = super().__getitem__(idx)
            channels = list(headers.keys())
        else:
            imgs = super().__getitem__(idx)
        # print(imgs.shape)
        # print(headers.keys())
        # print(imgs.shape, headers.shape)

        if self.drop_frame_dim:
            imgs = imgs.reshape((imgs.shape[0], 1, imgs.shape[1], imgs.shape[2]))
            # print(imgs.shape)
            # headers = np.array([headers])

        frame_dim = []

        for f in range(imgs.shape[1]):
            projected_maps = []

            for idx in range(imgs.shape[0]):
                if self.fast_mode:
                    if self.pixel_lookup_cache is None:
                        map = Map(
                            (np.array(imgs[idx, f, :, :]), headers[channels[idx]])
                        )
                        x, y = map.world_to_pixel(self.coords)
                        # Converts pixel locations to integers to use as indices to extract data from image array.
                        self.pixel_lookup_cache = (x.astype(int), y.astype(int))
                        self.get_header = False
                    projected_maps.append(
                        imgs[idx][f][
                            self.pixel_lookup_cache[0], self.pixel_lookup_cache[1]
                        ]
                    )
                else:  # recreate the map for every image for correct headers
                    # are headers for f>1 being correctly collected?
                    map = Map((np.array(imgs[idx, f, :, :]), headers[channels[idx]]))
                    x, y = map.world_to_pixel(self.coords)
                    projected_maps.append(imgs[idx][f][x.astype(int), y.astype(int)])

            frame_dim.append(np.stack(projected_maps))

        return (
            np.stack(frame_dim).transpose([1, 0, 2, 3])
            if not self.drop_frame_dim
            else frame_dim[-1]
        )


class HelioProjectedSDOMLDataModule(SDOMLDataModule):
    def __init__(
        self,
        *args,
        long_range: float = 90,  # Goes from 0 - 90. 0 = zero coverage, 90 = full coverage
        lat_range: float = 90,
        fast_mode: bool = False,
        img_size: int = 512,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.long_range = long_range
        self.lat_range = lat_range
        self.fast_mode = fast_mode
        self.img_size = img_size

        # if not self.get_header:
        #     raise ValueError("Header is required for HelioProjected data.")

        if (
            kwargs["hmi_path"] is not None or kwargs["eve_path"] is not None
        ):  # hmi or eve specified
            raise NotImplementedError(
                "Projected SDOML dataloader is designed for AIA only."
            )

    @override
    def setup(self, stage=None):
        self.train_ds = HelioProjectedSDOMLDataset(
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
            mask=None,
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            get_header=True,
            # Projection Info
            img_size=self.img_size,
            long_range=self.long_range,
            lat_range=self.lat_range,
            fast_mode=self.fast_mode,
        )

        self.valid_ds = HelioProjectedSDOMLDataset(
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
            mask=None,
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            get_header=True,
            # Projection Info
            img_size=self.img_size,
            long_range=self.long_range,
            lat_range=self.lat_range,
            fast_mode=self.fast_mode,
        )

        self.test_ds = HelioProjectedSDOMLDataset(
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
            mask=None,  # self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
            get_header=True,
            # Projection Info
            img_size=self.img_size,
            long_range=self.long_range,
            lat_range=self.lat_range,
            fast_mode=self.fast_mode,
        )
