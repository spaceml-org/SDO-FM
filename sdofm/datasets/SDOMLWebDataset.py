import json
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from torch.utils.data import Dataset
from tqdm import tqdm

from ..constants import ALL_COMPONENTS, ALL_IONS, ALL_WAVELENGTHS


class SDOMLWebDataset(Dataset):
    def __init__(
        self,
        aligndata,
        aia_tar_path,
        hmi_tar_path,
        eve_tar_path,
        components,
        wavelengths,
        ions,
        freq,
        months,
        normalizations=None,
        mask=None,
        num_frames=1,
        drop_frame_dim=False,
        min_date=None,
        max_date=None,
        get_header=False,
    ):
        super().__init__()

        self.aligndata = aligndata
        self.aia_tar_path = aia_tar_path
        self.hmi_tar_path = hmi_tar_path
        self.eve_tar_path = eve_tar_path

        self.components = components if components else ALL_COMPONENTS
        self.wavelengths = wavelengths if wavelengths else ALL_WAVELENGTHS
        self.ions = ions if ions else ALL_IONS

        self.cadence = freq
        self.months = months
        self.normalizations = normalizations
        self.mask = mask
        self.get_header = get_header
        self.num_frames = num_frames
        self.drop_frame_dim = drop_frame_dim

        # Filter alignment data based on months and date range
        self.aligndata = self.aligndata.loc[
            self.aligndata.index.month.isin(self.months), :
        ]
        if min_date and max_date:
            self.aligndata = self.aligndata[
                (self.aligndata.index >= min_date) & (self.aligndata.index <= max_date)
            ]

        # Create WebDataset pipelines
        self.aia_dataset = wds.WebDataset(aia_tar_path) if aia_tar_path else None
        self.hmi_dataset = wds.WebDataset(hmi_tar_path) if hmi_tar_path else None
        self.eve_dataset = wds.WebDataset(eve_tar_path) if eve_tar_path else None

    def __len__(self):
        return len(self.aligndata) - (self.num_frames - 1)

    def __getitem__(self, idx):
        image_stack = None
        header_stack = None
        if self.aia_dataset:
            image_stack, header_stack = self.get_aia_image(idx)

        if self.hmi_dataset:
            hmi_images, hmi_headers = self.get_hmi_image(idx)
            image_stack = np.concatenate((image_stack, hmi_images), axis=0)
            header_stack.update(hmi_headers)

        if not self.get_header:
            if self.eve_dataset:
                eve_data = self.get_eve(idx)
                return image_stack, eve_data
            else:
                return image_stack
        else:
            if self.eve_dataset:
                eve_data = self.get_eve(idx)
                return image_stack, header_stack, eve_data.reshape(-1)
            else:
                return image_stack, header_stack

    def get_aia_image(self, idx):
        aia_image_dict = {}
        aia_header_dict = {}

        for wavelength in self.wavelengths:
            aia_image_dict[wavelength] = []
            if self.get_header:
                aia_header_dict[wavelength] = []

            for frame in range(self.num_frames):
                idx_row_element = self.aligndata.iloc[idx + frame]
                idx_wavelength = idx_row_element[f"idx_{wavelength}"]
                year = str(idx_row_element.name.year)

                # Retrieve data from WebDataset
                with self.aia_dataset as ds:
                    img = ds[f"{year}_{wavelength}_{idx_wavelength}.jpg"]

                if self.mask is not None:
                    img = img * self.mask

                aia_image_dict[wavelength].append(img)

                if self.get_header:
                    try:
                        aia_header_dict[wavelength].append(
                            {
                                keys: values[idx_wavelength]
                                for keys, values in ds.attrs.items()
                            }
                        )
                    except:
                        aia_header_dict[wavelength].append(None)

                if self.normalizations:
                    aia_image_dict[wavelength][-1] -= self.normalizations["AIA"][
                        wavelength
                    ]["mean"]
                    aia_image_dict[wavelength][-1] /= self.normalizations["AIA"][
                        wavelength
                    ]["std"]

        aia_image = np.array(list(aia_image_dict.values()))
        return (
            (aia_image[:, 0, :, :], aia_header_dict)
            if self.drop_frame_dim
            else (aia_image, aia_header_dict)
        )

    def get_hmi_image(self, idx):
        hmi_image_dict = {}
        hmi_header_dict = {}

        for component in self.components:
            hmi_image_dict[component] = []
            if self.get_header:
                hmi_header_dict[component] = []

            for frame in range(self.num_frames):
                idx_row_element = self.aligndata.iloc[idx + frame]
                idx_component = idx_row_element[f"idx_{self.components[0]}"]
                year = str(idx_row_element.name.year)

                # Retrieve data from WebDataset
                with self.hmi_dataset as ds:
                    img = ds[f"{year}_{component}_{idx_component}.jpg"]

                if self.mask is not None:
                    img = img * self.mask

                hmi_image_dict[component].append(img)

                if self.get_header:
                    hmi_header_dict[component].append(
                        {
                            keys: values[idx_component]
                            for keys, values in ds.attrs.items()
                        }
                    )

                if self.normalizations:
                    hmi_image_dict[component][-1] -= self.normalizations["HMI"][
                        component
                    ]["mean"]
                    hmi_image_dict[component][-1] /= self.normalizations["HMI"][
                        component
                    ]["std"]

        hmi_image = np.array(list(hmi_image_dict.values()))
        return (
            (hmi_image[:, 0, :, :], hmi_header_dict)
            if self.drop_frame_dim
            else (hmi_image, hmi_header_dict)
        )

    def get_eve(self, idx):
        eve_ion_dict = {}
        for ion in self.ions:
            eve_ion_dict[ion] = []
            for frame in range(self.num_frames):
                idx_eve = self.aligndata.iloc[idx + frame]["idx_eve"]
                with self.eve_dataset as ds:
                    eve_ion_dict[ion].append(ds[f"{ion}_{idx_eve}.json"])
                if self.normalizations:
                    eve_ion_dict[ion][-1] -= self.normalizations["EVE"][ion]["mean"]
                    eve_ion_dict[ion][-1] /= self.normalizations["EVE"][ion]["std"]

        eve_data = np.array(list(eve_ion_dict.values()), dtype=np.float32)
        return eve_data

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output


class SDOMLWebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hmi_tar_path,
        aia_tar_path,
        eve_tar_path,
        components,
        wavelengths,
        ions,
        frequency,
        batch_size: int = 32,
        num_workers=None,
        val_months=[10, 1],
        test_months=[11, 12],
        holdout_months=[],
        cache_dir="",
        apply_mask=True,
        num_frames=1,
        drop_frame_dim=False,
        min_date=None,
        max_date=None,
    ):
        super().__init__()
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count() // 2
        )
        self.hmi_tar_path = hmi_tar_path
        self.aia_tar_path = aia_tar_path
        self.eve_tar_path = eve_tar_path
        self.batch_size = batch_size
        self.cadence = frequency
        self.val_months = val_months
        self.test_months = test_months
        self.holdout_months = holdout_months
        self.cache_dir = cache_dir
        self.num_frames = num_frames
        self.drop_frame_dim = drop_frame_dim
        self.min_date = pd.to_datetime(min_date) if min_date is not None else None
        self.max_date = pd.to_datetime(max_date) if max_date is not None else None
        self.isAIA = True if self.aia_tar_path is not None else False
        self.isHMI = True if self.hmi_tar_path is not None else False
        self.isEVE = True if self.eve_tar_path is not None else False

        self.components = components
        self.wavelengths = wavelengths
        self.ions = ions

        self.train_months = [
            i
            for i in range(1, 13)
            if i not in self.test_months + self.val_months + self.holdout_months
        ]

        self.index_cache_filename = f"{cache_dir}/aligndata_{self.cache_id}.csv"
        self.normalizations_cache_filename = (
            f"{cache_dir}/normalizations_{self.cache_id}.json"
        )
        self.hmi_mask_cache_filename = f"{cache_dir}/hmi_mask_512x512.npy"

        self.aligndata = self.__aligntime()
        self.normalizations = self.__calc_normalizations()
        self.hmi_mask = self.__make_hmi_mask() if apply_mask else None

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output

    def setup(self, stage=None):
        self.train_ds = SDOMLWebDataset(
            self.aligndata,
            self.aia_tar_path,
            self.hmi_tar_path,
            self.eve_tar_path,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.train_months,
            normalizations=self.normalizations,
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
        )

        self.valid_ds = SDOMLWebDataset(
            self.aligndata,
            self.aia_tar_path,
            self.hmi_tar_path,
            self.eve_tar_path,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.val_months,
            normalizations=self.normalizations,
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
        )

        self.test_ds = SDOMLWebDataset(
            self.aligndata,
            self.aia_tar_path,
            self.hmi_tar_path,
            self.eve_tar_path,
            self.components,
            self.wavelengths,
            self.ions,
            self.cadence,
            self.test_months,
            normalizations=self.normalizations,
            mask=self.hmi_mask.numpy(),
            num_frames=self.num_frames,
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
