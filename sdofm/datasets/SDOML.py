# Adapted to be general from https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE/blob/main/src/irradiance/utilities/data_loader.py

import json
import os
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import zarr
from dask.array import stats
from dask.diagnostics import ProgressBar
from torch.utils.data import Dataset
from tqdm import tqdm

from ..constants import ALL_COMPONENTS, ALL_IONS, ALL_WAVELENGTHS


class SDOMLDataset(Dataset):
    def __init__(
        self,
        aligndata,
        hmi_data,
        aia_data,
        eve_data,
        components,
        wavelengths,
        ions,
        freq,
        months,
        normalizations=None,
        mask=None,
        num_frames=1,
        min_date=None,
        max_date=None,
    ):
        """
        aligndata --> aligned indexes for input-output matching
        aia_data --> zarr: aia data in zarr format
        eve_path --> zarr: eve data in zarr format
        hmi_path --> zarr: hmi data in zarr format
        components --> list: list of magnetic components for hmi (Bx, By, Bz)
        wavelengths   --> list: list of channels for aia (94, 131, 171, 193, 211, 304, 335, 1600, 1700)
        ions          --> list: list of ions for eve (MEGS A and MEGS B)
        freq          --> str: cadence used for rounding time series
        transformation: to be applied to aia in theory, but can stay None here
        use_normalizations: to use or not use normalizations, e.g. if this is test data, we don't want to use normalizations
        mask: to apply or not apply the HMI mask to AIA and HMI data
        """

        self.aligndata = aligndata

        self.aia_data = aia_data
        self.eve_data = eve_data
        self.hmi_data = hmi_data

        self.mask = mask

        # Select alls
        self.components = components
        self.wavelengths = wavelengths
        self.ions = ions
        if self.components is None:
            self.components = ALL_COMPONENTS
        if self.wavelengths is None:
            self.wavelengths = ALL_WAVELENGTHS
        if self.ions is None:
            self.ions = ALL_IONS

        # Loading data
        # HMI
        if self.hmi_data is not None:
            self.components.sort()
        # AIA
        if self.aia_data is not None:
            self.wavelengths.sort()
        # EVE
        self.ions.sort()
        self.cadence = freq
        self.months = months

        self.normalizations = normalizations

        # get data from path
        self.aligndata = self.aligndata.loc[
            self.aligndata.index.month.isin(self.months), :
        ]

        # if data filter, apply
        if min_date and max_date:
            if min_date < pd.to_datetime(
                "2010-09-09 00:00:11.08"
            ) or max_date > pd.to_datetime("2023-05-26 06:36:08.072"):
                raise ValueError("SDOML date range is not available. ")

            self.aligndata = self.aligndata[
                (self.aligndata.index >= min_date) & (self.aligndata.index <= max_date)
            ]

        # number of frames to return per sample
        self.num_frames = num_frames

    def __len__(self):
        # report slightly smaller such that all frame sets requested are available
        return self.aligndata.shape[0] - (self.num_frames - 1)

    def __getitem__(self, idx):

        image_stack = None
        if self.aia_data is not None:
            image_stack = self.get_aia_image(idx)

        if self.hmi_data is not None:
            image_stack = np.concatenate((image_stack, self.get_hmi_image(idx)), axis=0)

        if self.eve_data is not None:
            eve_data = self.get_eve(idx)
            return image_stack, eve_data
        else:
            return image_stack

        # if self.aia_data is not None and self.hmi_data is None:
        #     aia_image = self.get_aia_image(idx)
        #     eve_data = self.get_eve(idx)
        #     return aia_image, eve_data

        # elif self.hmi_data is not None and self.aia_data is None:
        #     hmi_image = self.get_hmi_image(idx)
        #     eve_data = self.get_eve(idx)
        #     return hmi_image, eve_data

        # else:
        #     aia_image = self.get_aia_image(idx)
        #     eve_data = self.get_eve(idx)
        #     hmi_image = self.get_hmi_image(idx)

        #     image_stack = np.concatenate((hmi_image, aia_image), axis=0)

        # return image_stack, eve_data

    def get_aia_image(self, idx):
        """Get AIA image for a given index.
        Returns a numpy array of shape (num_wavelengths, num_frames, height, width).
        """
        aia_image_dict = {}
        for wavelength in self.wavelengths:
            aia_image_dict[wavelength] = []
            for frame in range(self.num_frames):
                idx_row_element = self.aligndata.iloc[idx + frame]
                idx_wavelength = idx_row_element[f"idx_{wavelength}"]
                year = str(idx_row_element.name.year)
                img = self.aia_data[year][wavelength][idx_wavelength, :, :]

                if self.mask is not None:
                    img = img * self.mask

                aia_image_dict[wavelength].append(img)

                if self.normalizations:
                    aia_image_dict[wavelength][-1] -= self.normalizations["AIA"][
                        wavelength
                    ]["mean"]
                    aia_image_dict[wavelength][-1] /= self.normalizations["AIA"][
                        wavelength
                    ]["std"]

        aia_image = np.array(list(aia_image_dict.values()))

        return aia_image

    def get_eve(self, idx):
        """Get EVE data for a given index.
        Returns a numpy array of shape (num_ions, num_frames, ...).
        """
        eve_ion_dict = {}
        for ion in self.ions:
            eve_ion_dict[ion] = []
            for frame in range(self.num_frames):
                idx_eve = self.aligndata.iloc[idx + frame]["idx_eve"]
                eve_ion_dict[ion].append(self.eve_data[ion][idx_eve])
                if self.normalizations:
                    eve_ion_dict[ion][-1] -= self.normalizations["EVE"][ion]["mean"]
                    eve_ion_dict[ion][-1] /= self.normalizations["EVE"][ion]["std"]

        eve_data = np.array(list(eve_ion_dict.values()), dtype=np.float32)

        return eve_data

    def get_hmi_image(self, idx):
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

        return hmi_image

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output


class SDOMLDataModule(pl.LightningDataModule):
    """Loads paired data samples of AIA EUV images and EVE irradiance measures.

    Note: Input data needs to be paired.
    Parameters
    ----------
    hmi_path: path to hmi zarr file
    aia_path: path to aia zarr file
    eve_path: path to the EVE zarr data file
    components: list of magnetic field components
    batch_size: batch size (default is 32)
    num_workers: number of workers (needed for the training)
    val_months/test_months/holdout_monts: list of onths used to split the data
    cache_dir: path to directory for cashing data
    """

    def __init__(
        self,
        hmi_path,
        aia_path,
        eve_path,
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
        min_date=None,
        max_date=None,
    ):

        super().__init__()
        self.num_workers = (
            num_workers if num_workers is not None else os.cpu_count() // 2
        )
        self.hmi_path = hmi_path
        self.aia_path = aia_path
        self.eve_path = eve_path
        self.batch_size = batch_size
        self.cadence = frequency
        self.val_months = val_months
        self.test_months = test_months
        self.holdout_months = holdout_months
        self.cache_dir = cache_dir
        self.num_frames = num_frames
        self.min_date = pd.to_datetime(min_date) if min_date is not None else None
        self.max_date = pd.to_datetime(max_date) if max_date is not None else None
        self.isAIA = True if self.aia_path is not None else False
        self.isHMI = True if self.hmi_path is not None else False
        self.isEVE = True if self.eve_path is not None else False

        # Select alls
        self.components = components
        self.wavelengths = wavelengths
        self.ions = ions
        if self.components is None:
            self.components = ALL_COMPONENTS
        if self.wavelengths is None:
            self.wavelengths = ALL_WAVELENGTHS
        if self.ions is None:
            self.ions = ALL_IONS

        # checking if EVE is in the dataset
        if self.isEVE:
            self.eve_data = zarr.group(zarr.DirectoryStore(self.eve_path))
        else:
            self.eve_data = None

        # checking if AIA is in the dataset
        if self.isAIA:
            self.aia_data = zarr.group(zarr.DirectoryStore(self.aia_path))
        else:
            self.aia_data = None

        # checking if AIA is in the dataset
        if self.isHMI:
            self.hmi_data = zarr.group(zarr.DirectoryStore(self.hmi_path))
        else:
            self.hmi_data = None

        self.train_months = [
            i
            for i in range(1, 13)
            if i not in self.test_months + self.val_months + self.holdout_months
        ]

        if self.isAIA or self.isHMI:
            self.training_years = [int(year) for year in self.aia_data.keys()]
        else:  # EVE included, limit to 2010-2014
            self.training_years = [
                int(year) for year in self.hmi_data.keys() if int(year) < 2015
            ]

        # Cache filenames
        ids = []
        if self.isEVE:
            if len(self.ions) == 39:
                ions_id = "EVE_FULL"
            else:
                ions_id = "_".join(ions).replace(" ", "_")
            ids.append(ions_id)

        if self.isAIA:
            if len(self.wavelengths) == 9:
                wavelength_id = "AIA_FULL"
            elif len(self.wavelengths) > 0 and len(self.wavelengths) < 9:
                wavelength_id = "_".join(self.wavelengths)
            ids.append(wavelength_id)

        if self.isHMI:
            if len(self.components) == 3:
                component_id = "HMI_FULL"
            elif len(self.components) > 0 and len(self.components) < 3:
                component_id = "_".join(self.components)
            ids.append(component_id)

        self.cache_id = f"{'_'.join(ids)}_{self.cadence}"

        if self.aia_path is not None:
            if "small" in self.aia_path:
                self.cache_id += "_small"

        self.index_cache_filename = f"{cache_dir}/aligndata_{self.cache_id}.csv"
        self.normalizations_cache_filename = (
            f"{cache_dir}/normalizations_{self.cache_id}.json"
        )
        self.hmi_mask_cache_filename = f"{cache_dir}/hmi_mask_512x512.npy"

        self.aligndata = (
            self.__aligntime()
        )  # Temporal alignment of hmi, aia and eve data
        self.normalizations = self.__calc_normalizations()
        self.hmi_mask = self.__make_hmi_mask() if apply_mask else None

    def __str__(self):
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output

    def __aligntime(self):
        """
        This function extracts the common indexes across aia and eve datasets, considering potential missing values.
        """

        # Check the cache
        if Path(self.index_cache_filename).exists():
            print(
                f"[* CACHE SYSTEM *] Found cached index data in {self.index_cache_filename}."
            )
            aligndata = pd.read_csv(self.index_cache_filename)
            aligndata["Time"] = pd.to_datetime(aligndata["Time"])
            aligndata.set_index("Time", inplace=True)
            return aligndata
        print(f"\nData alignment calculation begin:")
        print("-" * 50)

        join_series = None

        # AIA
        if self.isAIA:
            print(f"Aligning AIA data")

            for i, wavelength in enumerate(self.wavelengths):
                print(f"Aligning AIA data for wavelength: {wavelength}")
                for j, year in enumerate(tqdm((self.training_years))):
                    aia_channel = self.aia_data[year][wavelength]

                    # get observation time
                    t_obs_aia_channel = aia_channel.attrs["T_OBS"]
                    if j == 0:
                        # transform to DataFrame
                        # AIA
                        df_t_aia = pd.DataFrame(
                            {
                                "Time": pd.to_datetime(
                                    t_obs_aia_channel, format="mixed"
                                ),
                                f"idx_{wavelength}": np.arange(
                                    0, len(t_obs_aia_channel)
                                ),
                            }
                        )

                    else:
                        df_tmp_aia = pd.DataFrame(
                            {
                                "Time": pd.to_datetime(
                                    t_obs_aia_channel, format="mixed"
                                ),
                                f"idx_{wavelength}": np.arange(
                                    0, len(t_obs_aia_channel)
                                ),
                            }
                        )
                        df_t_aia = pd.concat([df_t_aia, df_tmp_aia], ignore_index=True)

                # Enforcing same datetime format
                transform_datetime = lambda x: pd.to_datetime(
                    x, format="mixed"
                ).strftime("%Y-%m-%d %H:%M:%S")
                df_t_aia["Time"] = df_t_aia["Time"].apply(transform_datetime)
                df_t_aia["Time"] = pd.to_datetime(df_t_aia["Time"]).dt.tz_localize(
                    None
                )  # this is needed for timezone-naive type
                df_t_aia["Time"] = df_t_aia["Time"].dt.round(self.cadence)
                df_t_obs_aia = df_t_aia.drop_duplicates(
                    subset="Time", keep="first"
                )  # removing potential duplicates derived by rounding
                df_t_obs_aia.set_index("Time", inplace=True)

                # if i == 0:
                if join_series is None:
                    join_series = df_t_obs_aia
                else:
                    join_series = join_series.join(df_t_obs_aia, how="inner")

            print(f"AIA alignment completed with {join_series.shape[0]} samples.")

        # ----------------------------------------------------------------------------------------------------------------------------------

        # HMI
        if self.isHMI:
            print(f"Aligning HMI data")
            for i, component in enumerate(self.components):
                print(f"Aligning HMI data for component: {component}")
                for j, year in enumerate(
                    tqdm((self.training_years))
                ):  # EVE data only goes up to 2014

                    hmi_channel = self.hmi_data[year][component]

                    # get observation time
                    t_obs_hmi_channel_pre = hmi_channel.attrs["T_OBS"]

                    for idx, time_val in enumerate(t_obs_hmi_channel_pre):
                        t_obs_hmi_channel_pre[idx] = time_val[:19]

                    # substitute characters
                    replacements = {".": "-", "_": "T", "TTAI": "", "60": "59"}
                    t_obs_hmi_channel = []
                    for word in t_obs_hmi_channel_pre:
                        for old_char, new_char in replacements.items():
                            word = word.replace(old_char, new_char)
                        t_obs_hmi_channel.append(word)

                    if j == 0:
                        # transform to DataFrame
                        # HMI
                        df_t_hmi = pd.DataFrame(
                            {
                                "Time": pd.to_datetime(
                                    t_obs_hmi_channel, format="mixed"
                                ),
                                f"idx_{component}": np.arange(
                                    0, len(t_obs_hmi_channel)
                                ),
                            }
                        )

                    else:
                        df_tmp_hmi = pd.DataFrame(
                            {
                                "Time": pd.to_datetime(
                                    t_obs_hmi_channel, format="mixed"
                                ),
                                f"idx_{component}": np.arange(
                                    0, len(t_obs_hmi_channel)
                                ),
                            }
                        )
                        df_t_hmi = pd.concat([df_t_hmi, df_tmp_hmi], ignore_index=True)

                # Enforcing same datetime format
                transform_datetime = lambda x: pd.to_datetime(
                    x, format="mixed"
                ).strftime("%Y-%m-%d %H:%M:%S")
                df_t_hmi["Time"] = df_t_hmi["Time"].apply(transform_datetime)
                df_t_hmi["Time"] = pd.to_datetime(df_t_hmi["Time"]).dt.tz_localize(
                    None
                )  # this is needed for timezone-naive type
                df_t_hmi["Time"] = df_t_hmi["Time"].dt.round(self.cadence)
                df_t_obs_hmi = df_t_hmi.drop_duplicates(
                    subset="Time", keep="first"
                )  # removing potential duplicates derived by rounding
                df_t_obs_hmi.set_index("Time", inplace=True)

                if join_series is None:
                    join_series = df_t_obs_hmi
                else:
                    join_series = join_series.join(df_t_obs_hmi, how="inner")

        print(f"HMI alignment completed with {join_series.shape[0]} samples.")

        # ---------------------------------------------------------------------------------------------------------------------------------------

        # EVE
        if self.isEVE:
            print(f"Aligning EVE data")
            df_t_eve = pd.DataFrame(
                {
                    "Time": pd.to_datetime(self.eve_data["Time"][:]),
                    "idx_eve": np.arange(0, len(self.eve_data["Time"])),
                }
            )
            df_t_eve["Time"] = pd.to_datetime(df_t_eve["Time"]).dt.round(self.cadence)
            df_t_obs_eve = df_t_eve.drop_duplicates(
                subset="Time", keep="first"
            ).set_index("Time")

            if join_series is None:
                join_series = df_t_obs_eve
            else:
                join_series = join_series.join(df_t_obs_eve, how="inner")

            # remove missing eve data (missing values are labeled with negative values)
            for ion in self.ions:
                ion_data = self.eve_data[ion][:]
                join_series = join_series.loc[ion_data[join_series["idx_eve"]] > 0, :]

        if join_series is None:
            raise ValueError("No data found for alignment.")

        join_series.sort_index(inplace=True)

        print("")
        print("#" * 50)
        print(f"[*] Total Alignment Completed with {join_series.shape[0]} Samples.")
        print(f"[*] Saving alignment data to {self.index_cache_filename}.")
        print("#" * 50)
        print("")
        # creating csv dataset
        join_series.to_csv(self.index_cache_filename)

        return join_series

    def __calc_normalizations(self):

        if Path(self.normalizations_cache_filename).exists():
            print(
                f"[* CACHE SYSTEM *] Found cached normalization data in {self.normalizations_cache_filename}."
            )
            with open(self.normalizations_cache_filename, "r") as json_file:
                return json.load(json_file)

        normalizations = {}
        normalizations_align = self.aligndata.copy()
        normalizations_align = normalizations_align[
            normalizations_align.index.month.isin(self.train_months)
        ]

        if self.isEVE:
            normalizations["EVE"] = self.__calc_eve_normalizations(normalizations_align)

        if self.isAIA:
            normalizations["AIA"] = self.__calc_aia_normalizations(normalizations_align)

        if self.isHMI:
            normalizations["HMI"] = self.__calc_hmi_normalizations(normalizations_align)

        with open(self.normalizations_cache_filename, "w") as json_file:
            save_json = str(normalizations)
            save_json = save_json.replace("'", '"')
            json_file.write(save_json)

        return normalizations

    def __calc_eve_normalizations(self, normalizations_align) -> dict:

        # EVE Normalization
        normalizations_eve = {}
        for ion in self.ions:
            # Note that selecting on idx normalizations_align['idx_eve'] removes negative values from EVE data.
            channel_data = self.eve_data[ion][normalizations_align["idx_eve"]][:]
            normalizations_eve[ion] = {}
            normalizations_eve[ion]["count"] = channel_data.shape[0]
            normalizations_eve[ion]["sum"] = channel_data.sum()
            normalizations_eve[ion]["mean"] = channel_data.mean()
            normalizations_eve[ion]["std"] = channel_data.std()
            normalizations_eve[ion]["min"] = channel_data.min()
            normalizations_eve[ion]["max"] = channel_data.max()

        normalizations_eve["eve_norm"] = [
            [normalizations_eve[key]["mean"] for key in normalizations_eve.keys()],
            [normalizations_eve[key]["std"] for key in normalizations_eve.keys()],
        ]
        return normalizations_eve

    def __calc_aia_normalizations(self, normalizations_align) -> dict:
        normalizations_aia = {}

        for wavelength in self.wavelengths:
            wavelength_data = da.from_array(self.aia_data[2010][wavelength])

            for year in self.training_years:  # EVE data only goes up to 2014.
                wavelength_data_year = da.from_array(self.aia_data[year][wavelength])
                wavelength_data = da.concatenate(
                    [wavelength_data, wavelength_data_year], axis=0
                )

            wavelength_data = wavelength_data[normalizations_align[f"idx_{wavelength}"]]

            print(f"\nCalculating normalizations for wavelength {wavelength}:")
            print("-" * 50)

            normalizations_aia[wavelength] = {}

            print(f"Sum:")
            with ProgressBar():
                normalizations_aia[wavelength]["sum"] = wavelength_data.sum().compute()

            print(f"Max Pixel Value:")
            with ProgressBar():
                normalizations_aia[wavelength]["max"] = wavelength_data.max().compute()

            print(f"Standard Deviation:")
            with ProgressBar():
                normalizations_aia[wavelength]["std"] = wavelength_data.std().compute()

            print(f"Skew:")
            with ProgressBar():
                normalizations_aia[wavelength]["skew"] = stats.skew(
                    wavelength_data.flatten()
                ).compute()

            print(f"Kurtosis:")
            with ProgressBar():
                normalizations_aia[wavelength]["kurtosis"] = stats.kurtosis(
                    wavelength_data.flatten()
                ).compute()

            normalizations_aia[wavelength]["image_count"] = wavelength_data.shape[0]
            normalizations_aia[wavelength]["pixel_count"] = (
                wavelength_data.shape[0]
                * wavelength_data.shape[1]
                * wavelength_data.shape[2]
            )
            normalizations_aia[wavelength]["mean"] = (
                normalizations_aia[wavelength]["sum"]
                / normalizations_aia[wavelength]["pixel_count"]
            )

        return normalizations_aia

    def __calc_hmi_normalizations(self, normalizations_align) -> dict:
        normalizations_hmi = {}

        for component in self.components:
            component_data = da.from_array(self.hmi_data[2010][component])

            for year in self.training_years:  # EVE data only goes up to 2014.
                component_data_year = da.from_array(self.hmi_data[year][component])
                component_data = da.concatenate(
                    [component_data, component_data_year], axis=0
                )

            component_data = component_data[
                normalizations_align[f"idx_{self.components[0]}"]
            ]

            print(f"\nCalculating normalizations for component {component}:")
            print("-" * 50)

            normalizations_hmi[component] = {}

            print(f"Sum:")
            with ProgressBar():
                normalizations_hmi[component]["sum"] = component_data.sum().compute()

            print(f"Max Pixel Value:")
            with ProgressBar():
                normalizations_hmi[component]["max"] = component_data.max().compute()

            print(f"Standard Deviation:")
            with ProgressBar():
                normalizations_hmi[component]["std"] = component_data.std().compute()

            print(f"Skew:")
            with ProgressBar():
                normalizations_hmi[component]["skew"] = stats.skew(
                    component_data.flatten()
                ).compute()

            print(f"Kurtosis:")
            with ProgressBar():
                normalizations_hmi[component]["kurtosis"] = stats.kurtosis(
                    component_data.flatten()
                ).compute()

            normalizations_hmi[component]["image_count"] = component_data.shape[0]
            normalizations_hmi[component]["pixel_count"] = (
                component_data.shape[0]
                * component_data.shape[1]
                * component_data.shape[2]
            )
            normalizations_hmi[component]["mean"] = (
                normalizations_hmi[component]["sum"]
                / normalizations_hmi[component]["pixel_count"]
            )

        return normalizations_hmi

    def __make_hmi_mask(self):
        if Path(self.hmi_mask_cache_filename).exists():
            loaded_mask = np.load(self.hmi_mask_cache_filename)
            hmi_mask = torch.Tensor(loaded_mask).to(dtype=torch.uint8)
            print(
                f"[* CACHE SYSTEM *] Found cached HMI mask data in {self.hmi_mask_cache_filename}."
            )
            return hmi_mask
        elif not self.isHMI:
            raise ValueError(
                "Mask could not be found in cache and 2010 HMI data is not available to generate it, stopping..."
            )

        hmi = torch.Tensor(self.hmi_data[2010][ALL_COMPONENTS[0]][0])
        hmi_mask = (torch.abs(hmi) > 0.0).to(dtype=torch.uint8)
        hmi_mask_ratio = hmi_mask.sum().item() / hmi_mask.numel()
        if np.abs(hmi_mask_ratio - 0.496) > 0.2:
            print(
                f"WARNING: HMI mask ratio is {hmi_mask_ratio:.2f}, which is significantly different from expected (0.496)"
            )
        print(
            f"[*] Saving HMI mask with ratio {hmi_mask_ratio:.2f} to {self.hmi_mask_cache_filename}."
        )
        np.save(self.hmi_mask_cache_filename, hmi_mask.numpy())
        return hmi_mask

    def setup(self, stage=None):

        self.train_ds = SDOMLDataset(
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
        )

        self.valid_ds = SDOMLDataset(
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
        )

        self.test_ds = SDOMLDataset(
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
