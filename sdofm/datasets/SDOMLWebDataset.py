import json
import os
from pathlib import Path
from multiprocessing import Pool
import webdataset.shardlists as wds_shardlists
import glob
import re

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from torch.utils.data import Dataset
from tqdm import tqdm

def get_info_from_shard(shard_url):
    """
    Extracts metadata from a webdataset shard.

    Args:
        shard_url (str): The URL of the shard.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              key, timestamp, and item for a sample in the shard.
    """
    info = []
    
    # Infer item from shard URL based on the directory structure
    # e.g., /mnt/sdoml/webdataset/aia/2018/171A/shard-000000.tar.gz
    match = re.search(r"/(aia|hmi)/[0-9]{4}/([a-zA-Z0-9_]+)/", shard_url)
    if not match:
        return info
        
    item_type = match.group(1)
    item_from_path = match.group(2)
    
    # For AIA, we want the number part (e.g., '171A' -> '171')
    if item_type == 'aia':
        item = re.sub(r"[^0-9]", "", item_from_path)
    else:
        item = item_from_path

    try:
        # Use warn_and_continue to handle errors within shards gracefully
        ds = wds.WebDataset(shard_url, shardshuffle=False, handler=wds.warn_and_continue).decode()
        for sample in ds:
            # Ensure that the sample has a json file and that the T_OBS key exists
            if "json" in sample and "T_OBS" in sample["json"]:
                key = sample["__key__"]
                timestamp = sample["json"]["T_OBS"]
                info.append({"key": key, "Time": timestamp, "item": item})
    except Exception as e:
        # Log errors instead of silently passing
        print(f"Error processing shard {shard_url}: {e}")
        
    return info
def _calculate_stats_for_shard(args):
    shard_url, train_months, data_type, keys_of_interest = args
    
    stats = {key: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'min': np.inf, 'max': -np.inf} for key in keys_of_interest}

    try:
        ds = wds.WebDataset(shard_url, shardshuffle=False).decode()
        for sample in ds:
            key = sample["__key__"]
            
            timestamp_str = key.split('_')[1]
            month = int(timestamp_str[4:6])

            if month not in train_months:
                continue

            if data_type == "AIA":
                item_key = key.split('_')[2] # wavelength
            elif data_type == "HMI":
                item_key = key.split('_')[2] # component
            
            if item_key in keys_of_interest:
                img_key = [k for k in sample if k.endswith('npy')][0]
                img = sample[img_key]
                stats[item_key]['sum'] += np.sum(img)
                stats[item_key]['sum_sq'] += np.sum(np.square(img))
                stats[item_key]['count'] += img.size
                stats[item_key]['min'] = min(stats[item_key]['min'], np.min(img))
                stats[item_key]['max'] = max(stats[item_key]['max'], np.max(img))

    except Exception as e:
        print(f"Error processing shard {shard_url} for stats: {e}")
        
    return stats
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
        self.aia_dataset = (
            wds.WebDataset(aia_tar_path, shardshuffle=False) if aia_tar_path else None
        )
        self.hmi_dataset = (
            wds.WebDataset(hmi_tar_path, shardshuffle=False) if hmi_tar_path else None
        )
        self.eve_dataset = (
            wds.WebDataset(eve_tar_path, shardshuffle=False) if eve_tar_path else None
        )

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
                with self.aia_dataset.decode() as ds:
                    sample = ds[f"{idx_wavelength}"]
                    img = sample["npy"]

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
                with self.hmi_dataset.decode() as ds:
                    sample = ds[f"{idx_component}"]
                    img = sample["npy"]

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

        self.components = components if components is not None else ALL_COMPONENTS
        self.wavelengths = wavelengths if wavelengths is not None else ALL_WAVELENGTHS
        self.ions = ions if ions is not None else ALL_IONS

        self.train_months = [
            i
            for i in range(1, 13)
            if i not in self.test_months + self.val_months + self.holdout_months
        ]

        ids = []
        if self.isHMI:
            if len(self.components) == 3:
                component_id = "HMI_FULL"
            elif len(self.components) > 0 and len(self.components) < 3:
                component_id = "_".join(self.components)
            ids.append(component_id)

        if self.isAIA:
            if len(self.wavelengths) == 9:
                wavelength_id = "AIA_FULL"
            elif len(self.wavelengths) > 0 and len(self.wavelengths) < 9:
                wavelength_id = "_".join(self.wavelengths)
            ids.append(wavelength_id)

        if self.isEVE:
            if len(self.ions) == 38:  # excluding Fe XVI_2
                ions_id = "EVE_FULL"
            else:
                ions_id = "_".join(self.ions).replace(" ", "_")
            ids.append(ions_id)

        self.cache_id = f"{'_'.join(sorted(ids))}_{self.cadence}_{str(self.min_date).replace(' ', '')}-{str(self.max_date).replace(' ', '')}"

        if self.aia_tar_path is not None:
            if "small" in self.aia_tar_path:
                self.cache_id += "_small"

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

    def __aligntime(self):
        """
        This function extracts the common indexes across aia and eve datasets, considering potential missing values.
        It now runs multi-threaded to speed up index creation.
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
        print(f"No alignment cache found at {self.index_cache_filename}")
        print(f"\nData alignment calculation begin:")
        print("-" * 50)

        all_data = []
        
        def process_shards(tar_path, items, data_type):
            all_shards = []
            # Expand the tar_path for each item (wavelength or component)
            # Example: /path/to/{item}_webdataset -> /path/to/aia_94_webdataset
            for item in items:
                
                # Format the tar path with the specific item
                formatted_path = tar_path.replace("{item}", str(item))
                
                # Replace shard number pattern with a wildcard
                glob_path = re.sub(r'\{.*?\}', '*', formatted_path)
                
                found_shards = glob.glob(glob_path)
                if not found_shards:
                    print(f"Warning: No shards found for {data_type} item {item} with path {glob_path}")
                all_shards.extend(found_shards)

            print(f"Found {len(all_shards)} shards to process for {data_type}.")
            
            if not all_shards:
                return

            with Pool(self.num_workers) as pool:
                with tqdm(total=len(all_shards), desc=f"Building index from {data_type} shards") as pbar:
                    for info in pool.imap_unordered(get_info_from_shard, all_shards):
                        if info:
                            all_data.extend(info)
                        pbar.update()

        if self.isAIA:
            process_shards(self.aia_tar_path, self.wavelengths, "AIA")
        if self.isHMI:
            process_shards(self.hmi_tar_path, self.components, "HMI")

        df = pd.DataFrame(all_data)
        df['Time'] = pd.to_datetime(df['Time'])
        df['Time'] = df['Time'].dt.round(self.cadence)

        # Pivot the table to get idx_{wavelength} columns
        join_series = df.pivot_table(index='Time', columns='item', values='key').add_prefix('idx_')
        
        # Inner join to keep only timestamps present in all items
        join_series.dropna(inplace=True)
        
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
            print(f"[* CACHE SYSTEM *] Found cached normalization data in {self.normalizations_cache_filename}.")
            with open(self.normalizations_cache_filename, "r") as json_file:
                return json.load(json_file)

        print(f"No normalization cache found at {self.normalizations_cache_filename}. Calculating...")
        normalizations = {}

        # Helper to process a list of shards for a given data type (AIA or HMI)
        def calculate_stats_for_type(data_type, tar_path, keys_of_interest):
            shards = glob.glob(re.sub(r'\{.*?\}', '*', tar_path))
            
            pool_args = [(shard, self.train_months, data_type, keys_of_interest) for shard in shards]
            
            aggregated_stats = {key: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'min': np.inf, 'max': -np.inf} for key in keys_of_interest}

            with Pool(self.num_workers) as pool:
                with tqdm(total=len(shards), desc=f"Calculating {data_type} normalization") as pbar:
                    for shard_stats in pool.imap_unordered(_calculate_stats_for_shard, pool_args):
                        for key in keys_of_interest:
                            aggregated_stats[key]['sum'] += shard_stats[key]['sum']
                            aggregated_stats[key]['sum_sq'] += shard_stats[key]['sum_sq']
                            aggregated_stats[key]['count'] += shard_stats[key]['count']
                            aggregated_stats[key]['min'] = min(aggregated_stats[key]['min'], shard_stats[key]['min'])
                            aggregated_stats[key]['max'] = max(aggregated_stats[key]['max'], shard_stats[key]['max'])
                        pbar.update()
            
            type_normalizations = {}
            for key in keys_of_interest:
                count = aggregated_stats[key]['count']
                mean = aggregated_stats[key]['sum'] / count if count > 0 else 0
                # Var(X) = E[X^2] - (E[X])^2
                var = (aggregated_stats[key]['sum_sq'] / count) - (mean ** 2) if count > 0 else 0
                std = np.sqrt(var) if var > 0 else 0
                
                type_normalizations[key] = {
                    'mean': mean,
                    'std': std,
                    'min': aggregated_stats[key]['min'],
                    'max': aggregated_stats[key]['max'],
                    'count': count
                }
            return type_normalizations

        if self.isAIA:
            normalizations["AIA"] = calculate_stats_for_type("AIA", self.aia_tar_path, self.wavelengths)
        
        if self.isHMI:
            normalizations["HMI"] = calculate_stats_for_type("HMI", self.hmi_tar_path, self.components)

        with open(self.normalizations_cache_filename, "w") as json_file:
            json.dump(normalizations, json_file, indent=4)
        
        return normalizations

    def __make_hmi_mask(self):
        if Path(self.hmi_mask_cache_filename).exists():
            loaded_mask = np.load(self.hmi_mask_cache_filename)
            hmi_mask = torch.Tensor(loaded_mask).to(dtype=torch.uint8)
            print(f"[* CACHE SYSTEM *] Found cached HMI mask data in {self.hmi_mask_cache_filename}.")
            return hmi_mask
        
        if not self.isHMI:
            print("Warning: HMI data not available, cannot generate mask. Proceeding without mask.")
            return None

        print(f"No HMI mask cache found at {self.hmi_mask_cache_filename}. Generating...")
        
        shards = glob.glob(re.sub(r'\{.*?\}', '*', self.hmi_tar_path))
        if not shards:
            raise ValueError("No HMI shards found to generate mask.")

        # Use the first shard to generate the mask
        first_shard_url = shards[0]
        try:
            ds = wds.WebDataset(first_shard_url, shardshuffle=False).decode()
            first_sample = next(iter(ds))
            img_key = [k for k in first_sample if k.endswith('npy')][0]
            hmi_image = first_sample[img_key]
            
            hmi_mask = (torch.abs(torch.from_numpy(hmi_image.view(hmi_image.dtype.newbyteorder()))) > 0.0).to(dtype=torch.uint8)
            hmi_mask_ratio = hmi_mask.sum().item() / hmi_mask.numel()
            
            if np.abs(hmi_mask_ratio - 0.496) > 0.2:
                print(f"WARNING: HMI mask ratio is {hmi_mask_ratio:.2f}, which is significantly different from expected (0.496)")
            
            print(f"[*] Saving HMI mask with ratio {hmi_mask_ratio:.2f} to {self.hmi_mask_cache_filename}.")
            np.save(self.hmi_mask_cache_filename, hmi_mask.numpy())
            return hmi_mask

        except Exception as e:
            print(f"Error generating HMI mask from {first_shard_url}: {e}")
            raise e

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