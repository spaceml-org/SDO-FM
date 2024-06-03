from .SDOML import SDOMLDataModule, SDOMLDataset
from ..io import io
import pandas as pd
import numpy as np


class RandomIntervalSDOMLDataModule(SDOMLDataModule):

    def __init__(self, 
                 sampling_period="10days", 
                 num_frames=1, 
                 dim=False, 
                 blosc_cache=None, 
                 start_date=None,
                 end_date=None,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)        
        self.sampling_period = sampling_period
        self.num_frames = num_frames
        self.dim = dim
        self.blosc_cache = blosc_cache
        self.start_date = start_date
        self.end_date = end_date

        if start_date is not None:
            self.aligndata = self.aligndata[self.start_date:]

        if end_date is not None:
            self.aligndata = self.aligndata[:self.end_date]

        # check sampling_period is long enough for num_frames
        if pd.Timedelta(self.sampling_period) - pd.Timedelta(self.cadence)*num_frames < pd.Timedelta(self.cadence):
            raise ValueError(f"too many 'num_frames' ({num_frames}) for sampling_period={sampling_period} and cadence={self.cadence}")         
                
    def setup(self, stage=None):

        self.train_ds = RandomIntervalSDOMLDataset(
            sampling_period = self.sampling_period,
            num_frames = self.num_frames,
            dim = self.dim,
            blosc_cache = self.blosc_cache, 
            aligndata = self.aligndata,
            hmi_data = self.hmi_data,
            aia_data = self.aia_data,
            eve_data = self.eve_data,
            components = self.components,
            wavelengths = self.wavelengths,
            ions = self.ions,
            freq = self.cadence,
            months = self.train_months,
            normalizations=self.normalizations,
            mask=self.hmi_mask.numpy(),
        )

        self.valid_ds = RandomIntervalSDOMLDataset(
            sampling_period = self.sampling_period,
            num_frames = self.num_frames,
            dim = self.dim,
            blosc_cache = self.blosc_cache, 
            aligndata = self.aligndata,
            hmi_data = self.hmi_data,
            aia_data = self.aia_data,
            eve_data = self.eve_data,
            components = self.components,
            wavelengths = self.wavelengths,
            ions = self.ions,
            freq = self.cadence,
            months = self.val_months,
            normalizations=self.normalizations,
            mask=self.hmi_mask.numpy(),
        )

        self.test_ds = RandomIntervalSDOMLDataset(
            sampling_period = self.sampling_period,
            num_frames = self.num_frames,
            dim = self.dim,
            blosc_cache = self.blosc_cache, 
            aligndata = self.aligndata,
            hmi_data = self.hmi_data,
            aia_data = self.aia_data,
            eve_data = self.eve_data,
            components = self.components,
            wavelengths = self.wavelengths,
            ions = self.ions,
            freq = self.cadence,
            months = self.test_months,
            normalizations=self.normalizations,
            mask=self.hmi_mask.numpy(),
        )
        
        
class RandomIntervalSDOMLDataset(SDOMLDataset):
    
    def __init__(self, sampling_period="10days", num_frames=1, blosc_cache=None, dim=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_period = sampling_period
        self.num_frames = num_frames
        self.dim = dim
        self.aligndata_full = self.aligndata     
        self.blosc_cache = blosc_cache   
        
        # indexes for this aligndata start sampling_period after the first one
        first_timestamp = self.aligndata.index[0] + pd.Timedelta(self.sampling_period) + pd.Timedelta("1min")
        self.aligndata = self.aligndata.loc[first_timestamp:]
        
        self.aia_means  = np.r_[[self.normalizations["AIA"][wavelength]["mean"] for wavelength in self.wavelengths]]
        self.aia_stdevs = np.r_[[self.normalizations["AIA"][wavelength]["std"] for wavelength in self.wavelengths]]

        self.hmi_means  = np.r_[[self.normalizations["HMI"][component]["mean"] for component in self.components]]
        self.hmi_stdevs = np.r_[[self.normalizations["HMI"][component]["std"] for component in self.components]]

        
    def get_aia_image(self, idx, aligned_samples):
        """Get AIA image for a given index.
        Returns a numpy array of shape (num_wavelengths, num_frames, height, width).
        """
        aia_image = np.r_[[io.load_aia(self.blosc_cache, self.aia_data, idx_row_element, self.wavelengths) \
                           for _,idx_row_element in aligned_samples.iterrows()]]
                    
        if self.mask is not None:
            aia_image *= self.mask[np.newaxis, np.newaxis, :, :]        

        if self.normalizations:
            aia_image -=  self.aia_means[np.newaxis, :, np.newaxis, np.newaxis]
            aia_image /= self.aia_stdevs[np.newaxis, :, np.newaxis, np.newaxis]    
                        
        aia_image = np.transpose(aia_image, [1,0,2,3])
        return aia_image


    def get_eve(self, idx, aligned_samples):
        """Get EVE data for a given index.
        Returns a numpy array of shape (num_ions, num_frames, ...).
        """
        eve_ion_dict = {}
        for ion in self.ions:
            eve_ion_dict[ion] = []                
            for _, idx_row_element in aligned_samples.iterrows():
                idx_eve = idx_row_element["idx_eve"]
                eve_ion_dict[ion].append(self.eve_data[ion][idx_eve])
                if self.normalizations:
                    eve_ion_dict[ion][-1] -= self.normalizations["EVE"][ion]["mean"]
                    eve_ion_dict[ion][-1] /= self.normalizations["EVE"][ion]["std"]

        eve_data = np.array(list(eve_ion_dict.values()), dtype=np.float32)

        return eve_data
    

    def get_hmi_image(self, idx, aligned_samples):
        """Get HMI image for a given index.
        Returns a numpy array of shape (num_channels, num_frames, height, width).
        """
        hmi_image = np.r_[[io.load_hmi(self.blosc_cache, self.hmi_data, idx_row_element, self.components) \
                           for _,idx_row_element in aligned_samples.iterrows()]]
                    
        if self.mask is not None:
            hmi_image *= self.mask[np.newaxis, np.newaxis, :, :]        

        if self.normalizations:
            hmi_image -=  self.hmi_means[np.newaxis, :, np.newaxis, np.newaxis]
            hmi_image /= self.hmi_stdevs[np.newaxis, :, np.newaxis, np.newaxis]    
                        
        hmi_image = np.transpose(hmi_image, [1,0,2,3])
        return hmi_image

    
    def __getitem__(self, idx):
        
        # sample num_frames between idx and idx - sampling_period
        item = self.aligndata.iloc[idx]
        sampling_df = self.aligndata_full.loc[item.name - pd.Timedelta(self.sampling_period): item.name - pd.Timedelta("1sec")]
        aligned_samples = pd.concat([sampling_df.sample(self.num_frames-1), pd.DataFrame([item])])        
        timestamps = [i.strftime("%Y-%m-%d %H:%M:%S") for i in aligned_samples.index]
        
        image_stack = None
        if self.aia_data is not None:
            aia_img = self.get_aia_image(idx, aligned_samples)
            image_stack = aia_img

        if self.hmi_data is not None:
            hmi_img = self.get_hmi_image(idx, aligned_samples)
            image_stack = np.concatenate((image_stack, hmi_img), axis=0)

        if self.eve_data is not None:
            eve_data = self.get_eve(idx, aligned_samples)

        r = {'timestamps': timestamps,
             'image_stack': image_stack}

        if self.eve_data:
            r['eve_data'] = eve_data

        if self.dim:
            num_channels = image_stack.shape[0]
            dim_factor = np.random.random(num_channels).reshape([-1,1,1,1])
            r['dimmed_image_stack'] = image_stack * dim_factor
            r['dim_factor'] = dim_factor.reshape(-1)

        return r
            
