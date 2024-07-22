import numpy as np
import pandas as pd

from ..io import io
from .SDOML import SDOMLDataModule, SDOMLDataset


class NonLinearSDOMLDataModule(SDOMLDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):

        self.train_ds = NonLinearSDOMLDataset(
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
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
        )

        self.valid_ds = NonLinearSDOMLDataset(
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
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
        )

        self.test_ds = NonLinearSDOMLDataset(
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
            drop_frame_dim=self.drop_frame_dim,
            min_date=self.min_date,
            max_date=self.max_date,
        )


class NonLinearSDOMLDataset(SDOMLDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        # # sample num_frames between idx and idx - sampling_period
        # items = self.aligndata.iloc[idx]
        # # print(items.index, idx, pd.DataFrame([items]))
        # timestamps = [
        #     i.strftime("%Y-%m-%d %H:%M:%S") for i in pd.DataFrame([items]).index
        # ]

        # r = {"timestamps": timestamps}

        # if self.eve_data:
        #     image_stack, eve_data = super().__getitem__(idx)
        #     r["eve_data"] = eve_data
        # else:
        #     image_stack = super().__getitem__(idx)

        # r["image_stack"] = image_stack
        image_stack = np.log(super().__getitem__(idx))

        return image_stack
