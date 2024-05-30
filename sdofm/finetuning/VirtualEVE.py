# Adapted from https://github.com/FrontierDevelopmentLab/2023-FDL-X-ARD-EVE

import sys
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchvision
from torch.nn import HuberLoss

from ..BaseModule import BaseModule


class VirtualEVE(BaseModule):

    def __init__(
        self,
        # Backbone parameters
        d_input,
        d_output,
        eve_norm,
        model="efficientnet_b0",
        dp=0.75,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eve_norm = eve_norm
