from torch import nn
import torch
import numpy as np
from . import unet

class DimmingEstimator(nn.Module):
    def __init__(self, pretrained_unet = None, 
                       output_size = 12, 
                       fc_layers_size = [128,64,16]):
        super().__init__()

        self.fc_layers_size = fc_layers_size
        self.input_channels = 12
        self.output_size = output_size
        
        if pretrained_unet is None:
            self.unet = unet.UNet(n_channels=12, n_classes=1, bilinear=True, use_embeddings_block=False, size_factor=4)
        else:
            self.unet = pretrained_unet 
            
        unet_out_channels = self.unet.encoder_layers[-1].maxpool_conv[-1].double_conv[-3].out_channels
        
        # reduce number of channels
        self.conv1x1 = nn.Conv2d(unet_out_channels, 32, 1, stride=1)
            
        self.flatten = nn.Flatten()

        # fully connected layers to prediction
        self.fc_layers = [ nn.Linear(32768
        , fc_layers_size[0]),
                           nn.ReLU()]
    
        prev_size = fc_layers_size[0]
        for si in range(len(fc_layers_size)):
            self.fc_layers.append(nn.Linear(prev_size, fc_layers_size[si]))
            self.fc_layers.append(nn.ReLU())
            prev_size = fc_layers_size[si]
        
        self.fc_layers.append(nn.Linear(prev_size, self.output_size))
        self.fc_layers.append(nn.Sigmoid())                                  
                        
        self.fc_layers = nn.Sequential(*self.fc_layers)
        
    def forward(self, x):
        
        # pass through the unet encoder
        x = self.unet.forward_encode(x)
        x = self.conv1x1(x)
        x = self.flatten(x)

        # concat send to fc layers
        x = self.fc_layers(x)
        return x