

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

actfn_map = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid
}
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, output_act_fn='relu'):

        if not output_act_fn in actfn_map.keys():
            raise ValueError(f"invalid activation fn {output_act_fn}, only {list(actfn_map.keys())} allowed")

        super().__init__()
        self.output_act_fn = output_act_fn
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            actfn_map[output_act_fn]()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, output_act_fn='relu'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, output_act_fn=output_act_fn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with NO SKIPPED connections"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels // 2, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_embeddings_block=False, size_factor = 1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_embeddings_block = use_embeddings_block

        self.channels4 = 1024
        self.size_factor = size_factor
        bifactor = 2 if bilinear else 1

        # ----- encoder ------
        self.inc = (DoubleConv(n_channels, 64 // self.size_factor))
        self.down1 = (Down(64 // self.size_factor, 128 // self.size_factor))
        self.down2 = (Down(128 // self.size_factor, 256 // self.size_factor))
        self.down3 = (Down(256 // self.size_factor, 512 // self.size_factor))
        self.down4 = (Down(512 // self.size_factor, self.channels4 // bifactor // self.size_factor))
        self.encoder_layers = nn.ModuleList([self.inc, self.down1, self.down2, self.down3, self.down4])

        # ----- flat embeddings layer -----
        self.flatten = nn.Flatten()
        self.conv1x1 = nn.Conv2d(self.channels4 // bifactor // self.size_factor,1,1)
        self.fc1 = nn.Linear(1024 , 1024 // self.size_factor)
        self.fc2 = nn.Linear(1024 // self.size_factor , 2048 // self.size_factor)
        self.fc3 = nn.Linear(2048 // self.size_factor , 1024 // self.size_factor)
        self.fc4 = nn.Linear(1024 // self.size_factor, 1024 )        
        self.convto4 = nn.Conv2d(1,self.channels4 // self.size_factor // bifactor,1) 
        self.embeddings_layers = nn.ModuleList([self.conv1x1, self.fc1, self.fc2, self.fc3, self.fc4, self.convto4])

        # ----- decoder -----
        self.up1 = (Up(self.channels4 // self.size_factor, 128 // bifactor // self.size_factor, bilinear))
        self.up2 = (Up(128 // self.size_factor, 256 // bifactor // self.size_factor, bilinear))
        self.up3 = (Up(256 // self.size_factor, 128 // bifactor // self.size_factor, bilinear))
        self.up4 = (Up(128 // self.size_factor, 64 // self.size_factor, bilinear))
        self.outc = (OutConv(64 // self.size_factor, n_classes))
        self.decoder_layers = nn.ModuleList([self.up1, self.up2, self.up3, self.up4, self.outc])

        self.encoder_decoder_layers = nn.ModuleList([self.encoder_layers, self.decoder_layers])

        self.embeddings_decoder_layers = nn.ModuleList([self.embeddings_layers, self.decoder_layers])


    def param_checksum(self):
        sum_layers_params = lambda layers: sum([sum([i.sum() for i in list(layer.parameters())]) for layer in layers])

        return {'encoder': float(sum_layers_params(self.encoder_layers).detach().cpu().numpy()),
                'embeddings': float(sum_layers_params(self.embeddings_layers).detach().cpu().numpy()),
                'decoder': float(sum_layers_params(self.decoder_layers).detach().cpu().numpy())}


    def forward_encode(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x

    def forward_to_embeddings(self, x):
        x = self.conv1x1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward_from_embeddings(self, x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.reshape(len(x),1,32,32)
        x = F.relu(self.convto4(x))
        return x


    def forward_decode(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x

    def forward(self, x):
        x = self.forward_encode(x)
        if self.use_embeddings_block:
            x = self.forward_to_embeddings(x)
            x = self.forward_from_embeddings(x)
        x = self.forward_decode(x)
        x = F.relu(self.outc(x))
        return x
