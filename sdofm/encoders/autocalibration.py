# Modified from https://github.com/vale-salvatelli/sdo-autocal_pub/blob/master/src/sdo/models/autocalibration_models.py

"""
Torch architecture models for autocalibration.
"""

import logging

import torch
import torch.nn as nn

# from sdo.activations import positive_elu
## importing directly instead --


def positive_elu(x, alpha):
    """An elu translated of +1. This provides an activation function that:
    - is not capped at 1
    - never returns negative values
    - never returns exactly 0
    """
    return torch.nn.ELU(alpha)(x) + 1


class PositiveELU(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return positive_elu(x, self.alpha)


## --

_logger = logging.getLogger(__name__)


class Autocalibration1(nn.Module):
    """
    Two CNN layers and two fully connected layers.
    """

    def __init__(self, input_shape, output_dim):
        super(Autocalibration1, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))
        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels, out_channels=64, kernel_size=3
        )
        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))
        self._fc1 = nn.Linear(self._cnn_output_dim, 256)
        self._fc2 = nn.Linear(256, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = nn.MaxPool2d(kernel_size=3)(x)
        x = self._conv2d2(x)
        x = nn.MaxPool2d(kernel_size=3)(x)
        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc1(x)
        x = torch.relu(x)
        x = self._fc2(x)
        x = torch.sigmoid(x)
        return x


class Autocalibration2(nn.Module):
    """
    This model allows the dimension of all the parameters to be increased by
    'increase_dim'
    """

    def __init__(self, input_shape, output_dim, increase_dim=2):
        super(Autocalibration2, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))
        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels,
            out_channels=64 * increase_dim,
            kernel_size=3,
        )
        self._conv2d2 = nn.Conv2d(
            in_channels=64 * increase_dim,
            out_channels=128 * increase_dim,
            kernel_size=3,
        )
        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))
        self._fc1 = nn.Linear(self._cnn_output_dim, 256 * increase_dim)
        self._fc2 = nn.Linear(256 * increase_dim, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = nn.MaxPool2d(kernel_size=3)(x)
        x = self._conv2d2(x)
        x = nn.MaxPool2d(kernel_size=3)(x)
        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc1(x)
        x = torch.relu(x)
        x = self._fc2(x)
        x = torch.sigmoid(x)
        return x


class Autocalibration3(nn.Module):
    """
    Uses a leaky relu instead of a sigmoid as its final activation function.
    """

    def __init__(self, input_shape, output_dim):
        super(Autocalibration3, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))
        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels, out_channels=64, kernel_size=3
        )
        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))
        self._fc1 = nn.Linear(self._cnn_output_dim, 256)
        self._fc2 = nn.Linear(256, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = nn.MaxPool2d(kernel_size=3)(x)
        x = self._conv2d2(x)
        x = nn.MaxPool2d(kernel_size=3)(x)
        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc1(x)
        x = torch.relu(x)
        x = self._fc2(x)
        x = torch.nn.LeakyReLU()(x)
        return x


class Autocalibration4(nn.Module):
    """
    Scales free parameters by the size of the resolution, as well as uses
    a leaky relu at the end.
    """

    def __init__(self, input_shape, output_dim, scaled_resolution):
        super(Autocalibration4, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))

        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels,
            out_channels=int(scaled_resolution / 2),
            kernel_size=3,
        )
        _logger.info("CNN1, out_channels: {}".format(int(scaled_resolution / 2)))

        self._conv2d2 = nn.Conv2d(
            in_channels=int(scaled_resolution / 2),
            out_channels=scaled_resolution,
            kernel_size=3,
        )
        _logger.info("CNN2, out_channels: {}".format(int(scaled_resolution / 2)))

        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))

        self._fc1 = nn.Linear(self._cnn_output_dim, scaled_resolution * 2)
        _logger.info("FCN layer inter-connects: {}".format(scaled_resolution * 2))
        self._fc2 = nn.Linear(scaled_resolution * 2, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = nn.MaxPool2d(kernel_size=3)(x)
        x = self._conv2d2(x)
        x = nn.MaxPool2d(kernel_size=3)(x)
        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc1(x)
        x = torch.relu(x)
        x = self._fc2(x)
        x = torch.nn.LeakyReLU()(x)
        return x


class Autocalibration5(nn.Module):
    """
    Add more convolutional layers.
    """

    def __init__(self, input_shape, output_dim, scaled_resolution):
        super(Autocalibration5, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))

        conv1_in = self._input_channels
        conv1_out = int(scaled_resolution / 2)
        self._conv2d1 = nn.Conv2d(
            in_channels=conv1_in, out_channels=conv1_out, kernel_size=3
        )
        _logger.info("\nCNN1, in_channels: {}".format(conv1_in))
        _logger.info("CNN1, out_channels: {}".format(conv1_out))

        conv2_in = int(scaled_resolution / 2)
        conv2_out = scaled_resolution
        self._conv2d2 = nn.Conv2d(
            in_channels=conv2_in, out_channels=conv2_out, kernel_size=3
        )
        _logger.info("\nCNN2, in_channels: {}".format(conv2_in))
        _logger.info("CNN2, out_channels: {}".format(conv2_out))

        conv3_in = scaled_resolution
        conv3_out = scaled_resolution * 2
        self._conv2d3 = nn.Conv2d(
            in_channels=conv3_in, out_channels=conv3_out, kernel_size=3
        )
        _logger.info("\nCNN3, in_channels: {}".format(conv3_in))
        _logger.info("CNN3, out_channels: {}".format(conv3_out))

        cnn_output_dim = torch.zeros(input_shape).unsqueeze(0)
        _logger.info("\ncnn_output_dim.shape: {}".format(cnn_output_dim.shape))
        cnn_output_nelems = self._cnn(cnn_output_dim).nelement()
        _logger.info("cnn_output_nelems: {}".format(cnn_output_nelems))

        fc1_in = cnn_output_nelems
        fc1_out = scaled_resolution * 2
        self._fc1 = nn.Linear(fc1_in, fc1_out)
        _logger.info("\nFCN1, in_channels: {}".format(fc1_in))
        _logger.info("FCN1, out_channels: {}".format(fc1_out))

        fc2_in = scaled_resolution * 2
        fc2_out = output_dim
        self._fc2 = nn.Linear(fc2_in, fc2_out)
        _logger.info("FCN2, in_channels: {}".format(fc2_in))
        _logger.info("FCN2, out_channels: {}".format(fc2_out))

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = nn.MaxPool2d(kernel_size=3)(x)

        x = self._conv2d2(x)
        x = torch.relu(x)
        x = nn.MaxPool2d(kernel_size=3)(x)

        # Note: no relu on the final convolution.
        x = self._conv2d3(x)
        x = nn.MaxPool2d(kernel_size=3)(x)

        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)

        x = self._fc1(x)
        x = torch.relu(x)

        x = self._fc2(x)
        x = torch.nn.LeakyReLU()(x)

        return x


class Autocalibration6(nn.Module):
    """
    How simple can we get our network to be and still perform well at
    128x128 and 256x256?
    """

    def __init__(self, input_shape, output_dim):
        super(Autocalibration6, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))

        # Note: Two convolutional layers are needed to get results.
        # Wavelength 94 does bad _unless_ we restore the amount of filter banks to 64
        # across both CNN layer 1 and 2. Wavelength 171 was fine with smaller filter
        # banks (32) however
        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels, out_channels=64, kernel_size=3
        )
        self._conv2d1_maxpool = nn.MaxPool2d(kernel_size=3)

        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._conv2d2_maxpool = nn.MaxPool2d(kernel_size=3)

        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))

        self._fc = nn.Linear(self._cnn_output_dim, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = self._conv2d1_maxpool(x)

        x = self._conv2d2(x)
        x = torch.relu(x)
        x = self._conv2d2_maxpool(x)

        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc(x)
        x = torch.sigmoid(x)
        return x


class Autocalibration7(nn.Module):
    """
    Can we create a separate column for processing every wavelength independently,
    combining the final filter banks from all wavelengths feeding into a single
    fully connected layer so the channels can influence each other in the final
    regression results?
    """

    def __init__(self, input_shape, output_dim, device):
        super(Autocalibration7, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))

        self.device = device

        num_channels = input_shape[0]
        scaled_height = input_shape[1]
        scaled_width = input_shape[2]

        # Create an independent column of CNN layers for each wavelength channel;
        # flatten each of their outputs, then feed them all into a single final
        # fully connected layer.
        self._columns = nn.ModuleList(
            [self._create_column() for c in range(num_channels)]
        )

        fake_x = torch.zeros([1, scaled_height, scaled_width]).unsqueeze(0)
        cnn_output_dim = self._columns[0](fake_x).nelement()

        self._fc1 = nn.Linear(cnn_output_dim * num_channels, 256 * 3)
        self._fc2 = nn.Linear(256 * 3, output_dim)

    def _create_column(self):
        # Note: pytorch has 'magic' on members of the actual class, shifting them
        # into and out of the correct device, so put these column values on ourself.
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_dim = x.shape[0]
        num_channels = x.shape[1]

        col_outs = []
        for c in range(num_channels):
            # TODO: Technically these can all run in parallel, rather than
            # serially, for better performance.
            per_channel_in = x[:, c].unsqueeze(1)
            per_channel_out = self._columns[c](per_channel_in)

            # Since the values coming out of each wavelength column are
            # unbalanced (i.e. some channels can have larger values),
            # force normalize them over the batch so that later processing
            # is not dominated by the larger channels.
            per_channel_out = per_channel_out / per_channel_out.max()
            col_outs.append(per_channel_out.view(batch_dim, -1))

        x = torch.cat(col_outs, dim=1)

        x = self._fc1(x)
        x = torch.nn.LeakyReLU()(x)

        x = self._fc2(x)
        x = torch.nn.LeakyReLU()(x)

        return x


class Autocalibration8(nn.Module):
    """
    Same as Autocalibration6, but use a plain vanilla ReLU as the final activation function
    rather than a sigmoid.
    """

    def __init__(self, input_shape, output_dim):
        super(Autocalibration8, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))

        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels, out_channels=64, kernel_size=3
        )
        self._conv2d1_maxpool = nn.MaxPool2d(kernel_size=3)

        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._conv2d2_maxpool = nn.MaxPool2d(kernel_size=3)

        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))

        self._fc = nn.Linear(self._cnn_output_dim, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = self._conv2d1_maxpool(x)

        x = self._conv2d2(x)
        x = torch.relu(x)
        x = self._conv2d2_maxpool(x)

        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc(x)
        x = torch.relu(x)
        return x


class Autocalibration9(nn.Module):
    """
    Building off of the Autocalibration6 model, introduces a sigmoid scale
    value that can scale the final sigmoid activation function.
    """

    def __init__(self, input_shape, output_dim, sigmoid_scale):
        super(Autocalibration9, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        self._sigmoid_scale = sigmoid_scale
        _logger.info("input_channels: {}".format(self._input_channels))

        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels, out_channels=64, kernel_size=3
        )
        self._conv2d1_maxpool = nn.MaxPool2d(kernel_size=3)

        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._conv2d2_maxpool = nn.MaxPool2d(kernel_size=3)

        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))

        self._fc = nn.Linear(self._cnn_output_dim, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = self._conv2d1_maxpool(x)

        x = self._conv2d2(x)
        x = torch.relu(x)
        x = self._conv2d2_maxpool(x)

        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc(x)
        x = self._sigmoid_scale * torch.sigmoid(x)
        return x


class Autocalibration10(nn.Module):
    """
    Same as Autocalibration6, but use a ReLU6 activation function as the final function
    replacing the sigmoid to get a clipped relu value.
    """

    def __init__(self, input_shape, output_dim):
        super(Autocalibration10, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))

        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels, out_channels=64, kernel_size=3
        )
        self._conv2d1_maxpool = nn.MaxPool2d(kernel_size=3)

        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._conv2d2_maxpool = nn.MaxPool2d(kernel_size=3)

        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))

        self._fc = nn.Linear(self._cnn_output_dim, output_dim)

        self._relu6 = torch.nn.ReLU6()

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = self._conv2d1_maxpool(x)

        x = self._conv2d2(x)
        x = torch.relu(x)
        x = self._conv2d2_maxpool(x)

        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc(x)
        x = self._relu6(x)
        return x


class Autocalibration11(nn.Module):
    """
    Same as Autocalibration6, but it implements heteroscedastic regression.
    """

    def __init__(self, input_shape, output_dim):
        super(Autocalibration11, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))

        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels, out_channels=64, kernel_size=3
        )
        self._conv2d1_maxpool = nn.MaxPool2d(kernel_size=3)

        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._conv2d2_maxpool = nn.MaxPool2d(kernel_size=3)

        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))

        self._fc = nn.Linear(self._cnn_output_dim, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = self._conv2d1_maxpool(x)

        x = self._conv2d2(x)
        x = torch.relu(x)
        x = self._conv2d2_maxpool(x)

        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        x = self._fc(x)
        mean = torch.sigmoid(x)
        log_var = positive_elu(x, alpha=0.2)
        return torch.stack([mean, log_var], dim=0)


class Autocalibration12(nn.Module):
    """
    Same as Autocalibration11, but separate fully connected layer for x and y
    """

    def __init__(self, input_shape, output_dim):
        super(Autocalibration12, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))

        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels, out_channels=64, kernel_size=3
        )
        self._conv2d1_maxpool = nn.MaxPool2d(kernel_size=3)

        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._conv2d2_maxpool = nn.MaxPool2d(kernel_size=3)

        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))

        self._fc1 = nn.Linear(self._cnn_output_dim, output_dim)
        self._fc2 = nn.Linear(self._cnn_output_dim, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = self._conv2d1_maxpool(x)

        x = self._conv2d2(x)
        x = torch.relu(x)
        x = self._conv2d2_maxpool(x)

        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        mean = self._fc1(x)
        log_var = self._fc2(x)
        mean = torch.sigmoid(mean)
        log_var = positive_elu(log_var, alpha=0.2)
        return torch.stack([mean, log_var], dim=0)


class Autocalibration13(nn.Module):
    """
    Same as Autocalibration12, but using ELU as activation function for the log_var
    """

    def __init__(self, input_shape, output_dim):
        super(Autocalibration13, self).__init__()
        if len(input_shape) != 3:
            raise ValueError("Expecting an input_shape representing dimensions CxHxW")
        self._input_channels = input_shape[0]
        _logger.info("input_channels: {}".format(self._input_channels))

        self._conv2d1 = nn.Conv2d(
            in_channels=self._input_channels, out_channels=64, kernel_size=3
        )
        self._conv2d1_maxpool = nn.MaxPool2d(kernel_size=3)

        self._conv2d2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self._conv2d2_maxpool = nn.MaxPool2d(kernel_size=3)

        self._cnn_output_dim = self._cnn(
            torch.zeros(input_shape).unsqueeze(0)
        ).nelement()
        _logger.info("cnn_output_dim: {}".format(self._cnn_output_dim))

        self._fc1 = nn.Linear(self._cnn_output_dim, output_dim)
        self._fc2 = nn.Linear(self._cnn_output_dim, output_dim)

    def _cnn(self, x):
        x = self._conv2d1(x)
        x = torch.relu(x)
        x = self._conv2d1_maxpool(x)

        x = self._conv2d2(x)
        x = torch.relu(x)
        x = self._conv2d2_maxpool(x)

        return x

    def forward(self, x):
        batch_dim = x.shape[0]
        x = self._cnn(x).view(batch_dim, -1)
        mean = self._fc1(x)
        log_var = self._fc2(x)
        mean = torch.sigmoid(mean)
        log_var = nn.ELU(alpha=10.0)(log_var)
        return torch.stack([mean, log_var], dim=0)
