# Adapted from: FDL 2021 Solar Drag - Feature Extraction
# Learning the solar latent space: sigma-variational autoencoders for multiple channel solar imaging
# https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_83.pdf

from copy import deepcopy
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

RADIUS_FRACTION_OF_IMAGE = 0.40625
METRICS_METHODS = {
    "flux_difference": lambda real, generated: relative_total_flux_error(
        real, generated
    ),
    "ppe10s": lambda real, generated: pixel_percentage_error(real, generated, 0.1),
    "ppe50s": lambda real, generated: pixel_percentage_error(real, generated, 0.5),
    "rms_contrast_measure": lambda real, generated: rms_contrast_measure(
        real, generated
    ),
    "pixel_correlation": lambda real, generated: pixel_correlation_coefficient(
        real, generated
    ),
    "rmse_intensity": lambda real, generated: rmse_intensity(real, generated),
}

METRICS = list(METRICS_METHODS.keys())
MIN_CLIP = 1e-2
MAX_CLIP = np.inf
MASK_DISK = True


def get_metrics(real, generated, channels, mask_disk=MASK_DISK):
    ## Expect CxHxW
    if torch.is_tensor(real):
        real = real.cpu().detach().numpy()
        generated = generated.cpu().detach().numpy()

    ## Slightly hacky way of ignoring masked images
    if real.mean() == generated.mean():
        return None
    _, _, _, image_size = real.shape
    mask = disk_mask(image_size)

    metrics = {}
    for channel in channels:
        metrics[channel] = {}
        for metric in METRICS:
            metrics[channel][metric] = []

    for c, channel in enumerate(channels):
        if mask_disk and c <= 2:
            real_channel = (real[c, :, :] * mask).flatten()
            generated_channel = (generated[c, :, :] * mask).flatten()
        else:
            real_channel = real[c, :, :].flatten()
            generated_channel = generated[c, :, :].flatten()

        #         real_channel = np.clip(real_channel, MIN_CLIP, MAX_CLIP)
        #         generated_channel = np.clip(generated_channel, MIN_CLIP, MAX_CLIP)

        for metric, function in METRICS_METHODS.items():
            metrics[channel][metric].append(function(real_channel, generated_channel))

    return metrics


def get_batch_metrics(real_batch, generated_batch, channels):
    ## Expect BxCxHxW
    batch_size = real_batch.shape[0]
    metrics_list = []
    for sample_idx in range(batch_size):
        sample_metrics = get_metrics(
            real_batch[sample_idx, :, :, :],
            generated_batch[sample_idx, :, :, :],
            channels,
        )
        metrics_list.append(sample_metrics)

    merged_metrics = merge_metrics(metrics_list)
    return mean_metrics(merged_metrics)


@lru_cache(maxsize=10)
def disk_mask(image_size):
    img_half = image_size / 2
    radius = int(RADIUS_FRACTION_OF_IMAGE * float(image_size))
    disk_mask = np.zeros((image_size, image_size))
    for h in range(image_size):
        for w in range(image_size):
            distance_to_centre = np.sqrt((h - img_half) ** 2 + (w - img_half) ** 2)
            if distance_to_centre < radius:
                disk_mask[h][w] = 1
    return disk_mask


def merge_metrics(metrics_list):
    channels = list(metrics_list[0].keys())
    merged_metrics = {}
    for channel in channels:
        merged_metrics[channel] = {}
        for metric in METRICS:
            merged_metrics[channel][metric] = []
            for cm in metrics_list:
                # print(cm[channel][metric])
                merged_metrics[channel][metric] += cm[channel][metric]
                # print(merged_metrics[channel][metric])

    return merged_metrics


def mean_metrics(metrics):
    mean_metrics = {}
    for channel, channel_metrics in metrics.items():
        mean_metrics[channel] = {}
        for metric in METRICS:
            mean_metrics[channel][metric] = np.mean(channel_metrics[metric])

    return mean_metrics


def pixel_percentage_error(real, generated, threshold_ptc):
    difference = real - generated
    fraction = np.divide(difference, real)
    absolute_fraction = np.abs(fraction)
    ppe_binary = (absolute_fraction < threshold_ptc).astype(int)
    mean_ppe = ppe_binary.mean()

    return mean_ppe


def pixel_correlation_coefficient(real, generated):
    correlation = np.corrcoef(real, generated)[0, 1]
    return correlation


def rms_contrast_measure(real, generated):
    difference_squared = np.power(real.mean() - generated, 2)
    mean_difference = difference_squared.mean()
    rms = np.sqrt(mean_difference)
    return rms


def relative_total_flux_error(real, generated):
    return (generated.sum() - real.sum()) / real.sum()


def rmse_intensity(real, generated):
    difference_squared = np.power(real - generated, 2)
    mean_difference = difference_squared.mean()
    rms = np.sqrt(mean_difference)
    return rms
