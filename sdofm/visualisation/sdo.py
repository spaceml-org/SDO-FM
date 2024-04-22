import datetime
import os

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps as sunpycm
from tqdm import tqdm

# mpl.use("Agg")

# colors = ['#0000ff', '#ffffff', '#ff0000']
# cm = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors, 1024)

LABELS = [
    "HMI Bx",
    "HMI By",
    "HMI Bz",
    "AIA 94 Å",
    "AIA 131 Å",
    "AIA 171 Å",
    "AIA 193 Å",
    "AIA 211 Å",
    "AIA 304 Å",
    "AIA 335 Å",
    "AIA 1600 Å",
    "AIA 1700 Å",
]

cms = [
    # sunpycm.cmlist.get("hmimag"),
    # sunpycm.cmlist.get("hmimag"),
    # sunpycm.cmlist.get("hmimag"),
    sunpycm.cmlist.get("sdoaia94"),
    sunpycm.cmlist.get("sdoaia131"),
    sunpycm.cmlist.get("sdoaia171"),
    sunpycm.cmlist.get("sdoaia193"),
    sunpycm.cmlist.get("sdoaia211"),
    sunpycm.cmlist.get("sdoaia304"),
    sunpycm.cmlist.get("sdoaia335"),
    sunpycm.cmlist.get("sdoaia1600"),
    sunpycm.cmlist.get("sdoaia1700"),
]

import torch

aia_cutoff = torch.tensor(
    [
        1588.2617,
        2652.1470,
        22816.1035,
        23919.7168,
        13458.3203,
        24765.3145,
        1690.6078,
        3399.5896,
        16458.3223,
    ]
)
sqrt_aia_cutoff = aia_cutoff.sqrt().reshape(1, 9, 1, 1)

hmi_min = -1500
hmi_max = 1500
hmi_range = hmi_max - hmi_min


def normalize_hmi(data):
    data = torch.clamp(data, min=hmi_min, max=hmi_max)
    data = (data - hmi_min) / hmi_range
    return data


def unnormalize_hmi(data):
    data = data * hmi_range
    data = data + hmi_min
    return data


def normalize_aia(data):
    #     if data.ndim != 4:
    #         raise ValueError('Expecting 4d data')
    #     if data.shape[1] != 9:
    #         raise ValueError('Expecting 9 channels')
    c = sqrt_aia_cutoff.to(data.device).expand(data.shape)
    data = torch.sqrt(data)
    data = torch.clamp(data, max=c)
    data = data / c
    return data


def unnormalize_aia(data):
    #     if data.ndim != 4:
    #         raise ValueError('Expecting 4d data')
    #     if data.shape[1] != 9:
    #         raise ValueError('Expecting 9 channels')
    c = sqrt_aia_cutoff.to(data.device).expand(data.shape)
    data = data * c
    data = data.pow(2.0)
    return data


def normalize(data):
    if data.ndim == 4:
        hmi = data[:, :3]
        aia = data[:, 3:]
        hmi = normalize_hmi(hmi)
        aia = normalize_aia(aia)
        data = torch.cat([hmi, aia], dim=1)
        return data
    elif data.ndim == 3:
        data = data.unsqueeze(0)
        data = normalize(data)
        data = data.squeeze(0)
        return data
    else:
        raise ValueError("Expecting 3d or 4d data")


def unnormalize(data):
    if data.ndim == 4:
        hmi = data[:, :3]
        aia = data[:, 3:]
        hmi = unnormalize_hmi(hmi)
        aia = unnormalize_aia(aia)
        data = torch.cat([hmi, aia], dim=1)
        return data
    elif data.ndim == 3:
        data = data.unsqueeze(0)
        data = unnormalize(data)
        data = data.squeeze(0)
        return data
    else:
        raise ValueError("Expecting 3d or 4d data")


def add_file_name_suffix(file_name, suffix):
    fn, ext = os.path.splitext(file_name)
    return "{}{}{}".format(fn, suffix, ext)


def sdo_plot(
    data,
    file_name=None,
    data_normalized=False,
    colored=True,
    figsize=(12, 9),
    date=None,
):
    if data.ndim != 3:
        raise ValueError("Expecting 3d data (CxHxW)")
    data = data.detach().cpu()
    if data_normalized:
        data = unnormalize(data)

    fig, axs = plt.subplots(3, 4, figsize=figsize)
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    for i, ax in enumerate(axs.flat):
        ax.axis("off")
        if colored:
            cmap = cms[i]
        else:
            cmap = "viridis"
        # Clip values based on https://docs.sunpy.org/en/stable/generated/gallery/map_transformations/autoalign_aia_hmi.html
        if i <= 2:  # HMI
            vmin, vmax = -1500, 1500
            d = np.clip(data[i], vmin, vmax)
        else:  # AIA
            vmin, vmax = np.percentile(data[i], (1, 99.9))
            d = data[i]
        ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(LABELS[i])
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    if date is not None:
        plt.suptitle(date)
    if file_name is None:
        plt.show()
    else:
        # print('Saving plot to {}'.format(file_name))
        plt.savefig(file_name)
        plt.close(fig)


def sdo_reconstruction_plot(
    data_original,
    data_reconstruction,
    file_name=None,
    data_normalized=False,
    colored=True,
    figsize=(36, 12),
    date=None,
):
    if data_original.ndim != 3:
        raise ValueError("Expecting 3d data (CxHxW)")
    if data_reconstruction.ndim != 3:
        raise ValueError("Expecting 3d data (CxHxW)")
    data_original = data_original.detach().cpu()
    data_reconstruction = data_reconstruction.detach().cpu()
    # if data_normalized:
    #     data_original_normalized = data_original
    #     data_reconstruction_normalized = data_reconstruction
    #     data_original = unnormalize(data_original)
    #     data_reconstruction = unnormalize(data_reconstruction)
    # else:
    #     data_original_normalized = normalize(data_original)
    #     data_reconstruction_normalized = normalize(data_reconstruction)

    fig, axs = plt.subplots(4, 12, figsize=figsize)
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    axs = axs.flat
    for ax in axs:
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    for i in range(9):
        axs[i].set_title(LABELS[i])

        # original
        cmap = cms[i] if colored else "gray"
        # if i < 3:  # HMI
        #     vmin, vmax = -1500, 1500
        #     do = np.clip(data_original[i], vmin, vmax)
        # else:  # AIA
        do = data_original[i]
        vmin, vmax = np.percentile(data_original[i], (1, 99.9))
        axs[i].imshow(do, cmap=cmap, vmin=vmin, vmax=vmax)

        # reconstruction
        cmap = cms[i] if colored else "gray"
        # if i < 3:  # HMI
        #     vmin, vmax = -1500, 1500
        #     dr = np.clip(data_reconstruction[i], vmin, vmax)
        # else:  # AIA
        vmin, vmax = np.percentile(
            data_original[i], (1, 99.9)
        )  # use original data for vmin/vmax of reconstruction
        dr = data_reconstruction[i]
        axs[12 + i].imshow(dr, cmap=cmap, vmin=vmin, vmax=vmax)

        # histograms (unnormalized)
        n_samples = 10000
        n_bins = 50
        axs[24 + i].hist(
            np.random.choice(do.flatten(), n_samples),
            bins=n_bins,
            alpha=0.5,
            density=True,
            label="Original",
        )
        axs[24 + i].hist(
            np.random.choice(dr.flatten(), n_samples),
            bins=n_bins,
            alpha=0.5,
            density=True,
            label="Reconstruction",
        )
        axs[24 + i].set_yscale("log")
        axs[24 + i].set_xscale("linear")
        if i == 0:
            axs[24 + i].legend(loc="upper left")
        else:
            axs[24 + i].set_yticks([])
            axs[24 + i].spines["left"].set_visible(False)
            axs[24 + i].minorticks_off()

        # do_normalized = data_original_normalized[i]
        # dr_normalized = data_reconstruction_normalized[i]
        do_normalized = data_original[i]
        dr_normalized = data_reconstruction[i]

        # histograms (normalized)
        n_samples = 10000
        n_bins = 50
        axs[36 + i].hist(
            np.random.choice(do_normalized.flatten(), n_samples),
            bins=n_bins,
            alpha=0.5,
            density=True,
            label="Original",
        )
        axs[36 + i].hist(
            np.random.choice(dr_normalized.flatten(), n_samples),
            bins=n_bins,
            alpha=0.5,
            density=True,
            label="Reconstruction",
        )
        axs[36 + i].set_yscale("log")
        axs[36 + i].set_xscale("linear")
        if i == 0:
            axs[36 + i].legend(loc="upper left")
        else:
            axs[36 + i].set_yticks([])
            axs[36 + i].spines["left"].set_visible(False)
            axs[36 + i].minorticks_off()

    axs[0].set_ylabel("Original", fontsize="large")
    axs[12].set_ylabel("Reconstruction", fontsize="large")
    axs[24].set_ylabel("Histograms", fontsize="large")
    axs[36].set_ylabel("Histograms (normalized)", fontsize="large")

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    if date is not None:
        plt.suptitle(date)
    if file_name is None:
        plt.show()
    else:
        # print('Saving plot to {}'.format(file_name))
        plt.savefig(file_name)
        plt.close(fig)


def sdo_video(
    data,
    file_name=None,
    data_normalized=False,
    colored=True,
    figsize=(12, 9),
    date=None,
    delta_minutes=6,
    fps=15,
):
    if data.ndim != 4:
        raise ValueError("Expecting 4d data (SxCxHxW)")
    data = data.detach().cpu()
    if data_normalized:
        data = unnormalize(data)
    size_pixels = data.shape[-1]

    fig, axs = plt.subplots(3, 4, figsize=figsize)
    title = plt.suptitle(date)

    ims = []
    for j, ax in enumerate(axs.flat):
        ax.axis("off")
        if colored:
            cmap = cms[j]
        else:
            cmap = "viridis"
        if j <= 2:  # HMI
            vmin, vmax = -1500, 1500
            d = np.zeros([size_pixels, size_pixels])
        else:  # AIA
            vmin, vmax = np.percentile(data[:, j], (1, 99.9))
            d = np.zeros([size_pixels, size_pixels])
        ims.append(ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax))
        ax.set_title(LABELS[j])
    frames = data.shape[0]
    pbar = tqdm(total=frames)

    def run(i):
        pbar.update(1)
        for j in range(12):
            if i <= 2:  # HMI
                vmin, vmax = -1500, 1500
                data[i, j] = np.clip(data[i, j], vmin, vmax)
            else:  # AIA
                vmin, vmax = np.percentile(data[i, j], (1, 99.9))
            ims[j].set_data(data[i, j].clamp(vmin, vmax))
        if date is not None:
            title.set_text(date + datetime.timedelta(minutes=i * delta_minutes))

    plt.tight_layout(rect=[0, 0, 1, 0.975])
    anim = animation.FuncAnimation(fig, run, interval=300, frames=frames)
    if file_name is None:
        return anim
    else:
        # print('Saving video to {}'.format(file_name))
        writervideo = animation.FFMpegWriter(fps=fps)
        anim.save(file_name, writer=writervideo)
        plt.close(fig)
