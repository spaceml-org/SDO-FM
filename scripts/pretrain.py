# Main pretraining and evaluation script for SDO-FM

import os
from pathlib import Path

import torch

import wandb


def pretrain_sdofm(cfg):
    print("SDO-FM Model Pre-training")

    # set precision of torch tensors
    if cfg.experiment.precision == 64:
        torch.set_default_tensor_type(torch.DoubleTensor)
    elif cfg.experiment.precision == 32:
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        raise NotImplementedError(
            f"Precision {cfg.experiment.precision} not implemented"
        )

    output_dir = Path(cfg.data.output_directory)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Created directory for storing results: {cfg.data.output_directory}")
    cache_dir = Path(f"{cfg.data.output_directory}/.cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    os.environ["WANDB_CACHE_DIR"] = f"{cfg.data.output_directory}/.cache"
    os.environ["WANDB_MODE"] = "offline" if cfg.experiment.disable_wandb else "online"
    run = wandb.init(
        project=cfg.experiment.project,
        group=cfg.experiment.wandb_group,
        entity=cfg.experiment.wandb_entity,
        save_code=True,
        dir=cfg.data.output_directory,
        job_type=cfg.experiment.job_type,
        settings=wandb.Settings(start_method="thread"),
    )
