# Main pre-training and evaluation script for SDO-FM

import os
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb
from sdofm.callback import ImagePredictionLoggerHMI
from sdofm.datasets import ZarrIrradianceDataModuleHMI
from sdofm.models.mae import HybridIrradianceModel
from sdofm.utils import flatten_dict


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

    wandb_logger = WandbLogger(
        # WandbLogger params
        name=cfg.experiment.name,
        project=cfg.experiment.project,
        dir=cfg.data.output_directory,
        # kwargs for wandb.init
        tags=cfg.experiment.wandb.tags,
        notes=cfg.experimentw.wandb.notes,
        group=cfg.experiment.wandb_group,
        save_code=True,
        job_type=cfg.experiment.job_type,
        config=flatten_dict(cfg),
    )

    data_loader = ZarrIrradianceDataModuleHMI(
        hmi_path=os.path.join(
            cfg.data.sdoml.base_directory, cfg.data.sdoml.instrument_sub_directory.hmi
        ),
        aia_path=os.path.join(
            cfg.data.sdoml.base_directory, cfg.data.sdoml.instrument_sub_directory.hmi
        ),
        eve_path=os.path.join(
            cfg.data.sdoml.base_directory, cfg.data.sdoml.instrument_sub_directory.eve
        ),
        components=cfg.data.sdoml.components,
        wavelengths=cfg.data.sdoml.wavelengths,
        # ions=run_config["sci_parameters"]["eve_ions"],
        # frequency=run_config["sci_parameters"]["frequency"],
        batch_size=cfg.model.opt.batch_size,
        num_workers=cfg.data.num_workers,
        # val_months=run_config["training_parameters"]["val_months"],
        # test_months=run_config["training_parameters"]["test_months"],
        # holdout_months=run_config["training_parameters"]["holdout_months"],
        cache_dir=cfg.data.sdoml.metadata,
    )

    data_loader.setup()
