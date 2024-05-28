"""
Main entry point. Uses Hydra to load config files and override defaults with command line args
"""

import os
import random
import time
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
import warnings

from sdofm import utils  # import days_hours_mins_secs_str
from sdofm.utils import flatten_dict
import logging

# import torch_xla.debug.profiler as xp

wandb_logger = None


# loads the config file
@hydra.main(config_path="../experiments", config_name="default")
def main(cfg: DictConfig) -> None:

    match cfg.log_level:
        case "DEBUG":
            logging.basicConfig(level=logging.DEBUG)
        case _:
            logging.basicConfig(level=logging.INFO)

    # set seed
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    seed_everything(cfg.experiment.seed)

    # set device using config disable_cuda option and torch.cuda.is_available()
    profiler = None
    match cfg.experiment.profiler:
        case "XLAProfiler":
            from lightning.pytorch.profilers import XLAProfiler

            profiler = XLAProfiler(port=9014)
        case "PyTorchProfiler":
            from lightning.pytorch.profilers import PyTorchProfiler

            profiler = PyTorchProfiler(
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/sdofm"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
        case "Profiler":
            from lightning.pytorch.profilers import Profiler

            profiler = Profiler()
        case None:
            profiler = None
        case _:
            raise NotImplementedError(
                f"Profiler {cfg.experiment.profiler} is not implemented."
            )

    # set precision of torch tensors
    match cfg.experiment.precision:
        case 64:
            torch.set_default_tensor_type(torch.DoubleTensor)
        case 32:
            torch.set_default_tensor_type(torch.FloatTensor)
        case _:
            warnings.warn(
                f"Setting precision {cfg.experiment.precision} will pass through to the trainer but not other operations."
            )
            # raise NotImplementedError(
            #     f"Precision {cfg.experiment.precision} not implemented"
            # )

    # run experiment
    print(f"\nRunning with config:")
    print(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    print("\n")

    print(f"Using device: {cfg.experiment.accelerator}")

    # set up wandb logging
    if cfg.experiment.wandb.enable:
        wandb.login()
        output_dir = Path(cfg.experiment.wandb.output_directory)
        output_dir.mkdir(exist_ok=True, parents=True)
        print(
            f"Created directory for storing results: {cfg.experiment.wandb.output_directory}"
        )
        cache_dir = Path(f"{cfg.experiment.wandb.output_directory}/.cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        os.environ["WANDB_CACHE_DIR"] = (
            f"{cfg.experiment.wandb.output_directory}/.cache"
        )
        os.environ["WANDB_MODE"] = (
            "offline" if not cfg.experiment.wandb.enable else "online"
        )

        logger = WandbLogger(
            # WandbLogger params
            name=cfg.experiment.name,
            project=cfg.experiment.project,
            dir=cfg.experiment.wandb.output_directory,
            log_model=cfg.experiment.wandb.log_model,
            # kwargs for wandb.init
            tags=cfg.experiment.wandb.tags,
            notes=cfg.experiment.wandb.notes,
            group=cfg.experiment.wandb.group,
            save_code=True,
            job_type=cfg.experiment.wandb.job_type,
            config=flatten_dict(cfg),
        )
    else:
        logger = None

    # set up checkpointing to gcp bucket 
    if cfg.experiment.gcp_storage:
        from google.cloud import storage
        client = storage.Client() # project='myproject'
        bucket = client.get_bucket(cfg.experiment.gcp_storage.bucket)
        if not bucket.exists():
            raise ValueError("Not authenticated or cannot find provided Google Storage bucket, is your machine authenticated?")

    match cfg.experiment.task:
        case "pretrain":
            from scripts.pretrain import Pretrainer

            pretrainer = Pretrainer(cfg, logger=logger, profiler=profiler)
            pretrainer.run()
        case "finetune":
            from scripts.finetune import Finetuner

            finetuner = Finetuner(cfg, logger=logger, profiler=profiler)
            finetuner.run()
        case _:
            raise NotImplementedError(
                f"Experiment {cfg.experiment.task} not implemented"
            )


if __name__ == "__main__":
    # server = xp.start_server(9012)
    time_start = time.time()
    # errors
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace
    main()
    print(
        "\nTotal duration: {}".format(
            utils.days_hours_mins_secs_str(time.time() - time_start)
        )
    )
