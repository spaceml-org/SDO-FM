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
import tensorflow as tf
import wandb
# from lightning.pytorch import seed_everything
# from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from sdofm import utils  # import days_hours_mins_secs_str
from sdofm.utils import flatten_dict

# import torch_xla.debug.profiler as xp

wandb_logger = None


# loads the config file
@hydra.main(config_path="../experiments", config_name="default")
def main(cfg: DictConfig) -> None:
    # set seed
    tf.keras.utils.set_random_seed(cfg.experiment.seed)

    # set device using config disable_cuda option and torch.cuda.is_available()
    profiler = None
    match cfg.experiment.profiler:
        case None:
            profiler = None
        case _:
            raise NotImplementedError(
                f"Profiler {cfg.experiment.profiler} is not implemented."
            )

    # set precision of torch tensors
    # if cfg.experiment.device == "cuda":
    match cfg.experiment.precision:
        case "mixed_float16":
            tf.keras.mixed_precision.mixed_precision.set_global_policy(
                cfg.experiment.precision
            )
        case _:
            warnings.warn(
                f"Setting precision {cfg.experiment.precision} is not supported."
            )

    # run experiment
    print(f"\nRunning with config:")
    print(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    print("\n")

    print(f"Using device: {cfg.experiment.accelerator}")

    if not cfg.experiment.disable_wandb:
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
            "offline" if cfg.experiment.disable_wandb else "online"
        )

        logger = wandb.init(
            # WandbLogger params
            name=cfg.experiment.name,
            project=cfg.experiment.project,
            dir=cfg.experiment.wandb.output_directory,
            log_model="all",
            # kwargs for wandb.init
            tags=cfg.experiment.wandb.tags,
            notes=cfg.experiment.wandb.notes,
            group=cfg.experiment.wandb.group,
            save_code=True,
            job_type=cfg.experiment.wandb.job_type,
            config=flatten_dict(cfg, sep="___"),  # may want to + tf.FLAGS
            sync_tensorboard=True,
        )
    else:
        logger = None

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
