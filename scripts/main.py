"""
Main entry point. Uses Hydra to load config files and override defaults with command line args
"""

import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import XLAProfiler, Profiler
import warnings

from sdofm import utils  # import days_hours_mins_secs_str
from sdofm.utils import flatten_dict

# import torch_xla.debug.profiler as xp

wandb_logger = None


# loads the config file
@hydra.main(config_path="../experiments", config_name="default")
def main(cfg: DictConfig) -> None:
    # set seed
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)
    seed_everything(cfg.experiment.seed)

    # set device using config disable_cuda option and torch.cuda.is_available()
    match cfg.experiment.accelerator:
        case "cuda":
            if (not cfg.experiment.disable_cuda) and torch.cuda.is_available():
                cfg.experiment.device ="cpu"
                warnings.warn("CUDA not available, reverting to CPU!")
            profiler = Profiler()
        case "tpu":
            profiler = XLAProfiler(port=9012)

    # set precision of torch tensors
    if cfg.experiment.accelerator == "cuda":
        match cfg.experiment.precision:
            case 64:
                torch.set_default_tensor_type(torch.DoubleTensor)
            case 32:
                torch.set_default_tensor_type(torch.FloatTensor)
            case _:
                raise NotImplementedError(
                    f"Precision {cfg.experiment.precision} not implemented"
                )

    # run experiment
    print(f"\nRunning with config:")
    print(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    print("\n")

    print(f"Using device: {cfg.experiment.accelerator}")

    if not cfg.experiment.disable_wandb:
        wandb.login()
        output_dir = Path(cfg.data.output_directory)
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created directory for storing results: {cfg.data.output_directory}")
        cache_dir = Path(f"{cfg.data.output_directory}/.cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        os.environ["WANDB_CACHE_DIR"] = f"{cfg.data.output_directory}/.cache"
        os.environ["WANDB_MODE"] = "offline" if cfg.experiment.disable_wandb else "online"

        logger = WandbLogger(
            # WandbLogger params
            name=cfg.experiment.name,
            project=cfg.experiment.project,
            dir=cfg.data.output_directory,
            log_model="all",
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

    

    match cfg.experiment.task:
        case "pretrain":
            from scripts.pretrain import Pretrainer

            pretrainer = Pretrainer(cfg, logger=logger, profiler=profiler)
            pretrainer.run()
        case "finetune":
            from scripts.finetune import Finetuner

            finetuner = Finetuner(cfg, logger=logger)
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
