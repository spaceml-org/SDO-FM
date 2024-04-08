"""
Main entry point. Uses Hydra to load config files and override defaults with command line args
"""

import os
import random
import time

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from sdofm import utils  # import days_hours_mins_secs_str


# loads the config file
@hydra.main(config_path="../experiments", config_name="default")
def main(cfg: DictConfig) -> None:
    # set seed
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)

    # set device using config disable_cuda option and torch.cuda.is_available()
    cfg.experiment.device = (
        "cuda"
        if (not cfg.experiment.disable_cuda) and torch.cuda.is_available()
        else "cpu"
    )

    # set precision of torch tensors
    if cfg.experiment.device == "cuda":
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

    print(f"Using device: {cfg.experiment.device}")

    match cfg.experiment.task:
        case "pretrain_mae":
            from scripts.pretrain import pretrain_sdofm

            pretrain_sdofm(cfg)
        case "evaluate_mae":
            from scripts.pretrain import evaluate_sdofm

            pass
        case _:
            raise NotImplementedError(
                f"Experiment {cfg.experiment.task} not implemented"
            )


if __name__ == "__main__":
    time_start = time.time()
    # errors
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace
    main()
    print(
        "\nTotal duration: {}".format(
            utils.days_hours_mins_secs_str(time.time() - time_start)
        )
    )
