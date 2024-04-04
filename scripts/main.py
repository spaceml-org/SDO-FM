"""
Main entry point. Uses Hydra to load config files and override defaults with command line args
"""

import os
import random
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sdofm.utils import days_hours_mins_secs_str


# loads the config file
@hydra.main(config_path="../experiments", config_name="default")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    random.seed(cfg.experiment.seed)

    # set device using config disable_cuda option and torch.cuda.is_available()
    cfg.experiment.device = (
        "cuda"
        if (not cfg.experiment.disable_cuda) and torch.cuda.is_available()
        else "cpu"
    )

    # run experiment
    print(f"\nRunning with config:")
    print(OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False))
    print("\n")

    print(f"Using device: {cfg.experiment.device}")

    if cfg.experiment.project == "sdofm":
        from scripts.pretrain import pretrain_sdofm

        pretrain_sdofm(cfg)
    else:
        raise NotImplementedError(
            f"Experiment {cfg.experiment.project} not implemented"
        )


if __name__ == "__main__":
    time_start = time.time()
    # errors
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Produce a complete stack trace
    main()
    print(
        "\nTotal duration: {}".format(
            days_hours_mins_secs_str(time.time() - time_start)
        )
    )
