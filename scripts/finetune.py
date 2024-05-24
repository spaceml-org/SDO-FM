# Main pretraining and evaluation script for SDO-FM

import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import wandb

from sdofm import utils
from sdofm.datasets import DegradedSDOMLDataModule
from sdofm.finetuning import Autocalibration
from sdofm.pretraining import MAE


class Finetuner(object):
    def __init__(self, cfg, logger=None, profiler=None):
        self.cfg = cfg
        self.logger = logger
        self.profiler = profiler
        self.trainer = None
        self.data_module = None
        self.model = None

        match cfg.experiment.model:
            case "autocalibration":
                self.data_module = DegradedSDOMLDataModule(
                    hmi_path=None,
                    aia_path=os.path.join(
                        cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.aia
                    ),
                    eve_path=None,
                    components=cfg.data.sdoml.components,
                    wavelengths=cfg.data.sdoml.wavelengths,
                    ions=cfg.data.sdoml.ions,
                    frequency=cfg.data.sdoml.frequency,
                    batch_size=cfg.model.opt.batch_size,
                    num_workers=cfg.data.num_workers,
                    val_months=cfg.data.month_splits.val,
                    test_months=cfg.data.month_splits.test,
                    holdout_months=cfg.data.month_splits.holdout,
                    cache_dir=os.path.join(
                        cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.cache
                    ),
                    min_date=cfg.data.min_date,
                    max_date=cfg.data.max_date,
                    num_frames=1,
                )
                self.data_module.setup()

                backbone = MAE.load_from_checkpoint(
                    **cfg.model.mae,
                    optimiser=cfg.model.opt.optimiser,
                    lr=cfg.model.opt.learning_rate,
                    weight_decay=cfg.model.opt.weight_decay,
                    checkpoint_path="/home/walsh/SDO-FM/outputs/2024-05-24/18-35-29/sdofm/chvgwzkk/checkpoints/epoch=3-step=176.ckpt"
                )

                self.model = Autocalibration(
                    **self.cfg.model.mae,
                    **self.cfg.model.degragation,
                    optimiser=self.cfg.model.opt.optimiser,
                    lr=self.cfg.model.opt.learning_rate,
                    weight_decay=self.cfg.model.opt.weight_decay,
                    backbone=backbone,
                    hyperparam_ignore=['backbone']
                )
            case _:
                raise NotImplementedError(
                    f"Model {cfg.experiment.model} not implemented"
                )

    def run(self):
        print("\nFINE TUNING\n")

        if self.cfg.experiment.distributed:
            trainer = pl.Trainer(
                devices=self.cfg.experiment.distributed.world_size,
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.opt.epochs,
                precision=self.cfg.experiment.precision,
                profiler=self.profiler,
                logger=self.logger,
                enable_checkpointing=True,
            )
        else:
            trainer = pl.Trainer(
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.opt.epochs,
                logger=self.logger,
            )
        trainer.fit(model=self.model, datamodule=self.data_module)
        return trainer

    def evaluate(self):
        self.trainer.evaluate()

    def test_sdofm(self):
        self.trainer.test(ckpt_path="best")
