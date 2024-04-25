# Main pretraining and evaluation script for SDO-FM

import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import wandb

from sdofm import utils
from sdofm.datasets import DimmedSDOMLDataModule
from sdofm.finetuning import Autocalibration


class Finetuner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.trainer = None
        self.data_module = None
        self.model = None

        match cfg.experiment.model:
            case "autocalibration":
                self.data_module = DimmedSDOMLDataModule(
                    hmi_path=None,
                    aia_path=os.path.join(
                        self.cfg.data.sdoml.base_directory,
                        self.cfg.data.sdoml.sub_directory.aia,
                    ),
                    eve_path=None,
                    components=self.cfg.data.sdoml.components,
                    wavelengths=self.cfg.data.sdoml.wavelengths,
                    ions=self.cfg.data.sdoml.ions,
                    frequency=self.cfg.data.sdoml.frequency,
                    batch_size=self.cfg.model.opt.batch_size,
                    num_workers=self.cfg.data.num_workers,
                    val_months=self.cfg.data.month_splits.val,
                    test_months=self.cfg.data.month_splits.test,
                    holdout_months=self.cfg.data.month_splits.holdout,
                    cache_dir=os.path.join(
                        self.cfg.data.sdoml.base_directory,
                        self.cfg.data.sdoml.sub_directory.cache,
                    ),
                )
                self.data_module.setup()

                self.model = Autocalibration(
                    **self.cfg.model.mae,
                    optimiser=self.cfg.model.opt.optimiser,
                    lr=self.cfg.model.opt.learning_rate,
                    weight_decay=self.cfg.model.opt.weight_decay,
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
                logger=self.logger,
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
