# Main pretraining and evaluation script for SDO-FM

import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb

from sdofm import utils
from sdofm.datasets import SDOMLDataModule
from sdofm.pretraining import MAE


class Pretrainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.trainer = None

    def run(self):
        print("\nPRE-TRAINING\n")

        data_module = SDOMLDataModule(
            # hmi_path=os.path.join(
            #     self.cfg.data.sdoml.base_directory, self.cfg.data.sdoml.sub_directory.hmi
            # ),
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
        data_module.setup()

        model = MAE(
            **self.cfg.model.prithvi.mae,
            optimiser_str=self.cfg.model.opt.optimiser,
            lr=self.cfg.model.opt.learning_rate,
            weight_decay=self.cfg.model.opt.weight_decay,
        )

        if self.cfg.experiment.distributed:
            trainer = pl.Trainer(
                gpus=self.cfg.experiment.distributed.worldsize,
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.opt.epochs,
                precision=self.cfg.experiment.precision,
            )
        else:
            trainer = pl.Trainer(
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.opt.epochs,
            )
        trainer.fit(model=model, datamodule=data_module)
        return trainer

    def evaluate(self):
        self.trainer.evaluate()

    def test_sdofm(self):
        self.trainer.test(ckpt_path="best")
