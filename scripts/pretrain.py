# Main pretraining and evaluation script for SDO-FM

import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb

from sdofm import utils
from sdofm.datasets import SDOMLDataModule
from sdofm.pretraining import MAE, NVAE, SAMAE


class Pretrainer(object):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.data_module = None
        self.model = None

        match cfg.experiment.model:
            case "mae":
                self.data_module = SDOMLDataModule(
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
                self.data_module.setup()

                self.model = MAE(
                    **cfg.model.mae,
                    optimiser=cfg.model.opt.optimiser,
                    lr=cfg.model.opt.learning_rate,
                    weight_decay=cfg.model.opt.weight_decay,
                )
            case "samae":
                data_module = SDOMLDataModule(
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
                        cfg.data.sdoml.base_directory,
                        cfg.data.sdoml.sub_directory.cache,
                    ),
                )
                data_module.setup()
                model = SAMAE(
                    **cfg.model.mae,
                    **cfg.model.samae,
                    optimiser=cfg.model.opt.optimiser,
                    lr=cfg.model.opt.learning_rate,
                    weight_decay=cfg.model.opt.weight_decay,
                )

            case "nvae":
                self.data_module = SDOMLDataModule(
                    hmi_path=os.path.join(
                        self.cfg.data.sdoml.base_directory,
                        self.cfg.data.sdoml.sub_directory.hmi,
                    ),
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

                self.model = NVAE(
                    **cfg.model.nvae,
                    optimiser=cfg.model.opt.optimiser,
                    lr=cfg.model.opt.learning_rate,
                    weight_decay=cfg.model.opt.weight_decay,
                    hmi_mask=self.data_module.hmi_mask,
                )
            case _:
                raise NotImplementedError(
                    f"Model {cfg.experiment.model} not implemented"
                )

    def run(self):
        print("\nPRE-TRAINING\n")

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
