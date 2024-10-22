# Main pretraining and evaluation script for SDO-FM

import glob
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import wandb
from lightning.fabric.strategies import XLAFSDPStrategy

from sdofm import utils
from sdofm.datasets import (
    BrightSpotsSDOMLDataModule,
    HelioProjectedSDOMLDataModule,
    SDOMLDataModule,
)
from sdofm.pretraining import MAE, NVAE, SAMAE, BrightSpots


class Pretrainer(object):
    def __init__(self, cfg, logger=None, profiler=None, is_backbone=False):
        self.cfg = cfg
        self.logger = logger  # would be wandb but broken
        self.profiler = profiler  # if profiler is not None else Profiler()
        self.data_module = None
        self.model = None
        self.model_class = None
        self.data_module_class = SDOMLDataModule

        model_name = (
            cfg.experiment.model if not is_backbone else cfg.experiment.backbone.model
        )

        if cfg.data.sdoml.feature_engineering.enabled:
            match cfg.data.sdoml.feature_engineering.dclass:
                case "BrightSpots":
                    self.data_module_class = BrightSpotsSDOMLDataModule
                case "HelioProjected":
                    self.data_module_class = HelioProjectedSDOMLDataModule
        print(f"Using {self.data_module_class} Data Class")

        match model_name:
            case "mae":
                self.model_class = MAE

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
                    min_date=cfg.data.min_date,
                    max_date=cfg.data.max_date,
                    num_frames=cfg.model.mae.num_frames,
                    drop_frame_dim=cfg.data.drop_frame_dim,
                )
                self.data_module.setup()

                if cfg.experiment.resuming or is_backbone:
                    self.model = self.load_checkpoint(
                        cfg.experiment.checkpoint
                        if not is_backbone
                        else cfg.experiment.backbone.checkpoint
                    )
                else:
                    self.model = self.model_class(
                        **cfg.model.mae,
                        optimiser=cfg.model.opt.optimiser,
                        lr=cfg.model.opt.learning_rate,
                        weight_decay=cfg.model.opt.weight_decay,
                    )
            case "samae":
                self.model_class = SAMAE
                self.data_module = self.data_module_class(
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
                    min_date=cfg.data.min_date,
                    max_date=cfg.data.max_date,
                    num_frames=cfg.model.mae.num_frames,
                    drop_frame_dim=cfg.data.drop_frame_dim,
                )
                self.data_module.setup()

                if cfg.experiment.resuming or is_backbone:
                    self.model = self.load_checkpoint(
                        cfg.experiment.checkpoint
                        if not is_backbone
                        else cfg.experiment.backbone.checkpoint
                    )
                else:
                    self.model = self.model_class(
                        **cfg.model.mae,
                        **cfg.model.samae,
                        optimiser=cfg.model.opt.optimiser,
                        lr=cfg.model.opt.learning_rate,
                        weight_decay=cfg.model.opt.weight_decay,
                    )

            case "brightspots":
                self.model_class = BrightSpots
                self.data_module = BrightSpotsSDOMLDataModule(
                    hmi_path=os.path.join(
                        cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.hmi
                    ),
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
                    blosc_cache=None,  # "/home/walsh/blosc_cache", # disable for TPUs
                    val_months=cfg.data.month_splits.val,
                    test_months=cfg.data.month_splits.test,
                    holdout_months=cfg.data.month_splits.holdout,
                    cache_dir=os.path.join(
                        cfg.data.sdoml.base_directory,
                        cfg.data.sdoml.sub_directory.cache,
                    ),
                    min_date=cfg.data.min_date,
                    max_date=cfg.data.max_date,
                    num_frames=cfg.data.num_frames,
                )
                self.data_module.setup()

                if cfg.experiment.resuming or is_backbone:
                    self.model = self.load_checkpoint(
                        cfg.experiment.checkpoint
                        if not is_backbone
                        else cfg.experiment.backbone.checkpoint
                    )
                else:
                    self.model = self.model_class(
                        **cfg.model.brightspots,
                        optimiser=cfg.model.opt.optimiser,
                        lr=cfg.model.opt.learning_rate,
                        weight_decay=cfg.model.opt.weight_decay,
                    )

            case "nvae":
                self.model_class = NVAE

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
                    min_date=cfg.data.min_date,
                    max_date=cfg.data.max_date,
                    num_frames=cfg.data.num_frames,
                    drop_frame_dim=cfg.data.num_frames,
                )

                if cfg.experiment.resuming or is_backbone:
                    self.model = self.load_checkpoint(
                        cfg.experiment.checkpoint
                        if not is_backbone
                        else cfg.experiment.backbone.checkpoint
                    )
                else:
                    self.model = self.model_class(
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

    def load_checkpoint(self, checkpoint_reference):
        print("Loading checkpoint...")
        self.model = self.model_class.load_from_checkpoint(checkpoint_reference)
        print("Done")
        return self.model

    def run(self):
        print("\nPRE-TRAINING\n")

        if self.cfg.experiment.distributed.enabled:
            trainer = pl.Trainer(
                devices=self.cfg.experiment.distributed.world_size,
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.opt.epochs,
                precision=self.cfg.experiment.precision,
                profiler=self.profiler,
                logger=self.logger,
                enable_checkpointing=True,
                log_every_n_steps=self.cfg.experiment.log_every_n_steps,
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

    def test(self):
        self.trainer.test(ckpt_path="best")
