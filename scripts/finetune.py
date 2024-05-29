# Main pretraining and evaluation script for SDO-FM

import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import wandb

from sdofm import utils
from sdofm.datasets import DegradedSDOMLDataModule
from sdofm.finetuning import Autocalibration
from .pretrain import Pretrainer


class Finetuner(object):
    def __init__(self, cfg, logger=None, profiler=None):
        self.cfg = cfg
        self.logger = logger
        self.profiler = profiler
        self.trainer = None
        self.data_module = None
        self.model = None
        self.model_class = None

        backbone = Pretrainer(cfg, logger=logger, is_backbone=True)

        match cfg.experiment.model:
            case "autocalibration":
                self.model_class = Autocalibration

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
                    num_frames=cfg.data.num_frames,
                )
                self.data_module.setup()

                if cfg.experiment.resuming:
                    self.model = self.load_checkpoint(cfg.experiment.checkpoint)
                else:
                    self.model = self.model_class(
                        # **self.cfg.model.mae,
                        img_size=512,
                        patch_size=16,
                        embed_dim=128,
                        **self.cfg.model.autocalibration,
                        optimiser=self.cfg.model.opt.optimiser,
                        lr=self.cfg.model.opt.learning_rate,
                        weight_decay=self.cfg.model.opt.weight_decay,
                        backbone=backbone.model,
                        hyperparam_ignore=['backbone']
                    )
            case _:
                raise NotImplementedError(
                    f"Model {cfg.experiment.model} not implemented"
                )

                    
    def load_checkpoint(self, checkpoint_reference):
        print("Loading checkpoint...")
        if isinstance(self.logger, pl.loggers.wandb.WandbLogger):

            # download checkpoint
            try:
                artifact = self.logger.use_artifact(checkpoint_reference)#, type="model")
                artifact_dir = Path(artifact.download())  / "model.ckpt"
            except wandb.errors.CommError:
                print("W&B checkpoint not found, trying as direct path...")
                artifact_dir = checkpoint_reference

            # load checkpoint
            self.model = self.model_class.load_from_checkpoint(artifact_dir)
            print("Checkpoint loaded from", artifact_dir)
            return self.model
        else:
            raise NotImplementedError("Loading checkpoints without W&B run reference or ckpt path is not supported.")

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
                strategy='ddp',
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
