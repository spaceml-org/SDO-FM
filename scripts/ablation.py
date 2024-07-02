# Main pretraining and evaluation script for SDO-FM

import os
from pathlib import Path

import lightning.pytorch as pl
import torch
import wandb

from sdofm import utils
from sdofm.datasets import SDOMLDataModule, DegradedSDOMLDataModule
from sdofm.ablation import AblationAutocalibration
from sdofm.models import HybridIrradianceModel


class Ablation(object):
    def __init__(self, cfg, logger=None, profiler=None):
        self.cfg = cfg
        self.logger = logger
        self.profiler = profiler
        self.trainer = None
        self.data_module = None
        self.model = None
        self.model_class = None
        self.callbacks = []
        self.trainer = None

        match cfg.experiment.model:
            case "autocalibration":
                self.model_class = AblationAutocalibration

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
                        cfg.data.sdoml.base_directory,
                        cfg.data.sdoml.sub_directory.cache,
                    ),
                    min_date=cfg.data.min_date,
                    max_date=cfg.data.max_date,
                    num_frames=cfg.data.num_frames,
                    drop_frame_dim=cfg.data.drop_frame_dim,
                )
                self.data_module.setup()

                if cfg.experiment.resuming:
                    self.model = self.load_checkpoint(cfg.experiment.checkpoint)
                else:
                    self.model = self.model_class(
                        # **self.cfg.model.mae,
                        img_size=512,
                        channels=9,
                        # **self.cfg.model.autocalibration,
                        output_dim=self.cfg.model.autocalibration.output_dim,
                        loss=self.cfg.model.autocalibration.loss,
                        # general
                        optimiser=self.cfg.model.opt.optimiser,
                        lr=self.cfg.model.opt.learning_rate,
                        weight_decay=self.cfg.model.opt.weight_decay,
                    )
            case "virtualeve":
                import numpy as np

                self.model_class = HybridIrradianceModel

                self.data_module = SDOMLDataModule(
                    hmi_path=(
                        os.path.join(
                            cfg.data.sdoml.base_directory,
                            cfg.data.sdoml.sub_directory.hmi,
                        )
                        if cfg.data.sdoml.sub_directory.hmi
                        else None
                    ),
                    aia_path=(
                        os.path.join(
                            cfg.data.sdoml.base_directory,
                            cfg.data.sdoml.sub_directory.aia,
                        )
                        if cfg.data.sdoml.sub_directory.aia
                        else None
                    ),
                    eve_path=os.path.join(
                        cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.eve
                    ),
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
                    num_frames=cfg.data.num_frames,
                    drop_frame_dim=cfg.data.drop_frame_dim,
                )
                self.data_module.setup()

                if cfg.experiment.resuming:
                    self.model = self.load_checkpoint(cfg.experiment.checkpoint)
                else:
                    d_input = (
                        len(self.data_module.wavelengths)
                        if self.data_module.wavelengths
                        else 0
                    )
                    d_input += (
                        len(self.data_module.components)
                        if self.data_module.components
                        else 0
                    )
                    d_output = len(self.data_module.ions)

                    self.model = self.model_class(
                        # virtual eve
                        d_input=d_input,
                        d_output=d_output,
                        eve_norm=np.array(
                            self.data_module.normalizations["EVE"]["eve_norm"],
                            dtype=np.float32,
                        ),
                        **self.cfg.model.virtualeve,
                        # general
                        optimiser=self.cfg.model.opt.optimiser,
                        lr=self.cfg.model.opt.learning_rate,
                        weight_decay=self.cfg.model.opt.weight_decay,
                    )

                    self.model.set_train_mode("linear")
                    # add switch mode callback
                    self.callbacks.append(
                        pl.callbacks.LambdaCallback(
                            on_train_epoch_start=(
                                lambda trainer, pl_module: (
                                    self.model.set_train_mode("cnn")
                                    if self.trainer.current_epoch
                                    > self.cfg.model.virtualeve.epochs_linear
                                    else None
                                )
                            )
                        )
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
                artifact = self.logger.use_artifact(
                    checkpoint_reference
                )  # , type="model")
                artifact_dir = Path(artifact.download()) / "model.ckpt"
            except wandb.errors.CommError:
                print("W&B checkpoint not found, trying as direct path...")
                artifact_dir = checkpoint_reference

            # load checkpoint
            self.model = self.model_class.load_from_checkpoint(artifact_dir)
            print("Checkpoint loaded from", artifact_dir)
            return self.model
        else:
            raise NotImplementedError(
                "Loading checkpoints without W&B run reference or ckpt path is not supported."
            )

    def run(self):
        print("\nRUNNING MODEL FOR ABLATION STUDY...\n")

        if self.cfg.experiment.distributed:
            self.trainer = pl.Trainer(
                devices=self.cfg.experiment.distributed.world_size,
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.opt.epochs,
                precision=self.cfg.experiment.precision,
                profiler=self.profiler,
                logger=self.logger,
                enable_checkpointing=True,
                callbacks=self.callbacks,
                strategy=self.cfg.experiment.distributed.strategy,
                log_every_n_steps=10,
            )
        else:
            self.trainer = pl.Trainer(
                accelerator=self.cfg.experiment.accelerator,
                max_epochs=self.cfg.model.opt.epochs,
                logger=self.logger,
            )
        self.trainer.fit(model=self.model, datamodule=self.data_module)
        return self.trainer

    def evaluate(self):
        self.trainer.evaluate()

    def test_sdofm(self):
        self.trainer.test(ckpt_path="best")
