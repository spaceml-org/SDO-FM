# Main pretraining and evaluation script for SDO-FM

import os
from pathlib import Path

import torch
import wandb
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from sdofm import utils
from sdofm.models.prithvi import PrithviEncoderDecoder, evaluate, train


def init_model(cfg, karman_input_dim, output_dim=1):
    """Get model from cfg
    TODO shift to scripts (anything with cfg)
    TODO maybe seperate this out. Add other models
    """
    # TODO configs seperate for karman and sdoml - e.g. dropout etc.
    return PrithviEncoderDecoder(
        num_classes=cfg.prithvi.num_classes,
        freeze_encoder=cfg.prithvi.freeze_encoder,
        num_neck_filters=cfg.prithvi.num_neck_filters,
        **cfg.prithvi.mae,
    )


def init_optim(cfg, model):
    """Initialize optimizer, scheduler and loss function
    Params:
    - args (argparse.Namespace): parsed command line arguments
    - model (nn.Module): initialised model
    Returns:
    - optimizer (nn.optim): initialised optimizer
    - criterion: initialised loss function
    """
    if cfg.model.opt.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            cfg.model.opt.learning_rate,
            weight_decay=cfg.model.opt.weight_decay,
        )
    elif cfg.model.opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            cfg.model.opt.learning_rate,
            weight_decay=cfg.model.opt.weight_decay,
        )
    elif cfg.model.opt.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            cfg.model.opt.learning_rate,
            weight_decay=cfg.model.opt.weight_decay,
        )
    else:
        raise NameError(f"Unknown optimizer {cfg.optimizer}")

    if cfg.model.opt.loss == "mse":
        criterion = torch.nn.MSELoss()
    elif cfg.model.opt.loss == "mae":
        criterion = torch.nn.L1Loss()
    elif cfg.model.opt.loss == "mape":
        criterion = metrics.mape_loss
    else:
        raise NameError(f"Unknown loss function {cfg.loss}")

    if cfg.model.opt.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.model.opt.scheduler_warmup
        )
    elif cfg.model.opt.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20, verbose=True
        )
    elif cfg.model.opt.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.model.opt.scheduler_warmup,
            num_training_steps=cfg.model.opt.epochs,
        )
    elif cfg.model.opt.scheduler == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        raise NameError(f"Unknown scheduler {cfg.model.opt.scheduler}")

    return optimizer, scheduler, criterion


def pretrain_sdofm(
    cfg, model, optimizer, scheduler, criterion, train_loader, validation_loader
):
    print("\nPRE-TRAINING\n")
    print(">> Loading dataset")
    # Wandb info in other file
    dataset = None  # in other file

    model = init_model(cfg, karman_input_dim=dataset.input_dim)
    model.to(torch.float64 if cfg.experiment.precision == 64 else torch.float32).to(
        cfg.experiment.device
    )
    wandb.watch(model, log="all")  # log gradients and parameters
    optimizer, scheduler, criterion = init_optim(cfg, model)
    num_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )  # number of model parameters

    if cfg.experiment.checkpoint:
        model, optimizer, scheduler = utils.wandb_restore_checkpoint()

    model, final_epoch = train(
        model, optimizer, scheduler, criterion, train_loader, validation_loader
    )


def evaluate_sdofm(
    cfg, model, optimizer, scheduler, criterion, train_loader, validation_loader
):
    with torch.no_grad():
        print("\nEVALUATION\n")
        val_loss, val_mse, val_mae, val_mape, val_msa, val_results = evaluate(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=validation_loader,
            device=cfg.experiment.device,
            task="test",
        )
