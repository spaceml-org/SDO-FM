# Assortment of variously useful functions

import collections
import torch
import wandb
import os
import shutil

def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return "{0}d:{1:02}:{2:02}:{3:02}".format(int(d), int(h), int(m), int(s))


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def load_checkpoint(model, optimizer, scheduler, device, checkpoint_file: str):
    """Loads a model checkpoint.
    Params:
    - model (nn.Module): initialised model
    - optimizer (nn.optim): initialised optimizer
    - scheduler (nn.optim.lr_scheduler): initialised scheduler
    - device (torch.device): device model is on
    Returns:
    - model with loaded state dict
    - optimizer with loaded state dict
    - scheduler with loaded state dict
    - epoch (int): epoch checkpoint was saved at
    - best_loss (float): best loss seen so far
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    print(
        f"Loaded {checkpoint_file}, "
        f"trained to epoch {checkpoint['epoch']} with best loss {checkpoint['best_loss']}"
    )

    return model, optimizer, scheduler, checkpoint["epoch"], checkpoint["best_loss"]


def save_checkpoint(checkpoint_dict: dict, is_best: bool):
    """Saves a model checkpoint to file. Keeps most recent and best model.
    Params:
    - checkpoint_dict (dict): dict containing all model state info, to pickle
    - is_best (bool): whether this checkpoint is the best seen so far.
    """
    # files for checkpoints
    # TODO implement scratch dir in run script
    scratch_dir = os.getenv(
        "SCRATCH_DIR", wandb.run.dir
    )  # if given a scratch dir save models here
    checkpoint_file = os.path.join(scratch_dir, "ckpt.pth.tar")
    best_file = os.path.join(scratch_dir, "best.pth.tar")
    torch.save(checkpoint_dict, checkpoint_file)
    wandb.save(checkpoint_file, policy="live")  # save to wandb
    print(f"Saved checkpoint to {checkpoint_file}")

    if is_best:
        shutil.copyfile(checkpoint_file, best_file)
        print(f"Saved best checkpoint to {best_file}")
        wandb.save(best_file, policy="live")  # save to wandb

def wandb_restore_checkpoint(cfg):
    model_path = (
        f"{cfg.data.output_directory}/checkpoints/{cfg.experiment.checkpoint}"
    )
    os.makedirs(model_path, exist_ok=True)
    # restore from wandb
    wandb_best_file = wandb.restore(
        "best.pth.tar",
        run_path=f"{cfg.experiment.wandb_entity}/{cfg.experiment.name}/{cfg.experiment.checkpoint}",
        root=model_path,
    )
    # load state dict
    (
        model,
        optimizer,
        scheduler,
        best_epoch,
        best_loss,
    ) = load_checkpoint(
        model, optimizer, scheduler, cfg.experiment.device, wandb_best_file.name
    )
    print(
        f"Loaded checkpoint from {wandb_best_file.name} with loss {best_loss} at epoch {best_epoch}"
    )
    return model, optimizer, scheduler