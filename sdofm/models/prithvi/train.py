from .evaluate import evaluate
from ... import utils
import time

import torch 




def train(cfg, model, optimizer, scheduler, criterion, train_loader, validation_loader):
    # get best val loss
    best_loss, mse_loss, mae_loss, mape_loss, msa_loss, val_results = evaluate(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        data_loader=val_loader,
        device=cfg.experiment.device,
        task="val",
    )

    print(
        f"\ninitial loss: {best_loss}, mse: {mse_loss}, mae: {mae_loss}, mape: {mape_loss}, msa: {msa_loss}"
    )

    # save initial checkpoint
    checkpoint_dict = {
        "epoch": 0,
        "state_dict": model.state_dict(),
        "best_loss": best_loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": vars(cfg),
    }
    utils.save_checkpoint(checkpoint_dict, is_best=True)
    best_epoch = 0
    start_time = time.time()
    try:
        # Training loop
        for epoch in range(1, cfg.model.opt.epochs + 1):
            epoch_start = time.time()
            train_loss, train_mse, train_mae, train_mape = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=train_loader,
                device=cfg.experiment.device,
                task="train",
                epoch=epoch,
                log_n_batches=cfg.experiment.log_n_batches,
            )

            # evaluate on val set
            val_loss, val_mse, val_mae, val_mape, val_msa, val_results = evaluate(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=val_loader,
                device=cfg.experiment.device,
                task="val",
                epoch=epoch,
            )

            # scheduler step
            if cfg.model.opt.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # checkpoint as best if best val loss
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_epoch = epoch
                # save results to file
                if cfg.experiment.save_results:
                    val_results.to_csv(f"{wandb.run.dir}/val_best_results.csv")

            # log step
            to_log = {
                "train/loss": train_loss,
                "train/mse": train_mse,
                "train/mae": train_mae,
                "train/mape": train_mape,
                "val/loss": val_loss,
                "val/mse": val_mse,
                "val/mae": val_mae,
                "val/mape": val_mape,
                "val/msa": val_msa,
                "val/mape_f107_200_inf": metrics.mape(val_results['targets'].values[val_results['solar_activity_bins']=='F10.7: 200 (high)'], val_results['preds'].values[val_results['solar_activity_bins']=='F10.7: 200 (high)'] ),
                "val/results": wandb.Table(
                    dataframe=val_results.sample(
                        min([len(val_results), cfg.experiment.wandb_save_results_n]),
                        random_state=1,
                    )
                ),
                "epoch": epoch,
                "lr": scheduler._last_lr[0],
            }
            wandb.log(to_log)

            # checkpoint as epoch
            checkpoint_dict = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            utils.save_checkpoint(checkpoint_dict, is_best)

            print(
                f"\nTime for Epoch: {time.time() - epoch_start:.2f}s, Total Time: {time.time() - start_time:.2f}s"
                f"\nEpoch {epoch}/{cfg.model.opt.epochs}: \ntrain/loss: {train_loss}, mse: {train_mse}, mae: {train_mae}, mape: {train_mape}"
                f"\nval/loss: {val_loss}, mse: {val_mse}, mae: {val_mae}, mape: {val_mape}, msa: {val_msa}"
            )

            # patience
            if epoch - best_epoch > cfg.model.opt.patience:
                print("\nEarly stopping...")
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt... stopping training")
        pass

    return model, epoch