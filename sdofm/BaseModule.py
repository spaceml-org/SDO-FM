import lighting.pytorch as pl
import torch


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        optimiser: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        # pass to pl.LightningModule
    ):
        super().__init__()

        # optimiser values
        self.optimiser = optimiser
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        match (self.optimiser):
            case "adam":
                optimiser = torch.optim.Adam(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case "sgd":
                optimiser = torch.optim.SGD(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case "adamw":
                optimiser = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case _:
                raise NameError(f"Unknown optimizer {optimiser}")
        return optimiser
