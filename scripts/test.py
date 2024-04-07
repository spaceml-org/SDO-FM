import wandb

settings = wandb.Settings()

with wandb.init(settings=settings) as run:
    run.log({"hello": "world"})
