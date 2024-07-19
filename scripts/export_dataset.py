import argparse
import codecs
import datetime
import os
import pprint
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import omegaconf
import torch
import webdataset as wds
from tqdm import tqdm

from sdofm.datasets import TimestampedSDOMLDataModule
from sdofm.models import ConvTransformerTokensToEmbeddingNeck
from sdofm.pretraining import MAE, NVAE
from sdofm.utils import days_hours_mins_secs_str


def main():
    # fmt: off
    parser=argparse.ArgumentParser(description="SDO Latent Dataset generation script", formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("--experiment", "-e", help="Training configuration file", default="../experiments/pretrain", type=str, )
    parser.add_argument("--model", "-m", help="Pretrained model filename", required=True, type=str)
    parser.add_argument("--src_id", "-i", help="Source Identify, e.g. W&B run ID", default=None, type=str, )
    parser.add_argument("--out_dir", "-o", help="Output directory to save the dataset", default="", type=str, )
    parser.add_argument("--batch_size", "-b", help="Batch size", default=32, type=int)
    parser.add_argument("--num_workers", "-n", help="Number of workers", default=0, type=int)
    parser.add_argument("--max", help="Maximum number of samples to process", default=None, type=int)
    parser.add_argument("--device", help="Compute device(cpu, cuda:0, cuda:1, etc.)", default="cpu", type=str, )
    opt = parser.parse_args()
    # fmt: on

    print("SDO-FM Dataset Export Script, modified from work by")
    print("NASA FDL-X 2023 Thermospheric Drag Team")
    command_line_arguments = " ".join(sys.argv[1:])
    print("Arguments:\n{}\n".format(command_line_arguments))
    print("Script Config:")
    pprint.pprint(vars(opt), depth=2, width=50)

    # Datamodule
    print("\nLoading Data:")
    cfg = omegaconf.OmegaConf.load(opt.experiment)

    data_module = TimestampedSDOMLDataModule(
        # hmi_path=os.path.join(
        #     cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.hmi
        # ),
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
        val_months=[],  # cfg.data.month_splits.val,
        # fmt: off
        test_months=[1,2,3,4,5,6,7,8,9,10,11,12,],
        # cfg.data.month_splits.test,
        # fmt: on
        holdout_months=[],  # =cfg.data.month_splits.holdout,
        cache_dir=os.path.join(
            cfg.data.sdoml.base_directory, cfg.data.sdoml.sub_directory.cache
        ),
        min_date=cfg.data.min_date,
        max_date=cfg.data.max_date,
        num_frames=cfg.data.num_frames,
        drop_frame_dim=cfg.data.drop_frame_dim,
    )
    data_module.setup()

    print("\nLoading model:")
    try:
        match (cfg.experiment.model):
            case "nvae":
                model = NVAE.load_from_checkpoint(
                    opt.model, map_location=torch.device(opt.device)
                )
                z_dim, z_shapes = model.z_dim, model.z_shapes
                latent_dim = z_dim.shape[0]
            case "mae":
                model = MAE.load_from_checkpoint(
                    opt.model, map_location=torch.device(opt.device)
                )
                latent_dim = cfg.model.mae.embed_dim

                num_tokens = 512 // 16
                emb_decoder = ConvTransformerTokensToEmbeddingNeck(
                    embed_dim=latent_dim,
                    # output_embed_dim=32,
                    output_embed_dim=latent_dim,
                    Hp=num_tokens,
                    Wp=num_tokens,
                    drop_cls_token=True,
                    num_frames=1,
                )
            case _:
                raise ValueError(
                    f"Model export of {cfg.experiment.model} not yet supported."
                )
        print("Model ready.")
    except Exception as e:
        print(f"Could not load model at {opt.model}")
        raise e

    # Begin

    output = f"SDOFM-{datetime.datetime.now().strftime('%Y_%m_%d_%H%M')}-embsize_{latent_dim}-period_{data_module.min_date.strftime('%Y_%m_%d_%H%M')}_{data_module.max_date.strftime('%Y_%m_%d_%H%M')}"
    if opt.src_id:
        output += f"-{opt.src_id}"
    output = Path(opt.out_dir) / (output + ".tar")

    print(f"\nBeginning webdataset creation at {output}")
    count = 0
    latent_dim_checked = False

    sink = wds.TarWriter(f"{output}.tar")

    dl = data_module.test_dataloader()
    pbar = tqdm(total=len(dl))
    for data in dl:
        pbar.update(1)

        x, timestamps = data["image_stack"], data["timestamps"][0]
        batch_size = x.shape[0]

        match (cfg.experiment.model):
            case "nvae":
                z = model.encode(x)
            case "mae":
                x_hat, _, _ = model.autoencoder.forward_encoder(x, mask_ratio=0.0)
                x_hat = x_hat.squeeze(dim=2)
                z = emb_decoder(x_hat)

        z = z.detach().cpu().numpy()
        if not latent_dim_checked:
            if latent_dim != z.shape[1]:
                warnings.warn(
                    "Latent dimension mismatch: {} vs {}".format(latent_dim, z.shape[1])
                )
            latent_dim_checked = True

        for i in range(batch_size):
            sink.write(
                {
                    "__key__": timestamps[i],
                    "emb.pyd": z[i],
                }
            )
            count += 1

        if opt.max is not None and count >= opt.max:
            print("Stopping after {} samples".format(opt.max))
            break

    sink.close()


if __name__ == "__main__":
    time_start = time.time()
    main()
    print(
        "\nTotal duration: {}".format(
            days_hours_mins_secs_str(time.time() - time_start)
        )
    )
    sys.exit(0)
