import os
import yaml
import shutil
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import convert_numerics
from training_module import SegmentationModule
from data.datamodule import WHUDataModule

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    # Recursively convert string numeric values to floats
    return convert_numerics(config)


def main(params=None):
    if params is not None:
        # Use provided params dict as args, setting defaults for optional args
        params = params.copy()
        if 'resume' not in params:
            params['resume'] = None
        args = argparse.Namespace(**params)
    else:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True, help="Path to config file")
        parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint root directory")
        parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
        parser.add_argument("--train_h5", required=True, help="Path to training H5 file")
        parser.add_argument("--val_h5", required=True, help="Path to validation H5 file")

        args = parser.parse_args()

    config = load_config(args.config)

    experiment_name = config["experiment_name"]
    checkpoint_dir = os.path.join(args.checkpoint_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(checkpoint_dir, "config.yaml"))

    pl.seed_everything(config["seed"], workers=True)

    # WandB logger
    wandb_logger = WandbLogger(
        project=config["logging"]["project"],
        name=experiment_name,
        id=experiment_name,
        resume="allow",
        log_model=False
    )

    wandb_logger.experiment.config.update(config, allow_val_change=True)

    # BEST checkpoint
    best_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{experiment_name}_best",
        monitor=config["scheduler"]["monitor"],
        mode="max",
        save_top_k=1
    )

    # LAST checkpoint
    last_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{experiment_name}_last",
        every_n_epochs=1,
        save_top_k=1
    )

    # Early stopping
    callbacks = [best_checkpoint, last_checkpoint]
    if "early_stopping" in config:
        early_stopping = EarlyStopping(
            monitor=config["early_stopping"]["monitor"],
            patience=config["early_stopping"]["patience"],
            mode=config["early_stopping"]["mode"]
        )
        callbacks.append(early_stopping)

    model = SegmentationModule(config)
    datamodule = WHUDataModule(config, args.train_h5, args.val_h5)

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu",
        devices=1,
        precision=config["precision"],
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=config["training"]["progress_bar"],
        gradient_clip_val=config["training"].get("gradient_clip_val", 1.0)
    )

    trainer.fit(model, datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()

