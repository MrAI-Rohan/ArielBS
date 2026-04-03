from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset_factory import build_dataset

class WHUDataModule(pl.LightningDataModule):
    def __init__(self, config, train_h5, val_h5):
        super().__init__()
        self.config = config
        self.train_h5 = train_h5
        self.val_h5 = val_h5
        self.data_cfg = config["data"]

        self.num_workers = self.config.get("num_workers", self.get_num_workers(config))

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset = build_dataset(
            self.data_cfg,
            train_h5=self.train_h5,
            val_h5=self.val_h5,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg["train_batch_size"],
            num_workers=self.num_workers,
            shuffle=self.data_cfg["shuffle"],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg["val_batch_size"],
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    def get_num_workers(self, config):
        if config["platform"] == "colab":
            return 2
