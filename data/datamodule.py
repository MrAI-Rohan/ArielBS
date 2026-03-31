from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import WHUDataset, WHUValDataset
from .transforms import build_transforms

class WHUDataModule(pl.LightningDataModule):
    def __init__(self, config, train_h5, val_h5):
        super().__init__()
        self.config = config
        self.train_h5 = train_h5
        self.val_h5 = val_h5

        train_transform_cfg = config["data"]["train_transform"]
        valid_transform_cfg = config["data"]["val_transform"]
        self.train_transform = build_transforms(train_transform_cfg)
        self.val_transform = build_transforms(valid_transform_cfg)

        self.num_workers = self.config.get("num_workers", self.get_num_workers(config))

    def setup(self, stage=None):
        self.train_dataset = WHUDataset(h5_path=self.train_h5,
                                        transform=self.train_transform,
                                        patch_size=self.config["data"]["patch_size"],
                                        samples_per_epoch=self.config["data"]["samples_per_epoch"]
                                        )
        
        self.val_dataset = WHUValDataset(h5_path=self.val_h5,
                                      transform=self.val_transform,
                                      )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["train_batch_size"],
            num_workers=self.num_workers,
            shuffle=self.config["data"]["shuffle"],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["val_batch_size"],
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    def get_num_workers(self, config):
        if config["platform"] == "colab":
            return 2
