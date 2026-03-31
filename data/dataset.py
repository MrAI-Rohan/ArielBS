import cv2
import h5py
import random
from pathlib import Path
from torch.utils.data import Dataset

from .data_utils import sample_patch


class WHUDataset(Dataset):
    def __init__(self, h5_path,
                 transform=None,
                 patch_size=256,
                 samples_per_epoch=6000,):

        self.h5_file = h5_path
        self.file = None
        self.transform = transform
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_file, "r")

        # pick random image
        i = random.randint(0, len(self.file["images"]) - 1)

        image = self.file["images"][i]
        mask = self.file["masks"][i]

        # sample patch
        image, mask = sample_patch(image, mask, patch_size=self.patch_size)

        # augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # ensure mask is 0/1
        mask = (mask > 0).float().unsqueeze(0)

        return image, mask
    

class WHUValDataset(Dataset):
    def __init__(self, h5_path, transform=None,):
        self.h5_file = h5_path
        self.file = None
        self.transform = transform
        
        with h5py.File(self.h5_file, "r") as f:
            self.length = len(f["images"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_file, "r")

        image = self.file["images"][idx]
        mask = self.file["masks"][idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # mask to 0/1
        mask = (mask > 0).float().unsqueeze(0)

        return image, mask
