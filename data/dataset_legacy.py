import cv2
import random
from pathlib import Path
from torch.utils.data import Dataset

from .data_utils import sample_patch


class WHUDataset(Dataset):
    def __init__(self, image_dir, mask_dir,
                 transform=None,
                 patch_size=256,
                 samples_per_epoch=6000,):

        self.image_paths = sorted(Path(image_dir).glob("*"))
        self.mask_paths = sorted(Path(mask_dir).glob("*"))
        self.transform = transform
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # pick random image
        i = random.randint(0, len(self.image_paths) - 1)
        
        image = cv2.imread(str(self.image_paths[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_paths[i]), 0)

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
    def __init__(self, image_dir, mask_dir, transform=None,):
        self.image_paths = sorted(Path(image_dir).glob("*"))
        self.mask_paths = sorted(Path(mask_dir).glob("*"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_paths[idx]), 0)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # mask to 0/1
        mask = (mask > 0).float().unsqueeze(0)

        return image, mask
