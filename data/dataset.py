import cv2
import h5py
import random
from pathlib import Path
import numpy as np
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


class BuildingDataset(Dataset):
    def __init__(self, h5_path, transform=None, samples_per_epoch=None):
        self.h5_file = h5_path
        self.file = None
        self.transform = transform
        self.samples_per_epoch = samples_per_epoch

        with h5py.File(self.h5_file, "r") as f:
            self.real_length = len(f["images"])

    def __len__(self):
        if self.samples_per_epoch is None:
            return self.real_length
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_file, "r")

        # If samples_per_epoch is set → random sampling
        if self.samples_per_epoch is not None:
            idx = random.randint(0, self.real_length - 1)

        image = self.file["images"][idx]
        mask = self.file["masks"][idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # mask to 0/1
        mask = (mask > 0).float().unsqueeze(0)

        return image, mask

class TiledDataset(Dataset):
    def __init__(self, h5_path, patch_size=256, stride=None, transform=None):
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size//2
        self.transform = transform
        self.file = None

        # build index of all patches upfront
        self.patch_index = []  # (image_idx, y, x, pad_bottom, pad_right)

        with h5py.File(h5_path, 'r') as f:
            self.n_images = len(f['images'])
            h, w = f['images'].shape[1], f['images'].shape[2]

        self.image_h = h
        self.image_w = w
        self._build_index()

    def _build_index(self):
        p = self.patch_size
        s = self.stride
        h, w = self.image_h, self.image_w

        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p

        padded_h = h + pad_h
        padded_w = w + pad_w

        for img_idx in range(self.n_images):
            for y in range(0, padded_h - p + 1, s):
                for x in range(0, padded_w - p + 1, s):
                    self.patch_index.append((img_idx, y, x, pad_h, pad_w))

    def _pad_image(self, image, pad_h, pad_w):
        # image: H x W x 3 numpy
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        return image

    def _pad_mask(self, mask, pad_h, pad_w):
        # mask: H x W numpy
        if pad_h > 0 or pad_w > 0:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
        return mask

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        img_idx, y, x, pad_h, pad_w = self.patch_index[idx]
        p = self.patch_size

        image = self.file['images'][img_idx]  # H x W x 3
        mask = self.file['masks'][img_idx]    # H x W

        image = self._pad_image(image, pad_h, pad_w)
        mask = self._pad_mask(mask, pad_h, pad_w)

        image_patch = image[y:y+p, x:x+p]
        mask_patch = mask[y:y+p, x:x+p]

        if self.transform:
            transformed = self.transform(image=image_patch, mask=mask_patch)
            image_patch = transformed['image']
            mask_patch = transformed['mask']

        mask_patch = (mask_patch > 127).float()

        return image_patch, mask_patch, img_idx, y, x, pad_h, pad_w
    
    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
