import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def random_crop(image, mask, patch_size=256):
    h, w = image.shape[:2]

    x = np.random.randint(0, w - patch_size + 1)
    y = np.random.randint(0, h - patch_size + 1)

    image_patch = image[y:y+patch_size, x:x+patch_size]
    mask_patch = mask[y:y+patch_size, x:x+patch_size]

    return image_patch, mask_patch


def sample_patch(image, mask, patch_size=256,
                 building_threshold=0.005,
                 random_prob=0.3,
                 max_trials=10):

    # 30% random patch
    if np.random.rand() < random_prob:
        return random_crop(image, mask, patch_size)

    # 70% building patches
    for _ in range(max_trials):
        img_patch, mask_patch = random_crop(image, mask, patch_size)

        building_pixels = (mask_patch > 0).sum()
        total_pixels = mask_patch.size
        pct = building_pixels / total_pixels

        if pct > building_threshold:
            return img_patch, mask_patch

    # fallback if no building patch found
    return random_crop(image, mask, patch_size)

