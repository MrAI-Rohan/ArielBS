import h5py
from torch.utils.data import ConcatDataset

from .transforms import build_transforms
from .dataset import WHUDataset, WHUValDataset, BuildingDataset

def build_dataset(data_cfg, train_h5, val_h5,):
    dataset_name = data_cfg["dataset"]

    train_transform = build_transforms(data_cfg, mode="train")
    val_transform = build_transforms(data_cfg, mode="val")

    samples_per_epoch = data_cfg.get("samples_per_epoch", None)

    if dataset_name == "whu":
        train_dataset = WHUDataset(h5_path=train_h5,
                                    transform=train_transform,
                                    patch_size=data_cfg["patch_size"],
                                    samples_per_epoch=data_cfg["data"]["samples_per_epoch"]
                                    )
        
        val_dataset = WHUValDataset(h5_path=val_h5,
                                      transform=val_transform,
                                      )
        
    elif dataset_name == "reproduction":
        if not samples_per_epoch:
            s1 = None
            s2 = None
        else:
            with h5py.File(train_h5+"/re1_train.h5", "r") as f:
                len1 = len(f["images"])
            with h5py.File(train_h5+"/re2_train.h5", "r") as f:
                len2 = len(f["images"])
            total = len1 + len2
            s1 = int(samples_per_epoch * len1 / total)
            s2 = samples_per_epoch - s1
        
        train_dataset1 = BuildingDataset(h5_path=train_h5+"/re1_train.h5",
                                         transform=train_transform,
                                         samples_per_epoch=s1)
        train_dataset2 = BuildingDataset(h5_path=train_h5+"/re2_train.h5",
                                         transform=train_transform,
                                         samples_per_epoch=s2)
        
        val_dataset1 = BuildingDataset(h5_path=val_h5+"/re1_val.h5",
                                       transform=val_transform,)
        val_dataset2 = BuildingDataset(h5_path=val_h5+"/re2_val.h5",
                                       transform=val_transform,)
        
        train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        val_dataset = ConcatDataset([val_dataset1, val_dataset2])

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, val_dataset