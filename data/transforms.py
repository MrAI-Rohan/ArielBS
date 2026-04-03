import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(data_cfg):
    transform_cfg = data_cfg.get("transforms", None)
    if transform_cfg in [None, False]:
        return A.Compose([
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),

            ToTensorV2()
        ])

    transform_map = {
        "hflip": A.HorizontalFlip,
        "vflip": A.VerticalFlip,
        "rotate90": A.RandomRotate90,
        "brightness_contrast": A.RandomBrightnessContrast,
        "gauss_noise": A.GaussNoise,
        "blur": A.Blur,
        "elastic": A.ElasticTransform,
        "grid_distortion": A.GridDistortion,
        "shift_scale_rotate": A.ShiftScaleRotate,
        "resize": A.Resize,
    }

    transforms = []

    for name, prob in transform_cfg.items():
        if name == "resize" and prob:
            sz = data_cfg["patch_size"]
            transforms.append(transform_map[name](height=sz, width=sz))
        if name in transform_map and isinstance(prob, (int, float)) and prob > 0:
            transforms.append(transform_map[name](p=prob))

    # Always normalize + tensor
    transforms.append(
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    )

    transforms.append(ToTensorV2())

    return A.Compose(transforms)

