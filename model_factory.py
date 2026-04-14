import segmentation_models_pytorch as smp


def build_model(config):
    model_cfg = config["model"]
    if model_cfg["name"] == "unet":
        if model_cfg["encoder"] == "tu-seresnet34":
            return smp.Unet(
                encoder_name=model_cfg["encoder"],
                encoder_weights=None,
                in_channels=3,
                classes=1
            )

        return smp.Unet(
                encoder_name=model_cfg["encoder"],
                encoder_weights="imagenet" if model_cfg.get("pretrained", False) else None,
                in_channels=3,
                classes=1
            )
    
    elif model_cfg["name"] == "dlab":
        return smp.DeeplabV3Plus(
                encoder_name=model_cfg["encoder"],
                encoder_weights="imagenet" if model_cfg.get("pretrained", False) else None,
                in_channels=3,
                classes=1
            )

