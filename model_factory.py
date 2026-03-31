import segmentation_models_pytorch as smp


def build_model(config):
    if config["model"]["name"] == "unet":
        return smp.Unet(
                encoder_name=config["model"]["encoder"],
                encoder_weights="imagenet" if config["model"]["pretrained"] else None,
                in_channels=3,
                classes=1
            )
