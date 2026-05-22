import segmentation_models_pytorch as smp
from .mod_deeplab_v3_plus import replace_aspp_with_attentive


def build_model(config):
    model_cfg = config["model"]
    atrous_rates = model_cfg.get("atrous_rates", [12, 24, 36])
    output_stride = model_cfg.get("output_stride", 16)
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
        return smp.DeepLabV3Plus(
                encoder_name=model_cfg["encoder"],
                encoder_weights="imagenet" if model_cfg.get("pretrained", False) else None,
                in_channels=3,
                classes=1,
                encoder_output_stride=output_stride,
                decoder_atrous_rates=atrous_rates
            )
    elif model_cfg["name"] == "mod_dlab":
        model = smp.DeepLabV3Plus(
                encoder_name=model_cfg["encoder"],
                encoder_weights="imagenet" if model_cfg.get("pretrained", False) else None,
                in_channels=3,
                classes=1,
                encoder_output_stride=output_stride,
                decoder_atrous_rates=atrous_rates
            )
        replace_aspp_with_attentive(model)
        return model

