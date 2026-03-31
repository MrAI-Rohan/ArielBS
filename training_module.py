import time
import wandb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from model_factory import build_model
from loss_functions import get_loss_function
from optimizer_factory import build_optimizer
from scheduler_factory import build_scheduler

class SegmentationModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Save config to checkpoints
        self.save_hyperparameters(config)

        # Build components
        self.model = build_model(config)
        self.loss_fn = get_loss_function(config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)

        loss = self.loss_fn(preds, masks)
        iou = self.compute_iou(preds, masks)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_iou", iou, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)

        loss = self.loss_fn(preds, masks)
        metrics = self.compute_metrics(preds, masks)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_iou", metrics["iou"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", metrics["precision"], on_step=False, on_epoch=True)
        self.log("val_recall", metrics["recall"], on_step=False, on_epoch=True)
        self.log("val_f1", metrics["f1"], on_step=False, on_epoch=True)

        if batch_idx == 0:
            self.log_images(images, masks, preds)

    def configure_optimizers(self):
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        encoder_lr = float(self.config['training']['encoder_lr'])
        decoder_lr = float(self.config['training']['decoder_lr'])

        param_groups = [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': decoder_params, 'lr': decoder_lr}
        ]
        optimizer = build_optimizer(param_groups, self.config)
        scheduler = build_scheduler(optimizer, self.config)

        if scheduler is None:
            return optimizer

        if self.config["scheduler"]["name"] == "reduce_on_plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config["scheduler"]["monitor"]
                }
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        print(f"\nEpoch {self.current_epoch + 1} started")

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        encoder_lr = float(self.optimizers().param_groups[0]["lr"])
        decoder_lr = float(self.optimizers().param_groups[1]["lr"])

        print(f"Epoch {self.current_epoch + 1} finished "
            f"| Time: {epoch_time:.2f}s "
            f"| Encoder LR: {encoder_lr:.6f} "
            f"| Decoder LR: {decoder_lr:.6f}")

        self.log("epoch_time", epoch_time)
        self.log("encoder_lr", encoder_lr, on_epoch=True)
        self.log("decoder_lr", decoder_lr, on_epoch=True)
    
    def compute_iou(self, preds, targets, threshold=0.5, smooth=1e-6):
        preds = torch.sigmoid(preds)
        preds = (preds > threshold).float()

        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()
    
    def compute_metrics(self, preds, targets, threshold=0.5, smooth=1e-6):
        preds = torch.sigmoid(preds)
        preds = (preds > threshold).float()

        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (preds * targets).sum(dim=1)
        FP = (preds * (1 - targets)).sum(dim=1)
        FN = ((1 - preds) * targets).sum(dim=1)
        TN = ((1 - preds) * (1 - targets)).sum(dim=1)

        precision = (TP + smooth) / (TP + FP + smooth)
        recall = (TP + smooth) / (TP + FN + smooth)
        f1 = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
        iou = (TP + smooth) / (TP + FP + FN + smooth)

        return {
            "precision": precision.mean(),
            "recall": recall.mean(),
            "f1": f1.mean(),
            "iou": iou.mean(),
        }


    def log_images(self, images, masks, preds):
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        images = images.cpu()
        masks = masks.cpu()
        preds = preds.cpu()

        log_list = []

        for i in range(min(3, images.shape[0])):
            img = images[i]
            gt = masks[i]
            pr = preds[i]

            log_list.append(
                wandb.Image(
                    img,
                    masks={
                        "ground_truth": {"mask_data": gt.squeeze().numpy()},
                        "prediction": {"mask_data": pr.squeeze().numpy()}
                    }
                )
            )

        self.logger.experiment.log({"predictions": log_list})

