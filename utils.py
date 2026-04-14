import gc
import torch
from tqdm import tqdm


def convert_numerics(obj):
        if isinstance(obj, dict):
            return {k: convert_numerics(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numerics(item) for item in obj]
        elif isinstance(obj, str):
            try:
                return float(obj)
            except ValueError:
                return obj
        else:
            return obj
        
def make_predictions(loader, model,):
    """Make predictions when TiledDataset is used while testing."""
    dataset = loader.dataset

    preds = []
    patch_info = []

    for batch in tqdm(loader, desc="Predicting patches"):
        images, masks, img_idx, y, x, pad_h, pad_w = batch
        images = images.cuda()

        with torch.no_grad():
            pred = model(images)
            pred = torch.sigmoid(pred).cpu()

        preds.append(pred)
        patch_info.extend(zip(
            img_idx.tolist(),
            y.tolist(),
            x.tolist(),
            pad_h.tolist(),
            pad_w.tolist()
        ))

        # free memory
        del images, pred
        torch.cuda.empty_cache()

    print(30)
    preds = torch.cat(preds, dim=0)
    print(31)
    final_preds = stitch_predictions(preds, patch_info, dataset.image_h, dataset.image_w, patch_size=dataset.patch_size)
    print(32)

    for img_idx in final_preds:
        final_preds[img_idx] = (final_preds[img_idx] > 0.5).float()
    print(33)
    

    print(34)
    del preds, patch_info, loader, dataset
    print(35)
    torch.cuda.empty_cache()
    print(36)
    gc.collect()
    print(37)

    return final_preds


def stitch_predictions(patch_preds, patch_info, image_h, image_w, patch_size=256):
    all_img_idx = [info[0] for info in patch_info]
    unique_imgs = set(all_img_idx)

    pad_h = patch_info[0][3]
    pad_w = patch_info[0][4]
    padded_h = image_h + pad_h
    padded_w = image_w + pad_w

    # sum + count for averaging
    full_preds = {
        img_idx: torch.zeros(padded_h, padded_w)
        for img_idx in unique_imgs
    }

    count_map = {
        img_idx: torch.zeros(padded_h, padded_w)
        for img_idx in unique_imgs
    }

    for pred, (img_idx, y, x, _, _) in zip(patch_preds, patch_info):
        pred = pred.squeeze()

        full_preds[img_idx][y:y+patch_size, x:x+patch_size] += pred
        count_map[img_idx][y:y+patch_size, x:x+patch_size] += 1

    # avoid division by zero
    final_preds = {}
    for img_idx in unique_imgs:
        avg_pred = full_preds[img_idx] / torch.clamp(count_map[img_idx], min=1)
        final_preds[img_idx] = avg_pred[:image_h, :image_w]

    return final_preds


def compute_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }
