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
