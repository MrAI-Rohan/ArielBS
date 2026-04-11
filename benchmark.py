import os
import gc
import h5py
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from data.dataset import TiledDataset
from data.transforms import build_transforms
from training_module import SegmentationModule
from utils import make_predictions, compute_metrics


def load_data(h5_path, patch_size,  batch_size, transform=None, num_workers=2):
    dataset = TiledDataset(h5_path, patch_size=patch_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def load_model(ckpt_path, device="cuda"):
    model = SegmentationModule.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    return model

def benchmark_counts(final_preds, h5_path):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    with h5py.File(h5_path, 'r') as f:
        for img_idx, pred_mask in final_preds.items():
            gt_mask = torch.tensor(f['masks'][img_idx]).float()

            pred_mask = pred_mask.float()

            tp += ((pred_mask == 1) & (gt_mask == 1)).sum().item()
            fp += ((pred_mask == 1) & (gt_mask == 0)).sum().item()
            fn += ((pred_mask == 0) & (gt_mask == 1)).sum().item()
            tn += ((pred_mask == 0) & (gt_mask == 0)).sum().item()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

def evaluate_datasets(counts_list, dataset_names=None):
    """
    counts_list: list of dicts (each dict = one dataset output from benchmark_counts)
    dataset_names: optional list of names for datasets

    Returns:
        {
            "per_dataset": {dataset_name: metrics_dict},
            "aggregate": metrics_dict
        }
    """

    results = {}
    total_tp = total_fp = total_fn = total_tn = 0

    # Assign default names if not provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(counts_list))]

    # Per dataset metrics
    for name, counts in zip(dataset_names, counts_list):
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]

        results[name] = compute_metrics(tp, fp, fn, tn)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    # Aggregate metrics (micro average)
    aggregate = compute_metrics(total_tp, total_fp, total_fn, total_tn)

    return {
        "per_dataset": results,
        "aggregate": aggregate
    }

def run_single_dataset(model, loader, h5_path):
    final_preds = make_predictions(loader, model)

    counts = benchmark_counts(final_preds, h5_path)

    # cleanup memory
    loader.dataset.close()  # close h5 file
    del final_preds
    del dataset

    torch.cuda.empty_cache()
    gc.collect()

    return counts

def save_results_to_csv(results, config_name, csv_path="benchmark_results.csv"):
    rows = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # per dataset
    for dataset, metrics in results["per_dataset"].items():
        row = {
            "config": config_name,
            "dataset": dataset,
            "timestamp": timestamp,
        }
        row.update(metrics)
        rows.append(row)

    # aggregate
    agg_row = {
        "config": config_name,
        "dataset": "aggregate",
        "timestamp": timestamp,
    }
    agg_row.update(results["aggregate"])
    rows.append(agg_row)

    df_new = pd.DataFrame(rows)

    # append or create
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Benchmarking script for WHU building segmentation.")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the HDF5 dataset.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size for testing.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for testing.")
    parser.add_argument("--normalization", type=str, default="imagenet", help="Normalization method (imagenet, standard).")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"Checkpoint path {ckpt_path} does not exist.")
        return
    
    model = load_model(args.ckpt_path)

    data_cfg = {
        "normalization": args.normalization,
    }
    transform = build_transforms(data_cfg, mode="val")

    mas = load_data(args.h5_path/"massachusetts.h5", patch_size=args.patch_size, 
                        batch_size=args.batch_size, transform=transform)
    counts_mas = run_single_dataset(mas, model, args.h5_path/"massachusetts.h5")

    whu_test = load_data(args.h5_path/"whu_test.h5", patch_size=args.patch_size, 
                        batch_size=args.batch_size, transform=transform)
    counts_whu = run_single_dataset(whu_test, model, args.h5_path/"whu_test.h5")

    zanzibar = load_data(args.h5_path/"zanzibar.h5", patch_size=args.patch_size, 
                        batch_size=args.batch_size, transform=transform)
    counts_zanzibar = run_single_dataset(zanzibar, model, args.h5_path/"zanzibar.h5")

    results = evaluate_datasets( 
        [counts_mas, counts_whu, counts_zanzibar],
        dataset_names=["Massachusetts", "WHU Test", "Zanzibar"]
    )

    save_results_to_csv(results, config_name=args.ckpt_path.stem)


if __name__ == "__main__":
    main()
