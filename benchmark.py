import os
import gc
import h5py
import argparse
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from utils import compute_metrics
from data.dataset import TiledDataset
from data.transforms import build_transforms
from training_module import SegmentationModule


def load_data(h5_path, patch_size,  batch_size, transform=None, num_workers=0):
    dataset = TiledDataset(h5_path, patch_size=patch_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return dataloader

def load_model(ckpt_path, device="cuda"):
    model = SegmentationModule.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    return model


def make_predictions_and_count(loader, model, h5_path, patch_size):
    tp = fp = fn = tn = 0

    current_img = None
    full_pred = None
    count_map = None

    with h5py.File(h5_path, 'r') as f:
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting patches", unit="batch", total=len(loader)):
                images, _, img_idx, y, x, pad_h, pad_w = batch
                images = images.cuda(non_blocking=True)

                preds = torch.sigmoid(model(images)).cpu()

                for i in range(preds.shape[0]):
                    img = img_idx[i].item()
                    yi = y[i].item()
                    xi = x[i].item()

                    # NEW IMAGE → finalize previous one
                    if current_img is not None and img != current_img:
                        avg = full_pred / torch.clamp(count_map, min=1)
                        avg = avg[:orig_h, :orig_w]
                        pred_mask = (avg > 0.5)

                        gt = torch.from_numpy(f['masks'][current_img]).float()

                        tp += ((pred_mask == 1) & (gt == 1)).sum().item()
                        fp += ((pred_mask == 1) & (gt == 0)).sum().item()
                        fn += ((pred_mask == 0) & (gt == 1)).sum().item()
                        tn += ((pred_mask == 0) & (gt == 0)).sum().item()

                        # cleanup
                        del full_pred, count_map
                        torch.cuda.empty_cache()

                    # initialize new image
                    if img != current_img:
                        current_img = img

                        orig_h, orig_w = f['masks'][img].shape
                        padded_h = orig_h + pad_h[i].item()
                        padded_w = orig_w + pad_w[i].item()

                        full_pred = torch.zeros(padded_h, padded_w, dtype=torch.float16)
                        count_map = torch.zeros(padded_h, padded_w, dtype=torch.float16)

                    patch = preds[i].squeeze()

                    full_pred[yi:yi+patch_size, xi:xi+patch_size] += patch
                    count_map[yi:yi+patch_size, xi:xi+patch_size] += 1

                del images, preds
                torch.cuda.empty_cache()

        # finalize last image
        if current_img is not None:
            avg = full_pred / torch.clamp(count_map, min=1)
            avg = avg[:orig_h, :orig_w]
            pred_mask = (avg > 0.5)

            gt = torch.from_numpy(f['masks'][current_img]).float()

            tp += ((pred_mask == 1) & (gt == 1)).sum().item()
            fp += ((pred_mask == 1) & (gt == 0)).sum().item()
            fn += ((pred_mask == 0) & (gt == 1)).sum().item()
            tn += ((pred_mask == 0) & (gt == 0)).sum().item()

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

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
        try:
            df_old = pd.read_csv(csv_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except pd.errors.EmptyDataError:
            # file exists but is empty
            df = df_new
    else:
        df = df_new

    df.to_csv(csv_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Benchmarking script for WHU building segmentation.")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the HDF5 dataset.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size for testing.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for testing.")
    parser.add_argument("--normalization", type=str, default="imagenet", help="Normalization method (imagenet, standard).")
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"Checkpoint path {ckpt_path} does not exist.")
        return
    
    model = load_model(args.ckpt_path)

    data_cfg = {
        "normalization": args.normalization,
    }
    transform = build_transforms(data_cfg, mode="val")

    whu_test = load_data(h5_path/"whu_test.h5", patch_size=args.patch_size, 
                        batch_size=args.batch_size, transform=transform)
    counts_whu = make_predictions_and_count(whu_test, model, h5_path/"whu_test.h5", patch_size=args.patch_size)
    del whu_test
    torch.cuda.empty_cache()
    gc.collect()

    mas = load_data(h5_path/"massachusetts.h5", patch_size=args.patch_size, 
                        batch_size=args.batch_size, transform=transform)
    counts_mas = make_predictions_and_count(mas, model, h5_path/"massachusetts.h5", patch_size=args.patch_size)
    del mas
    torch.cuda.empty_cache()
    gc.collect()

    zanzibar = load_data(h5_path/"zanzibar.h5", patch_size=args.patch_size, 
                        batch_size=args.batch_size, transform=transform)
    counts_zanzibar = make_predictions_and_count(zanzibar, model, h5_path/"zanzibar.h5", patch_size=args.patch_size)
    del zanzibar
    torch.cuda.empty_cache()
    gc.collect()

    results = evaluate_datasets( 
        [counts_mas, counts_whu, counts_zanzibar],
        dataset_names=["Massachusetts", "WHU Test", "Zanzibar"]
    )

    results2 = evaluate_datasets( 
        [counts_whu, counts_zanzibar],
        dataset_names=["WHU Test", "Zanzibar"]
    )


    try:
        save_results_to_csv(results, config_name=ckpt_path.stem, csv_path="benchmark_results.csv")
        save_results_to_csv(results2, config_name=ckpt_path.stem, csv_path="mz_benchmark.csv")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        print("Results:")
        print(results)
        print(results2)


if __name__ == "__main__":
    main()
