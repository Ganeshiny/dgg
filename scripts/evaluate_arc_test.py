#!/usr/bin/env python3
"""Evaluate selected ARC checkpoints on the reserved test split.

The validation threshold saved with each checkpoint is used for the primary
test F1 so the test set is never used for model or threshold selection.
Test-set F-max is reported only as a descriptive upper bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_DIR))

from src.arc_dataset import ArcGraphDataset, make_dataloader
from src.model import HybridGNN, HybridGNN_JK
from src.tune_hybrid import micro_fmax


SEEDS = (1103, 2207, 3301, 4409, 5501)
ONTOLOGIES = ("molecular_function", "biological_process", "cellular_component")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tuning-root", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("Hybrid", "Hybrid_JK"), default="Hybrid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


def build_model(checkpoint: dict, sample, num_classes: int, model_name: str, device: torch.device):
    config = checkpoint["config"]
    hidden = int(config["hidden_dim"])
    dropout = float(config["dropout"])
    model_class = HybridGNN if model_name == "Hybrid" else HybridGNN_JK
    model = model_class(
        int(sample.x.shape[1]), [hidden, hidden], num_classes,
        num_attention_heads=4, dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def evaluate_checkpoint(model, loader, device: torch.device, threshold: float):
    labels, probabilities = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            probabilities.append(torch.sigmoid(model(batch.x, batch.edge_index, batch.batch)).cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    y_true = np.vstack(labels)
    y_prob = np.vstack(probabilities)
    y_pred = y_prob >= threshold
    tp = float(np.logical_and(y_pred, y_true == 1).sum())
    fp = float(np.logical_and(y_pred, y_true == 0).sum())
    fn = float(np.logical_and(~y_pred, y_true == 1).sum())
    fixed_f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
    test_fmax, test_fmax_threshold = micro_fmax(y_true, y_prob)
    return {
        "test_micro_f1_at_validation_threshold": float(fixed_f1),
        "validation_threshold": float(threshold),
        "test_micro_fmax_diagnostic": float(test_fmax),
        "test_micro_fmax_threshold_diagnostic": float(test_fmax_threshold),
        "test_examples": int(y_true.shape[0]),
        "test_classes": int(y_true.shape[1]),
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    tuning_root = args.tuning_root.expanduser().resolve()
    checkpoints_root = args.checkpoints_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    rows = []
    for ontology in ONTOLOGIES:
        dataset_path = tuning_root / "datasets" / f"{ontology}_test.pkl"
        with dataset_path.open("rb") as handle:
            dataset = pickle.load(handle)
        if not isinstance(dataset, ArcGraphDataset) or dataset.split != "test":
            raise SystemExit(f"Unexpected test dataset schema: {dataset_path}")
        loader = make_dataloader(dataset, args.batch_size, shuffle=False, workers=args.workers)
        sample = dataset[0]
        for seed in SEEDS:
            checkpoint_path = checkpoints_root / ontology / f"seed_{seed}" / "best_checkpoint.pt"
            if not checkpoint_path.is_file():
                raise SystemExit(f"Missing checkpoint: {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, device)
            model, config = build_model(checkpoint, sample, dataset.num_classes, args.model, device)
            metrics = evaluate_checkpoint(
                model, loader, device,
                float(checkpoint["metrics"]["validation_micro_fmax_threshold"]),
            )
            row = {"ontology": ontology, "seed": seed, **metrics, **{
                "learning_rate": config.get("learning_rate"),
                "weight_decay": config.get("weight_decay"),
                "dropout": config.get("dropout"),
                "hidden_dim": config.get("hidden_dim"),
                "batch_size": config.get("batch_size"),
                "loss": config.get("loss"),
                "checkpoint": str(checkpoint_path),
            }}
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fields = list(rows[0])
    with (output_dir / "per_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for ontology in ONTOLOGIES:
        values = [r["test_micro_f1_at_validation_threshold"] for r in rows if r["ontology"] == ontology]
        summary[ontology] = {
            "seeds": len(values),
            "mean_test_micro_f1": float(np.mean(values)),
            "std_test_micro_f1": float(np.std(values, ddof=1)),
            "min_test_micro_f1": float(np.min(values)),
            "max_test_micro_f1": float(np.max(values)),
        }
    (output_dir / "summary.json").write_text(json.dumps({"model": args.model, "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
