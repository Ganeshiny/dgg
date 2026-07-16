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
from src.evals import compute_ic, evaluate_all


SEEDS = (1103, 2207, 3301, 4409, 5501)
ONTOLOGIES = ("molecular_function", "biological_process", "cellular_component")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--tuning-root", type=Path, default=None)
    parser.add_argument("--checkpoints-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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


def evaluate_checkpoint(model, loader, device: torch.device, threshold: float, ic: np.ndarray):
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
    cafa = evaluate_all(y_true, y_prob, ic)
    return {
        "test_micro_f1_at_validation_threshold": float(fixed_f1),
        "test_micro_fmax": float(cafa["Micro_Fmax"]),
        "test_macro_fmax": float(cafa["Macro_Fmax"]),
        "test_micro_aupr": float(cafa["Micro_AUPRC"]),
        "test_macro_aupr": float(cafa["Macro_AUPRC"]),
        "test_micro_auroc": float(cafa["Micro_AUROC"]),
        "test_macro_auroc": float(cafa["Macro_AUROC"]),
        "test_smin": float(cafa["Smin"]),
        "validation_threshold": float(threshold),
        "test_micro_fmax_diagnostic": float(test_fmax),
        "test_micro_fmax_threshold_diagnostic": float(test_fmax_threshold),
        "test_examples": int(y_true.shape[0]),
        "test_classes": int(y_true.shape[1]),
    }


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root or os.environ.get("DGG_DATA_ROOT", project_dir / "preprocessing" / "data_arc_rebuild_2026_07_14")).expanduser().resolve()
    tuning_root = Path(args.tuning_root or os.environ.get("DGG_TUNING_ROOT", project_dir / "arc_tuning")).expanduser().resolve()
    model_name = args.model.lower()
    checkpoints_root = Path(args.checkpoints_root or (tuning_root / f"five_seed_{model_name}")).expanduser().resolve()
    output_dir = Path(args.output_dir or (tuning_root / "test_evaluation" / model_name)).expanduser().resolve()
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
        # Relocate legacy pickles whose embedded graph_dir predates project-level arc_tuning.
        dataset.graph_dir = str(tuning_root / "graphs_protbert")
        train_path = tuning_root / "datasets" / f"{ontology}_train.pkl"
        with train_path.open("rb") as handle:
            train_dataset = pickle.load(handle)
        if not isinstance(train_dataset, ArcGraphDataset) or train_dataset.split != "train":
            raise SystemExit(f"Unexpected train dataset schema: {train_path}")
        ic = compute_ic(train_dataset.labels)
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
                ic,
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
    aggregate_metrics = (
        "test_micro_f1_at_validation_threshold", "test_micro_fmax", "test_macro_fmax",
        "test_micro_aupr", "test_macro_aupr", "test_micro_auroc", "test_macro_auroc", "test_smin",
    )
    for ontology in ONTOLOGIES:
        subset = [r for r in rows if r["ontology"] == ontology]
        summary[ontology] = {"seeds": len(subset)}
        for metric in aggregate_metrics:
            values = np.asarray([r[metric] for r in subset], dtype=float)
            key = metric.removeprefix("test_")
            summary[ontology][f"mean_{key}"] = float(np.mean(values))
            summary[ontology][f"std_{key}"] = float(np.std(values, ddof=1))
            summary[ontology][f"min_{key}"] = float(np.min(values))
            summary[ontology][f"max_{key}"] = float(np.max(values))
    (output_dir / "summary.json").write_text(json.dumps({"model": args.model, "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
