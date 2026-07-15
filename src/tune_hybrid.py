#!/usr/bin/env python3
"""Validation-only Hybrid tuning for the ARC nominal 30%/80% split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.arc_dataset import ArcGraphDataset, make_dataloader
from src.model import HybridGNN, HybridGNN_JK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--ontology", choices=["molecular_function", "biological_process", "cellular_component"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trial-configs", type=Path)
    parser.add_argument("--trial-id", type=int)
    parser.add_argument("--selected-config", type=Path)
    parser.add_argument("--model", choices=["Hybrid", "Hybrid_JK"], default="Hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def read_config(args: argparse.Namespace) -> dict:
    if args.selected_config:
        selected = json.loads(args.selected_config.read_text())
        config = dict(selected["ontologies"][args.ontology]["config"])
        config["source"] = str(args.selected_config)
        return config
    if args.trial_configs is None or args.trial_id is None:
        raise SystemExit("Specify --trial-configs and --trial-id, or --selected-config")
    rows = [json.loads(line) for line in args.trial_configs.read_text().splitlines() if line.strip()]
    matches = [row for row in rows if int(row["trial_id"]) == args.trial_id]
    if len(matches) != 1:
        raise SystemExit(f"Expected one configuration for trial {args.trial_id}, found {len(matches)}")
    return matches[0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; Hybrid tuning requires one gpu-l40 GPU")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    value = (torch.ones(8, device=device) @ torch.ones(8, device=device)).item()
    print(f"CUDA verified: {name}; tensor dot product={value}")
    log_gpu("startup")
    return device


def log_gpu(stage: str) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True).strip()
    except Exception as exc:
        output = f"nvidia-smi failed: {exc}"
    print(f"GPU {stage}: {output}")


def load_training_data(dataset_dir: Path, ontology: str):
    loaded = {}
    for split in ("train", "valid"):
        path = dataset_dir / f"{ontology}_{split}.pkl"
        with path.open("rb") as handle:
            dataset = pickle.load(handle)
        if not isinstance(dataset, ArcGraphDataset) or dataset.split != split:
            raise SystemExit(f"Unexpected dataset schema: {path}")
        loaded[split] = dataset
    if loaded["train"].terms != loaded["valid"].terms:
        raise SystemExit("Train/validation ontology vocabularies differ")
    print(f"Loaded train={len(loaded['train'])}, valid={len(loaded['valid'])}; reserved test set is not loaded")
    return loaded["train"], loaded["valid"]


def positive_weights(train_labels: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    counts = train_labels.sum(axis=0)
    negatives = train_labels.shape[0] - counts
    weights = np.ones_like(counts, dtype=np.float32)
    observed = counts > 0
    weights[observed] = negatives[observed] / counts[observed]
    weights = np.clip(weights, 1.0, 1000.0)
    return torch.from_numpy(weights), counts


def weighted_loss(logits, targets, pos_weight, loss_name: str, gamma: float | None):
    if loss_name == "BCE":
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_true = torch.exp(-base)
    class_weight = 1.0 + targets * (pos_weight - 1.0)
    return (((1.0 - probability_true) ** float(gamma)) * base * class_weight).mean()


def micro_fmax(y_true: np.ndarray, y_probability: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_probability.ravel())
    scores = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.nanargmax(scores))
    threshold = float(thresholds[index]) if index < len(thresholds) else 1.0
    return float(scores[index]), threshold


def summarize_validation(y_true: np.ndarray, probabilities: np.ndarray, train_counts: np.ndarray) -> dict:
    overall, best_threshold = micro_fmax(y_true, probabilities)
    rare_mask = (train_counts > 0) & (train_counts <= 10)
    rare = None
    if rare_mask.any() and y_true[:, rare_mask].sum() > 0:
        rare = micro_fmax(y_true[:, rare_mask], probabilities[:, rare_mask])[0]
    sensitivity = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = probabilities >= threshold
        tp = float(np.logical_and(predictions, y_true == 1).sum())
        fp = float(np.logical_and(predictions, y_true == 0).sum())
        fn = float(np.logical_and(~predictions, y_true == 1).sum())
        score = 2 * tp / (2 * tp + fp + fn + 1e-12)
        sensitivity.append({"threshold": round(float(threshold), 1), "micro_f1": score})
    confidence = probabilities.ravel()
    truth = y_true.ravel()
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - truth[mask].mean())
    return {
        "validation_micro_fmax": overall,
        "validation_micro_fmax_threshold": best_threshold,
        "validation_rare_term_micro_fmax_train_count_1_to_10": rare,
        "rare_term_count": int(rare_mask.sum()),
        "brier_score": float(np.mean((probabilities - y_true) ** 2)),
        "expected_calibration_error_10_bins": float(ece),
        "threshold_sensitivity": sensitivity,
    }


def evaluate(model, loader, device, train_counts):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            labels.append(batch.y.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.vstack(labels)
    y_probability = np.vstack(probabilities)
    return summarize_validation(y_true, y_probability, train_counts)


def main() -> None:
    args = parse_args()
    config = read_config(args)
    set_seed(args.seed)
    device = require_cuda()
    train_dataset, valid_dataset = load_training_data(args.dataset_dir.resolve(), args.ontology)
    train_loader = make_dataloader(train_dataset, int(config["batch_size"]), True, args.workers)
    valid_loader = make_dataloader(valid_dataset, int(config["batch_size"]), False, args.workers)
    pos_weight, train_counts = positive_weights(train_dataset.labels)
    pos_weight = pos_weight.to(device)

    sample = train_dataset[0]
    hidden = int(config["hidden_dim"])
    model_class = HybridGNN if args.model == "Hybrid" else HybridGNN_JK
    model = model_class(
        int(sample.x.shape[1]), [hidden, hidden], train_dataset.num_classes,
        num_attention_heads=4, dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **config,
        "model": args.model,
        "ontology": args.ontology,
        "seed": args.seed,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "selection_data": "validation only",
        "positive_weight_source": "training labels only",
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    with (run_dir / "training_positive_weights.json").open("w") as handle:
        json.dump({"counts": train_counts.astype(int).tolist(), "weights": pos_weight.cpu().tolist()}, handle)

    best_score = -1.0
    epochs_without_improvement = 0
    history = []
    patience = int(config["patience"])
    gamma = config.get("focal_gamma")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = weighted_loss(logits, batch.y.float(), pos_weight, config["loss"], gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, valid_loader, device, train_counts)
        score = metrics["validation_micro_fmax"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
        print(f"epoch={epoch} loss={np.mean(losses):.6f} valid_micro_fmax={score:.6f}")
        if score > best_score + 1e-8:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": run_config, "metrics": metrics},
                run_dir / "best_checkpoint.pt",
            )
            (run_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs (patience={patience})")
                break

    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "validation_micro_fmax", "validation_micro_fmax_threshold",
                        "validation_rare_term_micro_fmax_train_count_1_to_10", "rare_term_count",
                        "brier_score", "expected_calibration_error_10_bins", "threshold_sensitivity"],
        )
        writer.writeheader()
        writer.writerows(history)
    log_gpu("completion")


if __name__ == "__main__":
    main()
