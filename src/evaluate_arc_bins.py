#!/usr/bin/env python3
"""Evaluate ARC ablation checkpoints in sequence-homology and IC bins.

Homology is the maximum BLAST identity of each validation/test query against
training sequences. IC is the maximum information content of a protein's positive
training-derived GO terms. Both are computed without using predictions to
define bins.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_DIR))

from src.arc_dataset import ArcGraphDataset, make_dataloader
from src.evals import compute_ic, evaluate_all
from src.train_arc_ablation import build, transform

ONTOLOGIES = ("molecular_function", "biological_process", "cellular_component")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--tuning-root", type=Path, default=None)
    p.add_argument("--ablations-root", type=Path, default=None)
    p.add_argument("--graph-root", type=Path, default=None)
    p.add_argument("--homology-tsv", type=Path, default=None,
                   help="Test-vs-train BLAST table.")
    p.add_argument("--validation-homology-tsv", type=Path, default=None,
                   help="Validation-vs-train BLAST table.")
    p.add_argument("--splits", nargs="+", choices=("valid", "test"),
                   default=("valid", "test"),
                   help="Evaluation splits to write (default: validation and test).")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--blastp", default="blastp")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    return p.parse_args()


def load_dataset(path: Path, graph_root: Path, split: str):
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, ArcGraphDataset) or dataset.split != split:
        raise RuntimeError(f"Unexpected ARC dataset: {path}")
    dataset.graph_dir = str(graph_root.resolve())
    return dataset



def ensure_validation_homology(path: Path, data_root: Path, blastp: str, threads: int) -> Path:
    """Build validation-vs-training BLAST hits once from the locked split."""
    if path.is_file() and path.stat().st_size > 0:
        return path
    split_root = data_root / "pdb_splits" / "threshold_30"
    query = split_root / "_valid_sequences.fasta"
    database = split_root / "blast_train_db"
    if not query.is_file() or not Path(str(database) + ".pin").is_file():
        raise FileNotFoundError(
            "Validation homology requires _valid_sequences.fasta and the existing "
            f"blast_train_db under {split_root}"
        )
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        blastp, "-query", str(query), "-db", str(database), "-out", str(partial),
        "-evalue", "1e-3", "-max_target_seqs", "50", "-num_threads", str(threads),
        "-outfmt", "6 qseqid sseqid pident length qlen slen evalue bitscore qcovs",
    ]
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(f"Validation BLAST wrote no hits to {partial}")
    os.replace(partial, path)
    return path

def homology_bins(path: Path, test_ids: list[str]) -> list[str]:
    maximum = {protein: 0.0 for protein in test_ids}
    if path.is_file():
        with path.open() as handle:
            for line in handle:
                fields = line.rstrip().split("\t")
                if len(fields) >= 3 and fields[0] in maximum:
                    try:
                        maximum[fields[0]] = max(maximum[fields[0]], float(fields[2]))
                    except ValueError:
                        pass
    result = []
    for protein in test_ids:
        value = maximum[protein]
        if value <= 0:
            result.append("no_hit")
        elif value < 30:
            result.append("<30%")
        elif value < 40:
            result.append("30-40%")
        elif value < 60:
            result.append("40-60%")
        else:
            result.append(">=60%")
    return result


def ic_bins(labels: np.ndarray, ic: np.ndarray) -> list[str]:
    values = []
    for row in labels:
        positive = ic[row > 0]
        values.append(float(np.max(positive)) if positive.size else float("nan"))
    result = []
    for value in values:
        if not np.isfinite(value):
            result.append("no_positive_terms")
        elif value < 2:
            result.append("<2_bits")
        elif value < 4:
            result.append("2-4_bits")
        elif value < 6:
            result.append("4-6_bits")
        else:
            result.append(">=6_bits")
    return result


def grouped_metrics(y_true, probabilities, groups, ic):
    rows = []
    for group in sorted(set(groups)):
        mask = np.asarray([item == group for item in groups])
        if not mask.any():
            continue
        y_group = y_true[mask]
        # A bin with no positive labels has no defined Fmax/AUPR/AUROC.
        # Record it explicitly as NaN instead of repeatedly invoking sklearn
        # on an undefined class and flooding the Slurm error log.
        if int(y_group.sum()) == 0:
            metrics = {"Micro_Fmax": np.nan, "Macro_Fmax": np.nan,
                       "Macro_AUROC": np.nan, "Micro_AUROC": np.nan,
                       "Macro_AUPRC": np.nan, "Micro_AUPRC": np.nan,
                       "Smin": np.nan}
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics = evaluate_all(y_group, probabilities[mask], ic)
        positive_counts = np.sum(y_group, axis=0)
        rows.append({
            "bin": group,
            "examples": int(mask.sum()),
            "positive_examples": int(np.sum(np.any(y_group > 0, axis=1))),
            "positive_term_assignments": int(np.sum(y_group)),
            "positive_terms": int(np.sum(positive_counts > 0)),
            "valid_auroc_terms": int(np.sum((positive_counts > 0) & (positive_counts < int(mask.sum())))),
            "all_positive_terms": int(np.sum(positive_counts == int(mask.sum()))),
            **{k: (float(v) if np.isfinite(v) else None) for k, v in metrics.items()},
        })
    return rows


def main():
    args = parse_args()
    data_root = Path(args.data_root or os.environ.get(
        "DGG_DATA_ROOT", PROJECT_DIR / "preprocessing" / "data_arc_rebuild_2026_07_14"
    )).expanduser().resolve()
    tuning_root = Path(args.tuning_root or os.environ.get(
        "DGG_TUNING_ROOT", PROJECT_DIR / "arc_tuning_cafa"
    )).expanduser().resolve()
    graph_root = Path(args.graph_root or os.environ.get(
        "DGG_GRAPH_ROOT", PROJECT_DIR / "arc_tuning_cafa" / "graphs_protbert"
    )).expanduser().resolve()
    ablations_root = Path(args.ablations_root or tuning_root / "ablations" /
                          "nominal_30_identity_80_coverage").expanduser().resolve()
    homology_paths = {
        "test": Path(args.homology_tsv or data_root / "pdb_splits" / "threshold_30" /
                     "blast_te_vs_tr.tsv").expanduser().resolve(),
        "valid": Path(args.validation_homology_tsv or data_root / "pdb_splits" /
                      "threshold_30" / "blast_va_vs_tr.tsv").expanduser().resolve(),
    }
    if "valid" in args.splits:
        homology_paths["valid"] = ensure_validation_homology(
            homology_paths["valid"], data_root, args.blastp, args.threads
        )
    output_dir = Path(args.output_dir or ablations_root / "bin_evaluation").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    all_checkpoint_paths = sorted(set(ablations_root.rglob("best_checkpoint.pt")))
    if not all_checkpoint_paths:
        raise FileNotFoundError(
            "No best_checkpoint.pt files found under "
            f"{ablations_root}. Check --tuning-root/--ablations-root; the ARC CAFA "
            "checkpoints normally live under arc_tuning_cafa."
        )

    for ontology in ONTOLOGIES:
        train = load_dataset(tuning_root / "datasets" / f"{ontology}_train.pkl",
                             graph_root, "train")
        ic = compute_ic(train.labels)
        checkpoint_paths = [path for path in all_checkpoint_paths
                            if ablations_root.joinpath(ontology) in path.parents]
        for evaluation_split in args.splits:
            dataset = load_dataset(
                tuning_root / "datasets" / f"{ontology}_{evaluation_split}.pkl",
                graph_root, evaluation_split,
            )
            hgroups = homology_bins(homology_paths[evaluation_split], dataset.protein_ids)
            igroups = ic_bins(dataset.labels, ic)
            loader = make_dataloader(dataset, args.batch_size, False, args.workers)
            for checkpoint_path in checkpoint_paths:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                config = checkpoint["config"]
                model_name = config["model"]
                modality = config.get("input_modality", "full")
                model = build(
                    model_name, int(dataset[0].x.shape[1]),
                    int(config.get("hidden_dim", 512)), dataset.num_classes,
                    float(config.get("dropout", 0.2)),
                ).to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                labels, probabilities = [], []
                with torch.inference_mode():
                    for batch in loader:
                        batch = transform(batch.to(device), modality)
                        labels.append(batch.y.cpu().numpy())
                        probabilities.append(
                            torch.sigmoid(model(batch.x, batch.edge_index, batch.batch)).cpu().numpy()
                        )
                y_true = np.vstack(labels)
                y_probability = np.vstack(probabilities)
                base = {
                    "ontology": ontology,
                    "model": model_name,
                    "input_modality": modality,
                    "checkpoint": str(checkpoint_path),
                    "split_label": config.get("split_label"),
                    "evaluation_split": evaluation_split,
                }
                for kind, groups in (("homology", hgroups), ("ic", igroups)):
                    for metric_row in grouped_metrics(y_true, y_probability, groups, ic):
                        rows.append({**base, "bin_type": kind, **metric_row})
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError(f"No bin-evaluation rows were produced from {ablations_root}")
    fields = list(rows[0]) if rows else [
        "ontology", "model", "input_modality", "checkpoint", "split_label",
        "evaluation_split", "bin_type", "bin", "examples",
    ]
    with (output_dir / "bin_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "bin_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {len(rows)} grouped model/bin rows for {', '.join(args.splits)} to {output_dir}")


if __name__ == "__main__": main()
