#!/usr/bin/env python3
"""Audit DPFunc checkpoint, protein, vocabulary, and score coverage.

This reads the raw DPFunc prediction workspaces and the benchmark's locked
label matrices. It does not recompute predictions or performance metrics.
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.core import load_label_npz


ONTOLOGIES = ("mf", "bp", "cc")
DEFAULT_WORKSPACE = Path(
    "arc_benchmark/nominal_30_identity_80_coverage"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--pre-name", default="DPFunc_model")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_dpfunc_vocabulary(dp_workspace: Path, ontology: str) -> list[str]:
    path = dp_workspace / "mlb" / f"{ontology}_go.mlb"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DPFunc label encoder: {path}")
    mlb = joblib.load(path)
    if not hasattr(mlb, "classes_"):
        raise TypeError(f"DPFunc label encoder has no classes_: {path}")
    terms = [str(term) for term in mlb.classes_]
    if len(terms) != len(set(terms)):
        raise ValueError(f"DPFunc label encoder contains duplicate GO IDs: {path}")
    return terms


def score_summary(
    frame: pd.DataFrame,
    protein_ids: list[str],
    benchmark_terms: list[str],
) -> dict[str, float | int]:
    requested = set(map(str, protein_ids))
    term_set = set(map(str, benchmark_terms))
    subset = frame[frame["protein_id"].astype(str).isin(requested)]
    observed: list[float] = []
    proteins_with_score = 0
    invalid_scores = 0
    out_of_range_scores = 0
    for predictions in subset["predictions"]:
        positive = False
        for term, raw_score in predictions.items():
            if str(term) not in term_set:
                continue
            score = float(raw_score)
            if not np.isfinite(score):
                invalid_scores += 1
                continue
            if score < 0 or score > 1:
                out_of_range_scores += 1
            observed.append(score)
            positive = positive or score > 0
        proteins_with_score += int(positive)

    total_cells = len(protein_ids) * len(benchmark_terms)
    implicit_zeros = max(total_cells - len(observed), 0)
    values = np.concatenate((
        np.asarray(observed, dtype=np.float64),
        np.zeros(implicit_zeros, dtype=np.float64),
    ))
    quantiles = np.quantile(values, [0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0])
    return {
        "proteins_with_any_positive_score": proteins_with_score,
        "observed_common_term_scores": len(observed),
        "implicit_zero_scores": implicit_zeros,
        "invalid_score_count": invalid_scores,
        "out_of_range_score_count": out_of_range_scores,
        "score_mean": float(values.mean()),
        "score_sd": float(values.std()),
        "score_q00": float(quantiles[0]),
        "score_q01": float(quantiles[1]),
        "score_q10": float(quantiles[2]),
        "score_q50": float(quantiles[3]),
        "score_q90": float(quantiles[4]),
        "score_q99": float(quantiles[5]),
        "score_q100": float(quantiles[6]),
    }


def audit_ontology(
    workspace: Path,
    ontology: str,
    pre_name: str,
) -> tuple[list[dict], dict]:
    dp_workspace = workspace / "raw" / "dpfunc" / ontology
    result_path = (
        dp_workspace / "results" / f"{pre_name}_{ontology}_final.pkl"
    )
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing DPFunc result: {result_path}")
    frame = pd.read_pickle(result_path)
    required = {"protein_id", "predictions"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{result_path} lacks columns {sorted(required - set(frame.columns))}"
        )
    if not frame["predictions"].map(lambda value: isinstance(value, dict)).all():
        raise TypeError(f"{result_path}: predictions must all be GO-score dictionaries")

    frame = frame.copy()
    frame["protein_id"] = frame["protein_id"].astype(str)
    duplicate_proteins = int(frame["protein_id"].duplicated().sum())
    vocabulary = load_dpfunc_vocabulary(dp_workspace, ontology)
    vocabulary_set = set(vocabulary)
    checkpoint_paths = sorted(
        (dp_workspace / "save_models").glob(
            f"{pre_name}_{ontology}_*of3model.pt"
        )
    )

    rows = []
    for split in ("valid", "test"):
        protein_ids, benchmark_terms, labels = load_label_npz(
            workspace, ontology, split
        )
        protein_set = set(map(str, protein_ids))
        frame_proteins = set(frame["protein_id"])
        overlap_mask = np.asarray(
            [str(term) in vocabulary_set for term in benchmark_terms],
            dtype=bool,
        )
        total_positive_labels = int(labels.sum())
        represented_positive_labels = int(labels[:, overlap_mask].sum())
        scores = score_summary(frame, protein_ids, benchmark_terms)
        row = {
            "ontology": ontology,
            "split": split,
            "expected_proteins": len(protein_ids),
            "matched_proteins": len(protein_set & frame_proteins),
            "protein_coverage": len(protein_set & frame_proteins) / len(protein_ids),
            "duplicate_result_proteins": duplicate_proteins,
            "checkpoint_count": len(checkpoint_paths),
            "dpfunc_vocabulary_terms": len(vocabulary),
            "benchmark_terms": len(benchmark_terms),
            "overlapping_terms": int(overlap_mask.sum()),
            "benchmark_term_coverage": float(overlap_mask.mean()),
            "total_positive_labels": total_positive_labels,
            "represented_positive_labels": represented_positive_labels,
            "positive_label_mass_coverage": (
                represented_positive_labels / total_positive_labels
                if total_positive_labels else float("nan")
            ),
            **scores,
        }
        problems = []
        if row["protein_coverage"] != 1.0:
            problems.append("incomplete protein coverage")
        if duplicate_proteins:
            problems.append("duplicate protein IDs")
        if len(checkpoint_paths) != 3:
            problems.append("expected three checkpoints")
        if scores["invalid_score_count"]:
            problems.append("non-finite scores")
        if scores["out_of_range_score_count"]:
            problems.append("scores outside [0,1]")
        if scores["score_sd"] < 1e-8:
            problems.append("near-constant score matrix")
        row["audit_status"] = "PASS" if not problems else "FAIL"
        row["audit_notes"] = "; ".join(problems)
        rows.append(row)

    provenance = {
        "ontology": ontology,
        "result_path": str(result_path.resolve()),
        "label_encoder": str(
            (dp_workspace / "mlb" / f"{ontology}_go.mlb").resolve()
        ),
        "checkpoint_paths": [str(path.resolve()) for path in checkpoint_paths],
    }
    return rows, provenance


def main() -> None:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    output = Path(
        args.output_dir or workspace / "results" / "dpfunc_integration_audit"
    ).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    provenance = []
    for ontology in ONTOLOGIES:
        ontology_rows, ontology_provenance = audit_ontology(
            workspace, ontology, args.pre_name
        )
        rows.extend(ontology_rows)
        provenance.append(ontology_provenance)

    table = pd.DataFrame(rows)
    csv_path = output / "dpfunc_integration_audit.csv"
    json_path = output / "dpfunc_integration_audit.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "workspace": str(workspace),
        "pre_name": args.pre_name,
        "rows": rows,
        "provenance": provenance,
    }, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    display = table[[
        "ontology", "split", "audit_status", "protein_coverage",
        "checkpoint_count", "benchmark_term_coverage",
        "positive_label_mass_coverage", "proteins_with_any_positive_score",
        "score_mean", "score_sd", "score_q50", "score_q99", "score_q100",
        "audit_notes",
    ]]
    print(display.to_string(index=False), flush=True)
    print(f"\nCSV: {csv_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    if (table["audit_status"] == "FAIL").any():
        raise SystemExit("DPFunc integration audit failed; inspect audit_notes")


if __name__ == "__main__":
    main()
