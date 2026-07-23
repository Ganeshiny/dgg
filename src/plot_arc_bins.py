#!/usr/bin/env python3
"""Plot homology/IC-bin metrics with explicit support and uncertainty audits.

The script refuses to treat missing bins as absent data. It reconstructs raw
test-bin support from the split labels, reports one-class AUROC cells, masks
archived evaluator zeros that cannot be interpreted, and plots mean +/- SD
across the five checkpoint seeds without connecting unsupported points.
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_style import (
    BIN_AXIS_LABEL,
    BIN_ORDER,
    DOUBLE_COLUMN_IN,
    ERROR_KIND_LABEL,
    METRIC_HIGHER_IS_BETTER,
    METRIC_LABEL,
    METRIC_ORDER,
    MODEL_COLOR,
    MODEL_MARKER,
    MODEL_ORDER,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    VARIANT_LABEL,
    VARIANT_ORDER,
    annotate_insufficient_data,
    apply_style,
    colorblind_audit,
    label_panel,
    mean_and_error,
    savefig,
)

DEFAULT_DATA_ROOT = Path("preprocessing/data_arc_rebuild_2026_07_14/datasets")
DEFAULT_HOMOLOGY = Path("preprocessing/data_arc_rebuild_2026_07_14/pdb_splits/threshold_30/blast_te_vs_tr.tsv")


def _records(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, list):
        return obj
    if hasattr(obj, "protein_ids") and hasattr(obj, "labels") and hasattr(obj, "terms"):
        return [
            {"id": pid, "labels": [term for term, value in zip(obj.terms, row) if value > 0]}
            for pid, row in zip(obj.protein_ids, obj.labels)
        ]
    raise TypeError(f"Unsupported dataset pickle: {path}")


def _threshold_from_label(frame: pd.DataFrame) -> int:
    for value in frame.get("split_label", pd.Series(dtype=str)).dropna().astype(str):
        match = re.search(r"(?:threshold_|nominal_)(\d+)", value)
        if match:
            return int(match.group(1))
    return 30


def _homology_groups(path: Path, ids: list[str]) -> dict[str, str]:
    maximum = {protein: 0.0 for protein in ids}
    if path.is_file():
        with path.open() as handle:
            for line in handle:
                fields = line.rstrip().split("\t")
                if len(fields) >= 3 and fields[0] in maximum:
                    try:
                        maximum[fields[0]] = max(maximum[fields[0]], float(fields[2]))
                    except ValueError:
                        continue
    groups = {}
    for protein, value in maximum.items():
        groups[protein] = (
            "no_hit" if value <= 0 else "<30%" if value < 30 else
            "30-40%" if value < 40 else "40-60%" if value < 60 else ">=60%"
        )
    return groups


def _ic_groups(test: list[dict], train: list[dict]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for row in train:
        for term in set(row.get("labels", [])):
            counts[term] = counts.get(term, 0) + 1
    n_train = max(len(train), 1)
    ic = {term: -np.log2(count / n_train) for term, count in counts.items() if count > 0}
    result = {}
    for row in test:
        values = [ic[term] for term in set(row.get("labels", [])) if term in ic]
        value = max(values) if values else np.nan
        result[row["id"]] = (
            "no_positive_terms" if not np.isfinite(value) else "<2_bits" if value < 2 else
            "2-4_bits" if value < 4 else "4-6_bits" if value < 6 else ">=6_bits"
        )
    return result


def build_integrity_audit(dataset_root: Path, homology_path: Path, threshold: int) -> pd.DataFrame:
    rows = []
    for ontology in ONTOLOGY_ORDER:
        train_path = dataset_root / f"threshold_{threshold}" / f"{ontology}_train.pkl"
        test_path = dataset_root / f"threshold_{threshold}" / f"{ontology}_test.pkl"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test labels for {ontology}: {test_path}")
        train = _records(train_path) if train_path.exists() else []
        test = _records(test_path)
        term_vocab = sorted(set(term for row in train + test for term in row.get("labels", [])))
        groups_by_type = {
            "homology": _homology_groups(homology_path, [row["id"] for row in test]),
            "ic": _ic_groups(test, train),
        }
        for bin_type, group_map in groups_by_type.items():
            for group in BIN_ORDER[bin_type]:
                selected = [row for row in test if group_map.get(row["id"]) == group]
                matrix = np.asarray([[term in set(row.get("labels", [])) for term in term_vocab] for row in selected], dtype=int)
                if not selected:
                    positive_terms = valid_terms = all_positive_terms = positive_assignments = 0
                else:
                    counts = matrix.sum(axis=0)
                    positive_terms = int(np.sum(counts > 0))
                    valid_terms = int(np.sum((counts > 0) & (counts < len(selected))))
                    all_positive_terms = int(np.sum(counts == len(selected)))
                    positive_assignments = int(matrix.sum())
                rows.append({
                    "ontology": ontology,
                    "bin_type": bin_type,
                    "bin": group,
                    "examples": len(selected),
                    "positive_examples": int(sum(bool(row.get("labels")) for row in selected)),
                    "positive_term_assignments": positive_assignments,
                    "positive_terms": positive_terms,
                    "valid_auroc_terms": valid_terms,
                    "all_positive_terms": all_positive_terms,
                    "zero_positive_terms": max(len(term_vocab) - positive_terms, 0),
                    "support_status": "empty" if not selected else "low_support" if len(selected) < 10 else "ok",
                })
    return pd.DataFrame(rows)


def _audit_status(row: pd.Series) -> str:
    if int(row.get("examples", 0)) == 0:
        return "no data"
    if int(row.get("positive_terms", 0)) == 0:
        return "no positive terms"
    if int(row.get("valid_auroc_terms", 0)) == 0:
        return "AUROC undefined: one-class terms"
    if int(row.get("examples", 0)) < 10:
        return "low support"
    return "ok"


def merge_and_validate(frame: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    keys = ["ontology", "bin_type", "bin"]
    merged = frame.merge(audit, on=keys, how="left", suffixes=("", "_audit"))
    if merged["examples_audit"].isna().any():
        missing = merged.loc[merged["examples_audit"].isna(), keys].drop_duplicates()
        raise RuntimeError(f"Missing raw-N audit rows for bin cells:\n{missing.to_string(index=False)}")
    merged["audit_status"] = merged.apply(_audit_status, axis=1)
    # The old evaluator returned 0 for a whole macro-AUROC bin when any term
    # was one-class. That value is an evaluator artifact, not a model score.
    bad_macro_auroc = (
        (merged["Macro_AUROC"].abs() <= 1e-12) &
        (merged["valid_auroc_terms"] > 0) &
        (merged["positive_terms"] > 0)
    )
    merged.loc[bad_macro_auroc, "Macro_AUROC"] = np.nan
    merged.loc[bad_macro_auroc, "audit_status"] = "archived AUROC evaluator bug; rerun required"
    merged.loc[merged["valid_auroc_terms"] == 0, "Macro_AUROC"] = np.nan
    merged.loc[merged["positive_terms"] == 0, ["Macro_AUPRC", "Macro_AUROC"]] = np.nan
    return merged


def _size_for_n(n: float, max_n: float) -> float:
    if not np.isfinite(n) or n <= 0:
        return 0.0
    return 22.0 + 58.0 * np.sqrt(n / max(max_n, 1.0))


def _metric_limits(frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    if metric == "Smin":
        return 0.0, max(1.0, hi * 1.10)
    pad = max((hi - lo) * .08, .02)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _marker_legend(max_n: float) -> list[Line2D]:
    refs = sorted(set([5.0, 10.0, max(10.0, min(max_n, 50.0))]))
    return [Line2D([0], [0], linestyle="", marker="o", color="#555555",
                   markersize=np.sqrt(_size_for_n(n, max_n)), label=f"n={int(n)}")
            for n in refs]


def plot_bin_grid(frame: pd.DataFrame, out: Path, bin_type: str, order: list[str],
                  metric: str, min_n: int, err_kind: str) -> None:
    higher_better = METRIC_HIGHER_IS_BETTER[metric]
    ymin, ymax = _metric_limits(frame, metric)
    fig, axes = plt.subplots(
        len(VARIANT_ORDER), len(ONTOLOGY_ORDER),
        figsize=(DOUBLE_COLUMN_IN, 2.65 * len(VARIANT_ORDER)),
        sharex=True, sharey=True,
    )
    axes = np.asarray(axes)
    x = np.arange(len(order), dtype=float)
    max_n = float(frame["examples_audit"].max())
    any_data = False
    panel = 0
    for row, variant in enumerate(VARIANT_ORDER):
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row, col]
            sub = frame[(frame.ontology == ontology) & (frame.input_modality == variant)]
            observed_bins = set(sub["bin"].dropna().astype(str))
            for index, label in enumerate(order):
                if label not in observed_bins:
                    ax.text(x[index], ymin + .04 * (ymax - ymin), "no data", ha="center",
                            va="bottom", rotation=90, fontsize=4.5, color="#777777")
            if sub.empty:
                annotate_insufficient_data(ax)
            for model in MODEL_ORDER:
                group = sub[sub.model == model]
                if group.empty:
                    continue
                any_data = True
                stats = group.groupby("bin", observed=False)[metric].agg(["mean", "std", "count"]).reindex(order)
                audit_rows = group.groupby("bin", observed=False)["examples_audit"].first().reindex(order)
                status_rows = group.groupby("bin", observed=False)["audit_status"].first().reindex(order)
                for index, label in enumerate(order):
                    n = float(audit_rows.iloc[index]) if pd.notna(audit_rows.iloc[index]) else 0.0
                    mean = stats["mean"].iloc[index]
                    err = stats["std"].iloc[index] if stats["count"].iloc[index] > 1 else 0.0
                    if n <= 0 or not np.isfinite(mean):
                        ax.text(x[index], ymin + .04 * (ymax - ymin), "no data", ha="center",
                                va="bottom", rotation=90, fontsize=4.5, color="#777777")
                        continue
                    marker = MODEL_MARKER[model]
                    ax.errorbar(x[index], mean, yerr=err, fmt="none", ecolor=MODEL_COLOR[model],
                                elinewidth=.7, capsize=1.8, zorder=3)
                    ax.scatter([x[index]], [mean], s=_size_for_n(n, max_n),
                               color=MODEL_COLOR[model], marker=marker, edgecolor="#111111",
                               linewidth=.25, zorder=4)
                    if n < min_n or "bug" in str(status_rows.iloc[index]):
                        ax.annotate("*", (x[index], mean), xytext=(3, 3),
                                    textcoords="offset points", fontsize=6, fontweight="bold")
            ax.set_ylim(ymin, ymax)
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            if col == 0:
                ax.set_ylabel(f"{VARIANT_LABEL[variant]}\n{METRIC_LABEL[metric]}", fontsize=7)
            if row == len(VARIANT_ORDER) - 1:
                ax.set_xticks(x, order, rotation=35, ha="right")
            label_panel(ax, chr(97 + panel))
            ax.set_xlim(-.5, len(order) - .5)
            panel += 1
    if not any_data:
        plt.close(fig)
        return
    fig.supxlabel(BIN_AXIS_LABEL[bin_type], fontsize=8)
    direction = "higher is better" if higher_better else "lower is better"
    fig.text(.5, -.035, f"Error bars = {ERROR_KIND_LABEL[err_kind]} across five seed checkpoints; "
             f"marker area ∝ sqrt(test examples); no connecting lines; '*' = n<{min_n} or archived "
             f"one-class-AUROC evaluator artifact; empty bins are explicit no-data gaps; {direction}.",
             ha="center", fontsize=5.6)
    model_handles = [
        Line2D([0], [0], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], linestyle="",
               markersize=5, label=m) for m in MODEL_ORDER
    ]
    legend = fig.legend(handles=model_handles, title="Model", loc="upper left",
                        bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6.5)
    fig.add_artist(legend)
    fig.legend(handles=_marker_legend(max_n), title="Test examples", loc="upper left",
               bbox_to_anchor=(1.0, .62), frameon=False, fontsize=6.5)
    savefig(fig, out / f"{bin_type}_{metric.lower()}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("plots/arc_tuning_cafa/bin_evaluation"))
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--homology-tsv", type=Path, default=DEFAULT_HOMOLOGY)
    ap.add_argument("--metrics", nargs="+", default=METRIC_ORDER, choices=METRIC_ORDER)
    ap.add_argument("--min-n", type=int, default=10, help="Flag points with fewer than this many test examples.")
    ap.add_argument("--err", choices=["sd", "sem", "ci95"], default="sd")
    args = ap.parse_args()
    apply_style()
    print("Colour audit:", colorblind_audit())
    source = args.bin_csv.resolve()
    frame = pd.read_csv(source)
    required = {"ontology", "model", "input_modality", "bin_type", "bin", "examples"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing required bin columns: {sorted(missing)}")
    threshold = _threshold_from_label(frame)
    dataset_root = args.dataset_root.resolve()
    audit = build_integrity_audit(dataset_root, args.homology_tsv.resolve(), threshold)
    merged = merge_and_validate(frame, audit)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "bin_integrity_audit.csv", index=False)
    merged.to_csv(out / "bin_metrics.csv", index=False)
    flagged = merged[(merged["Macro_AUROC"].isna()) | (merged["Macro_AUPRC"].abs() <= 1e-12)]
    if not flagged.empty:
        print("Integrity flags (deduplicated by ontology/bin):")
        print(flagged[["ontology", "bin_type", "bin", "examples_audit", "positive_examples",
                       "positive_term_assignments", "positive_terms", "valid_auroc_terms",
                       "audit_status"]].drop_duplicates().to_string(index=False))
    found_bin_types = set(merged["bin_type"].dropna().unique())
    for bin_type in [value for value in BIN_ORDER if value in found_bin_types]:
        order = BIN_ORDER[bin_type]
        subset = merged[merged.bin_type == bin_type].copy()
        subset["bin"] = pd.Categorical(subset["bin"], categories=order, ordered=True)
        for metric in args.metrics:
            if metric in subset:
                plot_bin_grid(subset, out, bin_type, order, metric, args.min_n, args.err)
    print(f"Wrote audited bin metrics and plots to {out}")


if __name__ == "__main__":
    main()
