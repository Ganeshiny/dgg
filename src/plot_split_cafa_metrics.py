#!/usr/bin/env python3
"""Plot five-seed CAFA metrics across homology split thresholds.

The 30% baseline is read from ``arc_tuning_cafa``. Alternative thresholds are
read from ``arc_tuning_threshold_<N>``. Missing test evaluations are retained
as explicit gaps, so running this while a split is pending cannot silently mix
validation and test metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THRESHOLDS = (30, 40, 50, 70, 90, 95)
SEEDS = (1103, 2207, 3301, 4409, 5501)
ONTOLOGIES = ("molecular_function", "biological_process", "cellular_component")
ONT_LABELS = {
    "molecular_function": "Molecular Function",
    "biological_process": "Biological Process",
    "cellular_component": "Cellular Component",
}
ONT_COLORS = {
    "molecular_function": "#0072B2",
    "biological_process": "#D55E00",
    "cellular_component": "#009E73",
}
METRICS = {
    "test_micro_f1_at_validation_threshold": "Micro F1 at validation threshold",
    "test_micro_fmax": "Micro Fmax (test-swept, descriptive)",
    "test_macro_fmax": "Macro Fmax (test-swept, descriptive)",
    "test_micro_aupr": "Micro AUPR",
    "test_macro_aupr": "Macro AUPR",
    "test_micro_auroc": "Micro AUROC",
    "test_macro_auroc": "Macro AUROC",
    "test_smin": "Smin (lower is better)",
}
REQUIRED_COLUMNS = {"ontology", "seed", *METRICS}
T_CRITICAL_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
    })


def metric_path(project_root: Path, threshold: int) -> Path:
    root = project_root / ("arc_tuning_cafa" if threshold == 30 else f"arc_tuning_threshold_{threshold}")
    return root / "test_evaluation" / "hybrid" / "per_seed_metrics.csv"


def validate_table(frame: pd.DataFrame, threshold: int, source: Path) -> None:
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{source}: missing columns {sorted(missing_columns)}")
    if frame.duplicated(["ontology", "seed"]).any():
        raise ValueError(f"{source}: duplicate ontology/seed rows")
    observed_ontologies = set(frame["ontology"])
    if observed_ontologies != set(ONTOLOGIES):
        raise ValueError(f"{source}: unexpected ontologies {sorted(observed_ontologies)}")
    for ontology in ONTOLOGIES:
        observed_seeds = set(frame.loc[frame["ontology"] == ontology, "seed"].astype(int))
        if observed_seeds != set(SEEDS):
            raise ValueError(
                f"{source}: threshold {threshold}, {ontology} has seeds "
                f"{sorted(observed_seeds)}; expected {list(SEEDS)}"
            )
    numeric = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{source}: non-finite CAFA metric values")


def load_tables(project_root: Path) -> tuple[pd.DataFrame, dict[int, str], list[int]]:
    frames: list[pd.DataFrame] = []
    sources: dict[int, str] = {}
    missing: list[int] = []
    for threshold in THRESHOLDS:
        source = metric_path(project_root, threshold)
        if not source.is_file():
            missing.append(threshold)
            continue
        frame = pd.read_csv(source)
        validate_table(frame, threshold, source)
        frame.insert(0, "split_threshold", threshold)
        frames.append(frame)
        sources[threshold] = str(source)
    if not frames:
        raise SystemExit("No complete per_seed_metrics.csv files found")
    return pd.concat(frames, ignore_index=True), sources, missing


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    long = frame.melt(
        id_vars=["split_threshold", "ontology", "seed"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )
    summary = (
        long.groupby(["split_threshold", "ontology", "metric"], sort=True)["value"]
        .agg(n="count", mean="mean", sd="std", minimum="min", maximum="max")
        .reset_index()
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n"])
    summary["ci95_half_width"] = [
        T_CRITICAL_975.get(int(n), 1.96) * sem if n > 1 else math.nan
        for n, sem in zip(summary["n"], summary["sem"])
    ]
    summary["ci95_low"] = summary["mean"] - summary["ci95_half_width"]
    summary["ci95_high"] = summary["mean"] + summary["ci95_half_width"]
    summary["relative_sd_percent"] = 100 * summary["sd"] / summary["mean"].abs()
    return summary


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png")
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def format_axis(ax: plt.Axes, metric: str, missing: list[int], bottom_row: bool) -> None:
    ax.set_xticks(THRESHOLDS)
    ax.set_xlim(min(THRESHOLDS) - 3, max(THRESHOLDS) + 3)
    if metric != "test_smin":
        ax.set_ylim(0, 1)
    if bottom_row:
        ax.set_xlabel("Nominal cluster threshold (%)")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    for threshold in missing:
        ax.axvline(threshold, color="#777777", linewidth=0.7, linestyle=":", alpha=0.6)


def plot_metric_grid(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: tuple[str, ...],
    missing: list[int],
    output_dir: Path,
    stem: str,
) -> None:
    fig, axes = plt.subplots(len(metrics), len(ONTOLOGIES), figsize=(12, 3.35 * len(metrics)), squeeze=False)
    seed_offsets = dict(zip(SEEDS, np.linspace(-0.9, 0.9, len(SEEDS))))
    for row_index, metric in enumerate(metrics):
        for col_index, ontology in enumerate(ONTOLOGIES):
            ax = axes[row_index, col_index]
            color = ONT_COLORS[ontology]
            raw = frame[frame["ontology"] == ontology]
            for seed in SEEDS:
                points = raw[raw["seed"].astype(int) == seed]
                ax.scatter(
                    points["split_threshold"] + seed_offsets[seed], points[metric],
                    s=17, color=color, alpha=0.34, edgecolors="none", zorder=2,
                )
            subset = summary[(summary["ontology"] == ontology) & (summary["metric"] == metric)]
            means = {int(row.split_threshold): row for row in subset.itertuples()}
            y = np.asarray([means[t].mean if t in means else np.nan for t in THRESHOLDS])
            ci = np.asarray([means[t].ci95_half_width if t in means else np.nan for t in THRESHOLDS])
            x = np.asarray(THRESHOLDS, dtype=float)
            ax.plot(x, y, color=color, marker="o", linewidth=1.8, markersize=4.5, zorder=3)
            ax.errorbar(x, y, yerr=ci, fmt="none", ecolor=color, capsize=2.5, linewidth=1, zorder=3)
            if row_index == 0:
                ax.set_title(ONT_LABELS[ontology])
            if col_index == 0:
                ax.set_ylabel(METRICS[metric])
            format_axis(ax, metric, missing, row_index == len(metrics) - 1)
    fig.suptitle("Five-seed held-out test performance; points are seeds, bars are 95% t intervals", y=1.005)
    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def plot_seed_stability(summary: pd.DataFrame, output_dir: Path) -> None:
    selected_metrics = (
        "test_micro_f1_at_validation_threshold", "test_micro_fmax", "test_macro_fmax",
        "test_micro_aupr", "test_macro_aupr", "test_smin",
    )
    rows = [(threshold, ontology) for threshold in THRESHOLDS for ontology in ONTOLOGIES]
    matrix = np.full((len(rows), len(selected_metrics)), np.nan)
    lookup = summary.set_index(["split_threshold", "ontology", "metric"])
    for row_index, (threshold, ontology) in enumerate(rows):
        for col_index, metric in enumerate(selected_metrics):
            key = (threshold, ontology, metric)
            if key in lookup.index:
                matrix[row_index, col_index] = float(lookup.loc[key, "relative_sd_percent"])
    finite = matrix[np.isfinite(matrix)]
    upper = max(10.0, float(np.nanpercentile(finite, 95))) if finite.size else 10.0
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    image = ax.imshow(matrix, aspect="auto", cmap="magma_r", vmin=0, vmax=upper)
    ax.set_xticks(range(len(selected_metrics)), [METRICS[m].replace(" (test-swept, descriptive)", "") for m in selected_metrics], rotation=28, ha="right")
    ax.set_yticks(range(len(rows)), [f"{t}% · {ONT_LABELS[o]}" for t, o in rows])
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(col_index, row_index, "pending" if np.isnan(value) else f"{value:.1f}%", ha="center", va="center", fontsize=7)
    bar = fig.colorbar(image, ax=ax, pad=0.015)
    bar.set_label("Relative standard deviation across five seeds (%)")
    ax.set_title("Seed stability by split and ontology (lower is more stable)")
    fig.tight_layout()
    save_figure(fig, output_dir, "03_seed_stability")


def load_similarity_audit(project_root: Path) -> pd.DataFrame:
    path = project_root / "preprocessing" / "data_arc_rebuild_2026_07_14" / "blast_leakage.json"
    if not path.is_file():
        return pd.DataFrame()
    payload = json.loads(path.read_text())
    rows = []
    for threshold in THRESHOLDS:
        if str(threshold) not in payload:
            continue
        item = payload[str(threshold)]
        rows.append({
            "split_threshold": threshold,
            "fraction_ge_60": 100 * float(item["fraction_at_or_above_60_percent"]),
            "fraction_ge_nominal_80cov": 100 * float(item["fraction_at_or_above_threshold_and_80pct_coverage"]),
            "mean_max_identity": float(item["mean_max_identity_percent"]),
        })
    return pd.DataFrame(rows)


def plot_homology_relation(summary: pd.DataFrame, audit: pd.DataFrame, output_dir: Path) -> None:
    if audit.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex="col")
    for col_index, ontology in enumerate(ONTOLOGIES):
        color = ONT_COLORS[ontology]
        for row_index, metric in enumerate(("test_micro_f1_at_validation_threshold", "test_micro_fmax")):
            ax = axes[row_index, col_index]
            perf = summary[(summary["ontology"] == ontology) & (summary["metric"] == metric)]
            merged = audit.merge(perf, on="split_threshold", how="inner")
            ax.errorbar(
                merged["fraction_ge_60"], merged["mean"], yerr=merged["ci95_half_width"],
                fmt="o", color=color, ecolor=color, capsize=2.5,
            )
            for point in merged.itertuples():
                ax.annotate(f"{int(point.split_threshold)}%", (point.fraction_ge_60, point.mean), xytext=(4, 4), textcoords="offset points", fontsize=7)
            ax.set_ylim(0, 1)
            if row_index == 0:
                ax.set_title(ONT_LABELS[ontology])
            if col_index == 0:
                ax.set_ylabel(METRICS[metric])
            if row_index == 1:
                ax.set_xlabel("Test proteins with ≥60% train identity (%)")
    fig.suptitle("Performance versus measured test-to-train similarity; labels are nominal split thresholds", y=1.005)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_actual_homology_vs_performance")


def load_split_context(project_root: Path) -> pd.DataFrame:
    root = project_root / "preprocessing" / "data_arc_rebuild_2026_07_14" / "pdb_splits"
    rows = []
    for threshold in THRESHOLDS:
        path = root / f"threshold_{threshold}" / "split_log.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        for ontology in ONTOLOGIES:
            test = payload["go_coverage"]["test"][ontology]
            train = payload["go_coverage"]["train"][ontology]
            rows.append({
                "split_threshold": threshold,
                "ontology": ontology,
                "test_labelled_proteins": int(test["n_proteins_with_labels"]),
                "test_unique_terms": int(test["n_unique_terms"]),
                "train_unique_terms": int(train["n_unique_terms"]),
            })
    return pd.DataFrame(rows)


def plot_split_context(context: pd.DataFrame, output_dir: Path) -> None:
    if context.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex="col")
    for col_index, ontology in enumerate(ONTOLOGIES):
        subset = context[context["ontology"] == ontology].sort_values("split_threshold")
        color = ONT_COLORS[ontology]
        for row_index, (column, label) in enumerate((
            ("test_labelled_proteins", "Labelled test proteins"),
            ("test_unique_terms", "Unique GO terms in test"),
        )):
            ax = axes[row_index, col_index]
            ax.plot(subset["split_threshold"], subset[column], marker="o", color=color, linewidth=1.7)
            if row_index == 0:
                ax.set_title(ONT_LABELS[ontology])
            if col_index == 0:
                ax.set_ylabel(label)
            if row_index == 1:
                ax.set_xlabel("Nominal cluster threshold (%)")
            ax.set_xticks(THRESHOLDS)
    fig.suptitle("Changing test-set label composition across independently rebuilt splits", y=1.005)
    fig.tight_layout()
    save_figure(fig, output_dir, "05_split_label_context")


def write_readme(output_dir: Path, available: list[int], missing: list[int]) -> None:
    text = f"""# Homology-split CAFA plots

Available held-out test splits: {', '.join(f'{x}%' for x in available)}

Missing held-out test splits: {', '.join(f'{x}%' for x in missing) if missing else 'none'}

The primary leakage-safe classification metric is micro F1 evaluated at the
validation-selected threshold. Fmax values sweep thresholds on the test set and
are therefore labelled descriptive. Smin is lower-is-better. Points represent
the five fixed seeds; intervals are 95% t intervals over those seeds.

The nominal split threshold is a clustering setting, not the measured identity
of every test protein to training. Figure 04 uses the independent BLAST audit.
The split memberships and GO-label composition change across thresholds, so
cross-split trends are descriptive rather than a controlled causal effect of
homology.

Regenerate from the repository root:

    python3 src/plot_split_cafa_metrics.py

Use `--require-complete` after every threshold test evaluation has finished.
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (args.output_dir or project_root / "plots" / "split_homology_cafa").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    frame, sources, missing = load_tables(project_root)
    if args.require_complete and missing:
        raise SystemExit(f"Missing held-out test evaluations for thresholds: {missing}")
    summary = summarise(frame)
    frame.to_csv(output_dir / "per_seed_metrics_combined.csv", index=False)
    summary.to_csv(output_dir / "cafa_metrics_summary.csv", index=False)

    plot_metric_grid(
        frame, summary,
        ("test_micro_f1_at_validation_threshold", "test_micro_fmax", "test_macro_fmax"),
        missing, output_dir, "01_f1_and_fmax_by_split",
    )
    plot_metric_grid(
        frame, summary,
        ("test_micro_aupr", "test_macro_aupr", "test_smin"),
        missing, output_dir, "02_aupr_and_smin_by_split",
    )
    plot_seed_stability(summary, output_dir)
    audit = load_similarity_audit(project_root)
    plot_homology_relation(summary, audit, output_dir)
    context = load_split_context(project_root)
    plot_split_context(context, output_dir)

    available = sorted(int(value) for value in frame["split_threshold"].unique())
    manifest = {
        "available_test_thresholds": available,
        "missing_test_thresholds": missing,
        "sources": sources,
        "rows_per_threshold": {
            str(threshold): int((frame["split_threshold"] == threshold).sum())
            for threshold in available
        },
        "expected_rows_per_threshold": len(SEEDS) * len(ONTOLOGIES),
        "figures": [
            "01_f1_and_fmax_by_split",
            "02_aupr_and_smin_by_split",
            "03_seed_stability",
            "04_actual_homology_vs_performance",
            "05_split_label_context",
        ],
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_readme(output_dir, available, missing)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
