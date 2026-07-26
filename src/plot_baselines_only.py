#!/usr/bin/env python3
"""Create publication-ready baseline-only figures from an ARC benchmark.

DeepGreenGO and its seed-specific predictions are deliberately excluded. The
script plots every completed comparator present in benchmark_metrics.csv and
therefore also picks up SOTA methods after a later complete benchmark run.

Examples
--------
python src/plot_baselines_only.py
python src/plot_baselines_only.py --journal nature
python src/plot_baselines_only.py --workspace arc_benchmark/nominal_30_identity_80_coverage
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_DIR / "arc_benchmark" / "nominal_30_identity_80_coverage"

# Keep the journal selection ahead of plot_style's import-time configuration.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--journal", choices=("bmc", "nature"), default=None)
_known, _ = _parser.parse_known_args()
if _known.journal:
    os.environ["DGG_JOURNAL"] = _known.journal

from plot_style import (  # noqa: E402
    CATEGORICAL_PALETTE,
    DOUBLE_COLUMN_IN,
    JOURNAL,
    MAIN,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    SPEC,
    apply_style,
    check_min_font,
    label_panel,
    savefig,
)


METHOD_ORDER = [
    "naive",
    "blast", "blast_max",
    "diamond", "diamond_max",
    "foldseek", "foldseek_max",
    "interproscan",
    "deepfri_sequence", "deepfri_structure",
    "dpfunc", "deepgoplus", "deepgose", "transfun",
    "eggnog_mapper", "hayai", "gomap",
]

METHOD_LABEL = {
    "naive": "Naive frequency",
    "blast": "BLAST (top-10)",
    "blast_max": "BLAST (max identity)",
    "diamond": "DIAMOND (top-10)",
    "diamond_max": "DIAMOND (max identity)",
    "foldseek": "Foldseek (top-10)",
    "foldseek_max": "Foldseek (max identity)",
    "interproscan": "InterProScan",
    "deepfri_sequence": "DeepFRI (sequence)",
    "deepfri_structure": "DeepFRI (structure)",
    "dpfunc": "DPFunc",
    "deepgoplus": "DeepGOPlus",
    "deepgose": "DeepGO-SE",
    "transfun": "TransFun",
    "eggnog_mapper": "eggNOG-mapper",
    "hayai": "Hayai",
    "gomap": "GOMAP",
}

METHOD_FAMILY = {
    "naive": "frequency",
    "blast": "sequence", "blast_max": "sequence",
    "diamond": "sequence", "diamond_max": "sequence",
    "foldseek": "structure", "foldseek_max": "structure",
    "interproscan": "domain",
    "deepfri_sequence": "deep_learning", "deepfri_structure": "deep_learning",
    "dpfunc": "deep_learning", "deepgoplus": "deep_learning",
    "deepgose": "deep_learning", "transfun": "deep_learning",
    "eggnog_mapper": "orthology", "hayai": "orthology", "gomap": "orthology",
}

FAMILY_COLOR = {
    "frequency": CATEGORICAL_PALETTE[2],
    "sequence": CATEGORICAL_PALETTE[0],
    "structure": CATEGORICAL_PALETTE[4],
    "domain": CATEGORICAL_PALETTE[7],
    "deep_learning": CATEGORICAL_PALETTE[3],
    "orthology": CATEGORICAL_PALETTE[5],
    "other": CATEGORICAL_PALETTE[6],
}

FAMILY_LABEL = {
    "frequency": "Frequency prior",
    "sequence": "Sequence alignment",
    "structure": "Structure alignment",
    "domain": "Domain annotation",
    "deep_learning": "External deep learning",
    "orthology": "Orthology/annotation pipeline",
    "other": "Other comparator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <workspace>/plots/baselines_only",
    )
    parser.add_argument("--journal", choices=("bmc", "nature"), default=None)
    return parser.parse_args()


def load_baselines(workspace: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = workspace / "results" / "benchmark_metrics.csv"
    bootstrap_path = workspace / "results" / "bootstrap_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing benchmark metrics: {metrics_path}")
    if not bootstrap_path.is_file():
        raise FileNotFoundError(f"Missing bootstrap metrics: {bootstrap_path}")

    metrics = pd.read_csv(metrics_path)
    bootstrap = pd.read_csv(bootstrap_path)
    required = {
        "method", "ontology", "cafa_fmax", "cafa_smin",
        "micro_aupr", "macro_aupr", "protein_coverage_any_score",
        "predicted_term_coverage", "test_terms",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"benchmark_metrics.csv lacks columns: {sorted(missing)}")

    # Exclude the proposed model and every seed/split-specific variant by name.
    baseline_mask = ~metrics["method"].astype(str).str.startswith("deepgreengo")
    boot_mask = ~bootstrap["method"].astype(str).str.startswith("deepgreengo")
    metrics = metrics.loc[baseline_mask].copy()
    bootstrap = bootstrap.loc[boot_mask].copy()
    if metrics.empty:
        raise ValueError("No completed baselines are present in benchmark_metrics.csv")
    if metrics.duplicated(["method", "ontology"]).any():
        duplicated = metrics.loc[
            metrics.duplicated(["method", "ontology"], keep=False),
            ["method", "ontology"],
        ]
        raise ValueError(f"Duplicate method/ontology rows:\n{duplicated}")

    expected = set(ONTOLOGY_ORDER)
    incomplete = {
        method: sorted(expected - set(group["ontology"]))
        for method, group in metrics.groupby("method")
        if set(group["ontology"]) != expected
    }
    if incomplete:
        raise ValueError(f"Incomplete baseline ontology results: {incomplete}")
    return metrics, bootstrap


def ordered_methods(metrics: pd.DataFrame) -> list[str]:
    present = set(metrics["method"].astype(str))
    known = [method for method in METHOD_ORDER if method in present]
    return known + sorted(present - set(known))


def family(method: str) -> str:
    return METHOD_FAMILY.get(method, "other")


def marker(method: str) -> str:
    if method.endswith("_max"):
        return "D"
    if method == "naive":
        return "s"
    return "o"


def marker_face(method: str, color: str) -> str:
    return "white" if method.endswith("_max") else color


def style_axis(ax: plt.Axes, ontology: str, xlabel: str, panel: str) -> None:
    ax.set_title(ONTOLOGY_SHORT[ontology])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    label_panel(ax, panel)



def assert_print_fonts(fig: plt.Figure) -> None:
    warnings = check_min_font(fig, MAIN)
    if warnings:
        raise ValueError("; ".join(warnings))


def plot_cafa(metrics: pd.DataFrame, methods: list[str], out: Path) -> None:
    fig, axes = plt.subplots(
        2, 3,
        figsize=(DOUBLE_COLUMN_IN, max(4.8, 0.38 * len(methods) + 2.4)),
        sharey=True,
    )
    y = np.arange(len(methods))
    panel_index = 0
    specifications = [
        ("cafa_fmax", "cafa_fmax_ci_low", "cafa_fmax_ci_high", "CAFA F$_{max}$"),
        ("cafa_smin", "cafa_smin_ci_low", "cafa_smin_ci_high", "CAFA S$_{min}$"),
    ]
    for row_index, (value_col, low_col, high_col, xlabel) in enumerate(specifications):
        for column_index, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row_index, column_index]
            subset = metrics[metrics["ontology"] == ontology].set_index("method")
            upper_values = []
            for yi, method in enumerate(methods):
                record = subset.loc[method]
                value = float(record[value_col])
                low = float(record[low_col])
                high = float(record[high_col])
                upper_values.append(high)
                color = FAMILY_COLOR[family(method)]
                ax.errorbar(
                    value, yi,
                    xerr=np.array([[max(0.0, value - low)], [max(0.0, high - value)]]),
                    fmt=marker(method), markersize=5.2,
                    markerfacecolor=marker_face(method, color),
                    markeredgecolor=color, markeredgewidth=1.0,
                    ecolor=color, elinewidth=1.0, capsize=2.2, capthick=0.8,
                    zorder=3,
                )
            maximum = max(upper_values)
            ax.set_xlim(0, maximum * 1.10 if maximum > 0 else 1)
            ax.set_yticks(y, [METHOD_LABEL.get(method, method) for method in methods])
            style_axis(ax, ontology, xlabel, chr(97 + panel_index))
            panel_index += 1
    # All panels share y; invert exactly once so METHOD_ORDER is top-to-bottom.
    axes[0, 0].invert_yaxis()
    fig.subplots_adjust(left=0.19, bottom=0.10, hspace=0.42, wspace=0.28)
    assert_print_fonts(fig)
    savefig(fig, out / "baseline_cafa_performance", MAIN)


def plot_aupr(metrics: pd.DataFrame, methods: list[str], out: Path) -> None:
    fig, axes = plt.subplots(
        2, 3,
        figsize=(DOUBLE_COLUMN_IN, max(4.8, 0.38 * len(methods) + 2.4)),
        sharey=True,
    )
    y = np.arange(len(methods))
    panel_index = 0
    for row_index, (column, xlabel) in enumerate((
        ("micro_aupr", "Micro-AUPR"),
        ("macro_aupr", "Macro-AUPR"),
    )):
        for column_index, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row_index, column_index]
            subset = metrics[metrics["ontology"] == ontology].set_index("method")
            values = []
            for yi, method in enumerate(methods):
                value = float(subset.loc[method, column])
                values.append(value)
                color = FAMILY_COLOR[family(method)]
                ax.plot(
                    value, yi, marker=marker(method), linestyle="none", markersize=5.5,
                    markerfacecolor=marker_face(method, color), markeredgecolor=color,
                    markeredgewidth=1.0, zorder=3,
                )
            maximum = max(values)
            ax.set_xlim(0, maximum * 1.12 if maximum > 0 else 1)
            ax.set_yticks(y, [METHOD_LABEL.get(method, method) for method in methods])
            style_axis(ax, ontology, xlabel, chr(97 + panel_index))
            panel_index += 1
    # All panels share y; invert exactly once so METHOD_ORDER is top-to-bottom.
    axes[0, 0].invert_yaxis()
    fig.subplots_adjust(left=0.19, bottom=0.10, hspace=0.42, wspace=0.28)
    assert_print_fonts(fig)
    savefig(fig, out / "baseline_aupr", MAIN)


def plot_coverage(metrics: pd.DataFrame, methods: list[str], out: Path) -> None:
    plot_data = metrics.copy()
    plot_data["protein_coverage_percent"] = 100 * plot_data["protein_coverage_any_score"]
    plot_data["term_coverage_percent"] = (
        100 * plot_data["predicted_term_coverage"] / plot_data["test_terms"]
    )
    fig, axes = plt.subplots(
        2, 3,
        figsize=(DOUBLE_COLUMN_IN, max(4.8, 0.38 * len(methods) + 2.4)),
        sharey=True,
    )
    y = np.arange(len(methods))
    panel_index = 0
    for row_index, (column, xlabel) in enumerate((
        ("protein_coverage_percent", "Proteins covered (%)"),
        ("term_coverage_percent", "GO terms covered (%)"),
    )):
        for column_index, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row_index, column_index]
            subset = plot_data[plot_data["ontology"] == ontology].set_index("method")
            values = np.array([float(subset.loc[method, column]) for method in methods])
            colors = [FAMILY_COLOR[family(method)] for method in methods]
            bars = ax.barh(
                y, values, height=0.66, color=colors,
                edgecolor="#222222", linewidth=0.45,
            )
            for bar, method in zip(bars, methods):
                if method.endswith("_max"):
                    bar.set_hatch("//")
            for yi, value in enumerate(values):
                ax.text(
                    min(value + 1.4, 97.0), yi, f"{value:.1f}",
                    va="center", ha="left" if value < 92 else "right",
                )
            ax.set_xlim(0, 105)
            ax.set_yticks(y, [METHOD_LABEL.get(method, method) for method in methods])
            style_axis(ax, ontology, xlabel, chr(97 + panel_index))
            panel_index += 1
    # All panels share y; invert exactly once so METHOD_ORDER is top-to-bottom.
    axes[0, 0].invert_yaxis()
    fig.subplots_adjust(left=0.19, bottom=0.10, hspace=0.42, wspace=0.28)
    assert_print_fonts(fig)
    savefig(fig, out / "baseline_prediction_coverage", MAIN)


def write_supporting_files(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    methods: list[str],
    workspace: Path,
    out: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "baseline_metrics_plotted.csv", index=False)
    bootstrap.to_csv(out / "baseline_bootstrap_plotted.csv", index=False)
    captions = """Unless noted, higher values indicate better performance for all metrics except CAFA Smin, where lower is better.

baseline_cafa_performance
Baseline-only CAFA performance on the nominal 30%-identity/80%-coverage test split (n = 754 proteins). Points show test-set CAFA Fmax and Smin; error bars show percentile 95% confidence intervals from 1,000 paired protein-level bootstrap replicates. DeepGreenGO is excluded.

baseline_aupr
Baseline-only area-under-the-precision-recall-curve comparison on the same test split. Micro-AUPR pools protein-term decisions; macro-AUPR averages per-term average precision across GO terms observed in the test set. These are test-set point estimates. DeepGreenGO is excluded.

baseline_prediction_coverage
Baseline prediction coverage on the same test split. Protein coverage is the percentage of test proteins receiving at least one nonzero score; term coverage is the percentage of evaluated test GO terms receiving at least one nonzero score. Coverage is not an accuracy measure. DeepGreenGO is excluded.
"""
    (out / "captions.txt").write_text(captions, encoding="utf-8")
    manifest = {
        "source_workspace": str(workspace.resolve()),
        "journal_profile": JOURNAL,
        "source_metrics": "results/benchmark_metrics.csv",
        "source_bootstrap": "results/bootstrap_metrics.csv",
        "excluded_method_prefixes": ["deepgreengo"],
        "included_methods": methods,
        "ontologies": ONTOLOGY_ORDER,
        "outputs": [
            f"{stem}.{suffix}"
            for stem in (
                "baseline_cafa_performance",
                "baseline_aupr",
                "baseline_prediction_coverage",
            )
            for suffix in ("pdf", SPEC["raster"])
        ],
    }
    (out / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    output = (args.output or workspace / "plots" / "baselines_only").expanduser().resolve()
    apply_style()
    metrics, bootstrap = load_baselines(workspace)
    methods = ordered_methods(metrics)
    plot_cafa(metrics, methods, output)
    plot_aupr(metrics, methods, output)
    plot_coverage(metrics, methods, output)
    write_supporting_files(metrics, bootstrap, methods, workspace, output)
    print(f"Plotted {len(methods)} baselines: {', '.join(methods)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
