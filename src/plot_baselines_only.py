#!/usr/bin/env python3
"""Create publication-ready DeepGreenGO-versus-baseline figures.

The benchmark's five-seed DeepGreenGO ensemble is the focal method. The script
plots it with every completed comparator present in benchmark_metrics.csv.

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

from matplotlib.patches import Patch


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
    "deepgreengo",
    "naive",
    "blast", "blast_max",
    "diamond", "diamond_max",
    "foldseek", "foldseek_max",
    "interproscan",
    "deepfri_sequence", "deepfri_structure",
    "dpfunc", "heal", "gat_go", "deepgraphgo", "deepgoplus", "deepgose", "transfun",
    "eggnog_mapper", "hayai", "gomap",
]

COVERAGE_METHOD_ORDER = [
    "blast", "blast_max",
    "diamond", "diamond_max",
    "foldseek", "foldseek_max",
    "interproscan", "eggnog_mapper", "hayai", "gomap",
]


METHOD_LABEL = {
    "deepgreengo": "DeepGreenGO Hybrid (this work)",
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
    "heal": "HEAL (PDB-only)",
    "gat_go": "GAT-GO",
    "deepgraphgo": "DeepGraphGO",
    "deepgoplus": "DeepGOPlus",
    "deepgose": "DeepGO-SE",
    "transfun": "TransFun",
    "eggnog_mapper": "eggNOG-mapper",
    "hayai": "Hayai",
    "gomap": "GOMAP",
}

# Comparators withheld from every figure produced by this module. Their
# predictions and metrics are still computed and remain in
# benchmark_metrics.csv; only the plotted comparator set is narrowed. Label,
# colour, and family entries are kept above so the choice is reversible by
# editing this one set.
EXCLUDED_FROM_PLOTS = frozenset({"deepgoplus", "deepgose"})

EXTERNAL_PRETRAINED_METHODS = (
    "deepfri_sequence",
    "deepfri_structure",
    "dpfunc",
    "heal",
    "gat_go",
    "deepgraphgo",
)

METHOD_FAMILY = {
    "deepgreengo": "proposed",
    "naive": "frequency",
    "blast": "sequence", "blast_max": "sequence",
    "diamond": "sequence", "diamond_max": "sequence",
    "foldseek": "structure", "foldseek_max": "structure",
    "interproscan": "domain",
    "deepfri_sequence": "deep_learning", "deepfri_structure": "deep_learning",
    "dpfunc": "deep_learning", "heal": "deep_learning",
    "gat_go": "deep_learning", "deepgraphgo": "deep_learning",
    "deepgoplus": "deep_learning",
    "deepgose": "deep_learning", "transfun": "deep_learning",
    "eggnog_mapper": "orthology", "hayai": "orthology", "gomap": "orthology",
}

FAMILY_COLOR = {
    "proposed": "#1B5E20",
    "frequency": CATEGORICAL_PALETTE[2],
    "sequence": CATEGORICAL_PALETTE[0],
    "structure": CATEGORICAL_PALETTE[4],
    "domain": CATEGORICAL_PALETTE[7],
    "deep_learning": CATEGORICAL_PALETTE[3],
    "orthology": CATEGORICAL_PALETTE[5],
    "other": "#777777",
}

# Stable method-level colours used by every comparison and stratified plot.
# External deep-learning methods deliberately do not share a family colour:
# overlapping curves must remain identifiable without tiny marker differences.
METHOD_COLOR = {
    "deepgreengo": "#006D2C",
    "naive": "#DAA520",
    "blast": "#0000CD",
    "blast_max": "#DC143C",
    "diamond": "#4682B4",
    "diamond_max": "#DA70D6",
    "foldseek": "#191970",
    "foldseek_max": "#32CD32",
    "deepfri_sequence": "#40E0D0",
    "deepfri_structure": "#8B4513",
    "dpfunc": "#DB7093",
    "heal": "#4169E1",
    "interproscan": "#D55E00",
    "gat_go": "#EE3377",
    "deepgraphgo": "#882255",
    "deepgoplus": "#E31A1C",
    "deepgose": "#8C510A",
    "transfun": "#666666",
    "eggnog_mapper": "#B79F00",
    "hayai": "#44AA99",
    "gomap": "#AA4499",
}


def method_color(method: str) -> str:
    """Return the fixed color assigned to an individual method."""
    base = method.removesuffix("_max")
    if method in METHOD_COLOR:
        return METHOD_COLOR[method]
    if base in METHOD_COLOR:
        return METHOD_COLOR[base]
    return FAMILY_COLOR[family(method)]

FAMILY_LABEL = {
    "proposed": "DeepGreenGO Hybrid (this work)",
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
        help="Default: <workspace>/plots/main_comparison",
    )
    parser.add_argument("--journal", choices=("bmc", "nature"), default=None)
    return parser.parse_args()


def load_comparison(workspace: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    deepgreengo = metrics.loc[metrics["method"] == "deepgreengo"]
    counts = deepgreengo.groupby("ontology").size().to_dict()
    expected_counts = {ontology: 1 for ontology in ONTOLOGY_ORDER}
    if counts != expected_counts:
        raise ValueError(
            "Expected exactly one DeepGreenGO ensemble result per ontology; "
            f"found {counts}"
        )
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
        raise ValueError(f"Incomplete comparison ontology results: {incomplete}")
    return metrics, bootstrap


def ordered_methods(metrics: pd.DataFrame) -> list[str]:
    # Filtering here, rather than by trimming METHOD_ORDER, also drops excluded
    # methods from the unknown-method tail below, so nothing reappears in a
    # figure just because it is absent from the explicit ordering.
    present = set(metrics["method"].astype(str)) - EXCLUDED_FROM_PLOTS
    known = [method for method in METHOD_ORDER if method in present]
    return known + sorted(present - set(known))


def family(method: str) -> str:
    return METHOD_FAMILY.get(method, "other")


def add_proposed_separator(ax: plt.Axes, methods: list[str]) -> None:
    if methods and methods[0] == "deepgreengo" and len(methods) > 1:
        ax.axhline(0.5, color="#777777", linewidth=0.65, zorder=1)


def build_legend_handles(methods: list[str]) -> list:
    """Explain transfer-rule styling; method colours are labelled on the axis."""
    handles = [
        Patch(
            facecolor="#777777",
            edgecolor="#333333",
            label="Top-10 weighted transfer",
        ),
        Patch(
            facecolor="white",
            edgecolor="#444444",
            hatch="//",
            label="Single best identity (within top-10)",
        ),
    ]
    return handles


def add_shared_legend(fig: plt.Figure, methods: list[str]) -> None:
    handles = build_legend_handles(methods)
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 4),
        bbox_to_anchor=(0.5, -0.045), frameon=False,
        handletextpad=0.5, columnspacing=1.1,
    )


def style_axis(ax: plt.Axes, ontology: str, xlabel: str, panel: str) -> None:
    ax.set_title(ONTOLOGY_SHORT[ontology])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    label_panel(ax, panel)


def set_method_axis(
    ax: plt.Axes,
    y: np.ndarray,
    methods: list[str],
    column_index: int,
) -> None:
    """Keep shared tick positions while showing method names only at left."""
    ax.set_yticks(y)
    if column_index == 0:
        ax.set_yticklabels([METHOD_LABEL.get(method, method) for method in methods])
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)


def assert_print_fonts(fig: plt.Figure) -> None:
    warnings = check_min_font(fig, MAIN)
    if warnings:
        raise ValueError("; ".join(warnings))


def plot_cafa(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    methods: list[str],
    out: Path,
) -> None:
    """Protein-centric point estimates as horizontal bars with paired-bootstrap CIs."""
    fig, axes = plt.subplots(
        2, 3,
        figsize=(DOUBLE_COLUMN_IN, max(4.8, 0.38 * len(methods) + 2.4)),
        sharey=True,
    )
    y = np.arange(len(methods))
    panel_index = 0
    specifications = [
        ("cafa_fmax", "Protein-centric F$_{max}$"),
        ("cafa_smin", "Protein-centric S$_{min}$"),
    ]
    for row_index, (value_col, xlabel) in enumerate(specifications):
        for column_index, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row_index, column_index]
            point = metrics[metrics["ontology"] == ontology].set_index("method")
            boot = bootstrap[bootstrap["ontology"] == ontology]
            upper_values = []
            for yi, method in enumerate(methods):
                value = float(point.loc[method, value_col])
                values = boot.loc[
                    boot["method"] == method, value_col
                ].dropna().to_numpy()
                if len(values) != 1000:
                    raise ValueError(
                        "Expected 1,000 bootstrap values for "
                        f"{method}/{ontology}/{value_col}; found {len(values)}"
                    )
                low, high = np.quantile(values, [0.025, 0.975])
                upper_values.append(float(high))
                color = method_color(method)
                bar = ax.barh(
                    yi,
                    value,
                    height=0.56,
                    color=color,
                    edgecolor="#2B2B2B",
                    linewidth=0.6,
                    zorder=3,
                )[0]
                if method.endswith("_max"):
                    bar.set_hatch("//")
                ax.errorbar(
                    value,
                    yi,
                    xerr=np.asarray([[max(0.0, value - low)], [max(0.0, high - value)]]),
                    fmt="none",
                    ecolor="#202020",
                    elinewidth=0.8,
                    capsize=2.0,
                    capthick=0.8,
                    zorder=4,
                )
            add_proposed_separator(ax, methods)
            maximum = max(upper_values)
            ax.set_xlim(0, maximum * 1.10 if maximum > 0 else 1)
            set_method_axis(ax, y, methods, column_index)
            style_axis(ax, ontology, xlabel, chr(97 + panel_index))
            panel_index += 1
    axes[0, 0].invert_yaxis()
    add_shared_legend(fig, methods)
    fig.subplots_adjust(left=0.17, bottom=0.22, hspace=0.42, wspace=0.24)
    assert_print_fonts(fig)
    savefig(fig, out / "comparison_cafa_performance", MAIN)

def plot_aupr(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    methods: list[str],
    out: Path,
) -> bool:
    """AUPR bars; add 95% CIs when the evaluation produced bootstrap draws."""
    has_uncertainty = {"micro_aupr", "macro_aupr"}.issubset(bootstrap.columns)
    fig, axes = plt.subplots(
        2, 3,
        figsize=(DOUBLE_COLUMN_IN, max(4.8, 0.38 * len(methods) + 2.4)),
        sharey=True,
    )
    y = np.arange(len(methods))
    panel_index = 0
    for row_index, (column, xlabel) in enumerate((
        ("micro_aupr", "Term-centric micro-AUPR"),
        ("macro_aupr", "Term-centric macro-AUPR"),
    )):
        for column_index, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row_index, column_index]
            point = metrics[metrics["ontology"] == ontology].set_index("method")
            boot = bootstrap[bootstrap["ontology"] == ontology]
            upper_values = []
            for yi, method in enumerate(methods):
                value = float(point.loc[method, column])
                if not np.isfinite(value):
                    upper_values.append(0.0)
                    continue
                color = method_color(method)
                bar = ax.barh(
                    yi,
                    value,
                    height=0.56,
                    color=color,
                    edgecolor="#2B2B2B",
                    linewidth=0.6,
                    zorder=3,
                )[0]
                if method.endswith("_max"):
                    bar.set_hatch("//")
                upper = value
                if has_uncertainty:
                    values = boot.loc[
                        boot["method"] == method, column
                    ].dropna().to_numpy()
                    if len(values) not in (0, 1000):
                        raise ValueError(
                            "Expected either 0 or 1,000 AUPR bootstrap values for "
                            f"{method}/{ontology}/{column}; found {len(values)}"
                        )
                    if len(values) == 0:
                        upper_values.append(upper)
                        continue
                    low, high = np.quantile(values, [0.025, 0.975])
                    upper = float(high)
                    ax.errorbar(
                        value,
                        yi,
                        xerr=np.asarray(
                            [[max(0.0, value - low)], [max(0.0, high - value)]]
                        ),
                        fmt="none",
                        ecolor="#202020",
                        elinewidth=0.8,
                        capsize=2.0,
                        capthick=0.8,
                        zorder=4,
                    )
                upper_values.append(upper)
            add_proposed_separator(ax, methods)
            maximum = max(upper_values)
            ax.set_xlim(0, maximum * 1.12 if maximum > 0 else 1)
            set_method_axis(ax, y, methods, column_index)
            style_axis(ax, ontology, xlabel, chr(97 + panel_index))
            panel_index += 1
    axes[0, 0].invert_yaxis()
    add_shared_legend(fig, methods)
    fig.subplots_adjust(left=0.17, bottom=0.22, hspace=0.42, wspace=0.24)
    assert_print_fonts(fig)
    savefig(fig, out / "comparison_aupr", MAIN)
    return has_uncertainty

def plot_coverage(metrics: pd.DataFrame, methods: list[str], out: Path) -> list[str]:
    """Plot abstention/retrieval coverage only for sparse retrieval pipelines."""
    coverage_methods = [
        method for method in COVERAGE_METHOD_ORDER if method in methods
    ]
    if not coverage_methods:
        raise ValueError("No sparse retrieval baselines are available for coverage plotting")
    plot_data = metrics.copy()
    plot_data["protein_coverage_percent"] = (
        100 * plot_data["protein_coverage_any_score"]
    )
    plot_data["term_coverage_percent"] = (
        100 * plot_data["predicted_term_coverage"] / plot_data["test_terms"]
    )
    fig, axes = plt.subplots(
        2, 3,
        figsize=(
            DOUBLE_COLUMN_IN,
            max(4.4, 0.38 * len(coverage_methods) + 2.4),
        ),
        sharey=True,
    )
    y = np.arange(len(coverage_methods))
    panel_index = 0
    for row_index, (column, xlabel) in enumerate((
        ("protein_coverage_percent", "Proteins with an eligible hit (%)"),
        ("term_coverage_percent", "GO terms transferred (%)"),
    )):
        for column_index, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row_index, column_index]
            subset = plot_data[
                plot_data["ontology"] == ontology
            ].set_index("method")
            values = np.asarray(
                [float(subset.loc[method, column]) for method in coverage_methods]
            )
            colors = [
                method_color(method) for method in coverage_methods
            ]
            bars = ax.barh(
                y,
                values,
                height=0.62,
                color=colors,
                edgecolor="#222222",
                linewidth=0.5,
                zorder=3,
            )
            for bar, method in zip(bars, coverage_methods):
                if method.endswith("_max"):
                    bar.set_hatch("//")
            for yi, value in enumerate(values):
                ax.text(
                    min(value + 1.4, 97.0),
                    yi,
                    f"{value:.1f}",
                    va="center",
                    ha="left" if value < 92 else "right",
                )
            ax.set_xlim(0, 105)
            set_method_axis(ax, y, coverage_methods, column_index)
            style_axis(ax, ontology, xlabel, chr(97 + panel_index))
            panel_index += 1
    axes[0, 0].invert_yaxis()
    add_shared_legend(fig, coverage_methods)
    fig.subplots_adjust(left=0.17, bottom=0.22, hspace=0.42, wspace=0.24)
    assert_print_fonts(fig)
    savefig(fig, out / "comparison_prediction_coverage", MAIN)
    return coverage_methods


def plot_threshold_coverage(
    metrics: pd.DataFrame,
    methods: list[str],
    out: Path,
) -> None:
    """Coverage at an actual decision threshold, every method included.

    protein_coverage_any_score (used above) is "does at least one score exceed
    zero" - for a sigmoid-family dense model that is essentially guaranteed and
    not evidence of anything. This figure uses
    test_coverage_at_validation_threshold instead: the threshold is selected on
    the validation split (never on test) and coverage is the fraction of test
    proteins with at least one prediction that clears it. That is a real,
    non-saturating quantity for a dense model - DeepGreenGO scores
    0.97/0.81/0.88 across MF/BP/CC here, not 1.00 - so DeepGreenGO belongs in
    this comparison rather than being excluded from it. The frequency prior
    still saturates at exactly 1.00 everywhere, but for a different and still
    structural reason (see caption), so its bar is not read the same way as a
    per-protein model's.
    """
    plot_data = metrics.copy()
    plot_data["coverage_percent"] = (
        100 * plot_data["test_coverage_at_validation_threshold"]
    )
    fig, axes = plt.subplots(
        2, 3,
        figsize=(DOUBLE_COLUMN_IN, max(4.8, 0.38 * len(methods) + 2.4)),
        sharey=True,
    )
    y = np.arange(len(methods))
    panel_index = 0
    for row_index, (column, xlabel) in enumerate((
        ("coverage_percent", "Coverage at threshold (%)"),
        ("mean_terms_per_protein_at_validation_threshold", "GO terms / protein (mean)"),
    )):
        for column_index, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row_index, column_index]
            subset = plot_data[plot_data["ontology"] == ontology].set_index("method")
            values = np.asarray([float(subset.loc[method, column]) for method in methods])
            colors = [method_color(method) for method in methods]
            bars = ax.barh(
                y, values, height=0.62, color=colors,
                edgecolor="#222222", linewidth=0.5, zorder=3,
            )
            for bar, method in zip(bars, methods):
                if method.endswith("_max"):
                    bar.set_hatch("//")
            add_proposed_separator(ax, methods)
            top = max(values) * 1.15 if max(values) > 0 else 1.0
            ax.set_xlim(0, top)
            for yi, value in enumerate(values):
                label = f"{value:.1f}" if row_index == 1 else f"{value:.1f}"
                ax.text(min(value + top * 0.02, top * 0.97), yi, label,
                       va="center", ha="left" if value < top * 0.85 else "right")
            set_method_axis(ax, y, methods, column_index)
            style_axis(ax, ontology, xlabel, chr(97 + panel_index))
            panel_index += 1
    axes[0, 0].invert_yaxis()
    add_shared_legend(fig, methods)
    fig.subplots_adjust(left=0.17, bottom=0.22, hspace=0.42, wspace=0.24)
    assert_print_fonts(fig)
    savefig(fig, out / "comparison_threshold_coverage", MAIN)

def load_deepgreengo_provenance(workspace: Path) -> tuple[list[int], str]:
    metadata_paths = sorted(
        (workspace / "predictions" / "deepgreengo").glob("*.metadata.json")
    )
    if len(metadata_paths) != len(ONTOLOGY_ORDER):
        raise ValueError(
            "Expected one DeepGreenGO metadata file per ontology; "
            f"found {len(metadata_paths)}"
        )
    seed_sets: set[tuple[int, ...]] = set()
    variants: set[str] = set()
    for path in metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("method") != "deepgreengo" or metadata.get("split") != "test":
            raise ValueError(f"Unexpected DeepGreenGO provenance in {path}")
        seeds = tuple(int(seed) for seed in metadata.get("ensemble_seeds", []))
        if not seeds:
            raise ValueError(f"Missing ensemble seeds in {path}")
        seed_sets.add(seeds)
        if metadata.get("model_variant"):
            variants.add(str(metadata["model_variant"]))

    if len(seed_sets) != 1:
        raise ValueError(f"Inconsistent DeepGreenGO ensemble seeds: {seed_sets}")
    ensemble_seeds = list(next(iter(seed_sets)))

    if not variants:
        seed_metadata = sorted(
            (workspace / "predictions").glob(
                "deepgreengo_seed_*_test/*.metadata.json"
            )
        )
        for path in seed_metadata:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("model_variant"):
                variants.add(str(metadata["model_variant"]))
                continue
            checkpoint = Path(str(metadata.get("checkpoint", "")))
            try:
                project_index = checkpoint.parts.index(PROJECT_DIR.name)
            except ValueError:
                continue
            local_checkpoint = PROJECT_DIR.joinpath(
                *checkpoint.parts[project_index + 1:]
            )
            config_path = local_checkpoint.with_name("config.json")
            if config_path.is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                variants.add(str(config.get("model", "Hybrid")))

    if len(variants) != 1:
        raise ValueError(
            "Could not verify one DeepGreenGO model variant across ensemble "
            f"checkpoints; found {sorted(variants)}"
        )
    return ensemble_seeds, next(iter(variants))


def set_focal_variant_label(model_variant: str) -> None:
    label = f"DeepGreenGO {model_variant} (this work)"
    METHOD_LABEL["deepgreengo"] = label
    FAMILY_LABEL["proposed"] = label


def validate_best_hit_provenance(workspace: Path, methods: list[str]) -> None:
    for method in methods:
        if not method.endswith("_max"):
            continue
        metadata_paths = sorted(
            (workspace / "predictions" / method).glob("*.metadata.json")
        )
        for path in metadata_paths:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            transfer = str(metadata.get("transfer", ""))
            if "one highest-identity" not in transfer or "top-k pool" not in transfer:
                raise ValueError(
                    f"{path} contains the legacy per-term max-identity transfer. "
                    "Regenerate the similarity baselines and reevaluate before plotting."
                )


def paired_fmax_report(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for ontology in ONTOLOGY_ORDER:
        point = metrics[metrics["ontology"] == ontology].set_index("method")
        baselines = point.drop(index="deepgreengo")
        competitor = str(baselines["cafa_fmax"].idxmax())
        subset = bootstrap[bootstrap["ontology"] == ontology]
        focal = subset[subset["method"] == "deepgreengo"].set_index("bootstrap")
        other = subset[subset["method"] == competitor].set_index("bootstrap")
        paired = focal[["cafa_fmax"]].join(
            other[["cafa_fmax"]],
            how="inner",
            lsuffix="_deepgreengo",
            rsuffix="_competitor",
        )
        if len(paired) != 1000:
            raise ValueError(
                f"Expected 1,000 paired Fmax draws for {ontology}/{competitor}; "
                f"found {len(paired)}"
            )
        delta = (
            paired["cafa_fmax_deepgreengo"]
            - paired["cafa_fmax_competitor"]
        ).to_numpy()
        rows.append({
            "ontology": ontology,
            "competitor": competitor,
            "competitor_label": METHOD_LABEL.get(competitor, competitor),
            "deepgreengo_fmax": float(point.loc["deepgreengo", "cafa_fmax"]),
            "competitor_fmax": float(point.loc[competitor, "cafa_fmax"]),
            "fmax_difference": float(
                point.loc["deepgreengo", "cafa_fmax"]
                - point.loc[competitor, "cafa_fmax"]
            ),
            "paired_difference_ci_low": float(np.quantile(delta, 0.025)),
            "paired_difference_ci_high": float(np.quantile(delta, 0.975)),
            "fraction_bootstraps_deepgreengo_better": float(np.mean(delta > 0)),
            "bootstrap_replicates": len(delta),
        })
    return pd.DataFrame(rows)


def build_captions(
    ensemble_seeds: list[int],
    model_variant: str,
    paired_report: pd.DataFrame,
    aupr_has_uncertainty: bool,
    bootstrap_unit: str = "protein",
    unique_sequences: int | None = None,
    methods: list[str] | None = None,
) -> str:
    seeds = ", ".join(str(seed) for seed in ensemble_seeds)
    external_labels = [
        METHOD_LABEL[method]
        for method in EXTERNAL_PRETRAINED_METHODS
        if methods is None or method in methods
    ]
    # Emit the caveat only when an external pretrained comparator is actually
    # plotted. Interpolating an empty list left a dangling " use externally
    # released ..." fragment with no subject.
    if external_labels:
        external_sentence = (
            f" {', '.join(external_labels)} "
            f"{'uses' if len(external_labels) == 1 else 'use'} externally released "
            "pretrained parameters or reference data that were not restricted to "
            "the locked ARC training split; their scores are descriptive "
            "comparators, not leakage-controlled generalization estimates."
        )
    else:
        external_sentence = ""
    paired_parts = []
    for row in paired_report.itertuples(index=False):
        paired_parts.append(
            f"{ONTOLOGY_SHORT[row.ontology]}: ΔFmax={row.fmax_difference:+.3f} "
            f"versus {row.competitor_label}, "
            f"bootstrap fraction better={row.fraction_bootstraps_deepgreengo_better:.3f}"
        )
    paired_text = "; ".join(paired_parts)
    if bootstrap_unit == "identical_sequence_cluster":
        bootstrap_description = (
            "1,000 paired identical-sequence-cluster bootstrap replicates"
        )
        if unique_sequences is not None:
            bootstrap_description += (
                f" ({unique_sequences} unique sequences represented by 754 PDB chains)"
            )
        bootstrap_warning = ""
    else:
        bootstrap_description = "1,000 paired protein-level bootstrap replicates"
        bootstrap_warning = (
            f" The saved results predate identical-sequence cluster resampling"
            + (
                f" and contain only {unique_sequences} unique sequences among 754 chains"
                if unique_sequences is not None
                else ""
            )
            + "; these confidence intervals treat duplicate chains as independent "
            "and must be regenerated before manuscript use."
        )
    if aupr_has_uncertainty:
        aupr_uncertainty = (
            "Error bars are percentile 95% confidence intervals from the same "
            f"{bootstrap_description}."
        )
    else:
        aupr_uncertainty = (
            "The pulled bootstrap file predates AUPR resampling, so these are "
            "point estimates only; rerun the evaluation before manuscript use "
            "to add paired-bootstrap confidence intervals."
        )

    return f"""comparison_cafa_performance
DeepGreenGO versus completed baselines on the nominal 30%-identity/80%-coverage test split (n = 754 PDB chains). The focal method is the five-seed {model_variant} GCN-GAT ensemble ({seeds}). Bars show chain-level test-set point estimates and error bars show percentile 95% confidence intervals from {bootstrap_description}.{bootstrap_warning} Colors denote method family; solid bars denote top-10 weighted transfer and hatched bars denote one highest-identity hit selected from the same eligible top-10 pool. Paired Fmax comparisons against the strongest baseline in each ontology: {paired_text}.{external_sentence}

comparison_aupr
DeepGreenGO versus completed baselines on the same test split. Micro-AUPR pools protein-term decisions; macro-AUPR averages per-term average precision across GO terms observed in the test set. Bars show test-set point estimates. {aupr_uncertainty} Colors and hatching follow the definitions above. External pretrained comparators were not audited against their original training corpora, so apparent gains from those methods cannot be attributed solely to architecture.

comparison_prediction_coverage
Retrieval coverage for sparse similarity-search baselines only. DeepGreenGO, the frequency prior, and other dense-output models are excluded from THIS figure because their nonzero score matrices make raw "any nonzero score" coverage saturate by construction - it is not a meaningful comparison at this specific definition. Protein coverage is the percentage of test proteins with at least one eligible training hit; term coverage is the percentage of evaluated test GO terms transferred from eligible hits. Hits must pass E-value 1e-3 and both query- and target-coverage thresholds of 50%. Coverage measures retrieval/abstention, not predictive accuracy. Solid bars denote top-10 weighted transfer and hatched bars denote one highest-identity hit selected from the same eligible top-10 pool. See comparison_threshold_coverage for a coverage definition that includes DeepGreenGO on equal footing.

comparison_threshold_coverage
Coverage at an operating decision threshold, all methods together. The threshold is selected on the validation split only (never on test), so this is not the trivial "any nonzero score" coverage above - it is the fraction of test proteins receiving at least one prediction that clears a real decision boundary, and the mean number of GO terms predicted per protein at that boundary. DeepGreenGO scores below 100% here (97.2% MF, 81.2% BP, 88.3% CC) because its per-protein sigmoid outputs genuinely vary. This establishes non-tautological coverage relative to the frequency and retrieval controls, but it is not a distinctive advantage over dense external models such as DeepGOPlus or DeepGO-SE, whose coverage is also protein-specific. The frequency prior still reaches exactly 100% at this threshold for a structural reason: it assigns every protein the same training-prevalence score per term, so a term's score either clears the threshold for every protein or for none. Solid bars denote top-10 weighted transfer and hatched bars denote one highest-identity hit selected from the same eligible top-10 pool.
"""


def build_manuscript_notes(
    metrics: pd.DataFrame,
    model_variant: str,
) -> str:
    mf = metrics[metrics["ontology"] == "molecular_function"].set_index("method")
    baselines = mf.drop(index="deepgreengo")
    best_fmax_method = str(baselines["cafa_fmax"].idxmax())
    best_smin_method = str(baselines["cafa_smin"].idxmin())
    dgg_fmax = float(mf.loc["deepgreengo", "cafa_fmax"])
    baseline_fmax = float(mf.loc[best_fmax_method, "cafa_fmax"])
    dgg_smin = float(mf.loc["deepgreengo", "cafa_smin"])
    baseline_smin = float(mf.loc[best_smin_method, "cafa_smin"])
    fmax_delta = dgg_fmax - baseline_fmax
    smin_delta = dgg_smin - baseline_smin
    if fmax_delta > 0 and smin_delta < 0:
        interpretation = (
            "DeepGreenGO outperforms the strongest baseline on both MF metrics."
        )
    elif fmax_delta < 0 and smin_delta > 0:
        interpretation = (
            "DeepGreenGO underperforms the strongest baseline on both MF metrics; "
            "the benchmark does not support an MF accuracy-gain claim."
        )
    else:
        interpretation = (
            "The MF comparison is mixed across Fmax and Smin and should not be "
            "summarized as an unqualified accuracy gain."
        )
    present_methods = set(metrics["method"].astype(str))
    external_labels = [
        METHOD_LABEL[method]
        for method in EXTERNAL_PRETRAINED_METHODS
        if method in present_methods
    ]
    if external_labels:
        external_caveat = (
            f"External-pretraining caveat: {', '.join(external_labels)} "
            f"{'was' if len(external_labels) == 1 else 'were'} evaluated with "
            "released pretrained parameters or reference data. Their original "
            "training corpora were not restricted to the locked ARC training "
            "split, so these comparisons are descriptive and cannot establish "
            "leakage-controlled generalization."
        )
    else:
        external_caveat = (
            "External-pretraining caveat: no externally pretrained comparator is "
            "present in this benchmark, so no such caveat applies."
        )
    return (
        f"Model identity and ablation framing: The headline benchmark uses the "
        f"five-seed DeepGreenGO {model_variant} GCN-GAT ensemble. Input ablations "
        "show that the ProtBERT sequence representation drives most of the "
        "performance; the comparison should not imply that graph fusion alone "
        "explains the result.\n\n"
        f"MF comparison: DeepGreenGO Fmax is {dgg_fmax:.3f}, versus "
        f"{baseline_fmax:.3f} for {METHOD_LABEL.get(best_fmax_method, best_fmax_method)}. "
        f"Its Smin is {dgg_smin:.3f}, versus the best baseline value of "
        f"{baseline_smin:.3f} for {METHOD_LABEL.get(best_smin_method, best_smin_method)}. "
        f"The corresponding differences are {fmax_delta:+.3f} Fmax and "
        f"{smin_delta:+.3f} Smin. {interpretation}\n\n"
        + external_caveat
    )


def write_supporting_files(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    methods: list[str],
    coverage_methods: list[str],
    workspace: Path,
    out: Path,
    aupr_has_uncertainty: bool,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ensemble_seeds, model_variant = load_deepgreengo_provenance(workspace)
    paired_report = paired_fmax_report(metrics, bootstrap)
    bootstrap_unit = (
        str(metrics["bootstrap_unit"].dropna().iloc[0])
        if "bootstrap_unit" in metrics and metrics["bootstrap_unit"].notna().any()
        else "protein"
    )
    if "test_unique_sequences" in metrics and metrics["test_unique_sequences"].notna().any():
        unique_sequences = int(metrics["test_unique_sequences"].dropna().iloc[0])
    else:
        sequence_path = workspace / "inputs" / "test.fasta"
        sequences = []
        current = []
        if sequence_path.is_file():
            for line in sequence_path.read_text().splitlines():
                if line.startswith(">"):
                    if current:
                        sequences.append("".join(current))
                    current = []
                else:
                    current.append(line.strip())
            if current:
                sequences.append("".join(current))
        unique_sequences = len(set(sequences)) if sequences else None
    metrics.to_csv(out / "comparison_metrics_plotted.csv", index=False)
    bootstrap.to_csv(out / "comparison_bootstrap_plotted.csv", index=False)
    paired_report.to_csv(out / "paired_fmax_vs_best_baseline.csv", index=False)
    (out / "captions.txt").write_text(
        build_captions(
            ensemble_seeds,
            model_variant,
            paired_report,
            aupr_has_uncertainty,
            bootstrap_unit,
            unique_sequences,
            methods,
        ),
        encoding="utf-8",
    )
    (out / "manuscript_results_notes.txt").write_text(
        build_manuscript_notes(metrics, model_variant) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source_workspace": str(workspace.resolve()),
        "journal_profile": JOURNAL,
        "source_metrics": "results/benchmark_metrics.csv",
        "source_bootstrap": "results/bootstrap_metrics.csv",
        "focal_method": "deepgreengo",
        "focal_method_label": METHOD_LABEL["deepgreengo"],
        "focal_model_variant": model_variant,
        "focal_method_ensemble_seeds": ensemble_seeds,
        "caption_file": "captions.txt",
        "manuscript_results_notes": "manuscript_results_notes.txt",
        "paired_fmax_report": "paired_fmax_vs_best_baseline.csv",
        "included_methods": methods,
        "coverage_methods": coverage_methods,
        "coverage_excludes_dense_outputs": True,
        "aupr_bootstrap_available": aupr_has_uncertainty,
        "ontologies": ONTOLOGY_ORDER,
        "outputs": [
            f"{stem}.{suffix}"
            for stem in (
                "comparison_cafa_performance",
                "comparison_aupr",
                "comparison_prediction_coverage",
                "comparison_threshold_coverage",
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
    output = (
        args.output or workspace / "plots" / "main_comparison"
    ).expanduser().resolve()
    apply_style()
    metrics, bootstrap = load_comparison(workspace)
    methods = ordered_methods(metrics)
    _, model_variant = load_deepgreengo_provenance(workspace)
    set_focal_variant_label(model_variant)
    validate_best_hit_provenance(workspace, methods)
    plot_cafa(metrics, bootstrap, methods, output)
    aupr_has_uncertainty = plot_aupr(metrics, bootstrap, methods, output)
    coverage_methods = plot_coverage(metrics, methods, output)
    plot_threshold_coverage(metrics, methods, output)
    write_supporting_files(
        metrics,
        bootstrap,
        methods,
        coverage_methods,
        workspace,
        output,
        aupr_has_uncertainty,
    )
    print(
        f"Plotted DeepGreenGO with {len(methods) - 1} comparators: "
        f"{', '.join(methods)}"
    )
    if not aupr_has_uncertainty:
        print(
            "WARNING: AUPR confidence intervals require reevaluation with the "
            "updated paired-bootstrap pipeline."
        )
    print(f"Output: {output}")

if __name__ == "__main__":
    main()
