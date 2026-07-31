#!/usr/bin/env python3
"""Stratified benchmark figures: homology, term IC, GO depth, IC-weighted PR.

Renders the DPFunc-style analyses (their Fig. 1a/b/c/d/e/f) from the tables
written by `python -m src.benchmark.run_stratified`.

Examples
--------
python src/plot_stratified.py
python src/plot_stratified.py --journal nature
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_DIR / "arc_benchmark" / "nominal_30_identity_80_coverage"

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--journal", choices=("bmc", "nature"), default=None)
_known, _ = _parser.parse_known_args()
if _known.journal:
    os.environ["DGG_JOURNAL"] = _known.journal

from plot_style import (  # noqa: E402
    DOUBLE_COLUMN_IN,
    MAIN,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    apply_style,
    label_panel,
    savefig,
)
from plot_baselines_only import (  # noqa: E402
    METHOD_LABEL,
    method_color,
)

# Ordered bins, mirroring src/benchmark/stratified.py.
HOMOLOGY_BINS = ["no hit", "<30%", "30-40%", "40-60%", ">=60%"]
IC_BINS = ["<2 bits", "2-4 bits", "4-6 bits", ">=6 bits"]
DEPTH_BINS = ["1-3", "4-6", "7-9", ">=10"]

# Keep the panel legible: these are the representative methods, one per
# evidence family plus the structure-aware deep comparators of interest.
# DeepGOPlus and DeepGO-SE are withheld from the figures here for the same
# reason as in plot_baselines_only.EXCLUDED_FROM_PLOTS; the graph-based
# baselines HEAL, GAT-GO, and DeepGraphGO take their place.
FOCUS_METHODS = [
    "deepgreengo", "naive", "blast", "diamond", "foldseek",
    "deepfri_sequence", "deepfri_structure", "dpfunc",
    "heal", "gat_go", "deepgraphgo", "transfun", "interproscan",
]

MARKERS = {
    "deepgreengo": "o", "naive": "s", "blast": "^", "diamond": "v",
    "foldseek": "D", "deepfri_sequence": "P", "deepfri_structure": "X",
    "dpfunc": "<", "heal": "H", "gat_go": "d", "deepgraphgo": "8",
    "transfun": "h", "interproscan": "p",
}

# Bins with fewer items than this cannot support a claim; drawn but flagged.
MIN_BIN_N = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--journal", choices=("bmc", "nature"), default=None)
    return parser.parse_args()


def present_methods(frame: pd.DataFrame) -> list[str]:
    available = set(frame["method"].astype(str))
    return [method for method in FOCUS_METHODS if method in available]


def _series(frame: pd.DataFrame, method: str, ontology: str,
            bins: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (value, sd, n) aligned to `bins`, NaN where a bin is absent."""
    subset = frame[(frame.method == method) & (frame.ontology == ontology)]
    lookup = subset.set_index("bin")
    values, sds, counts = [], [], []
    for name in bins:
        if name in lookup.index:
            row = lookup.loc[name]
            values.append(float(row["value"]) if pd.notna(row["value"]) else np.nan)
            sd = row.get("seed_sd", np.nan)
            sds.append(float(sd) if pd.notna(sd) else np.nan)
            counts.append(int(row["bin_n"]))
        else:
            values.append(np.nan)
            sds.append(np.nan)
            counts.append(0)
    return np.asarray(values), np.asarray(sds), np.asarray(counts)


def _draw_binned_row(
    axes_row,
    frame: pd.DataFrame,
    bins: list[str],
    ylabel: str,
    xlabel: str,
    panel_offset: int,
    show_xlabel: bool = True,
    letter: str | None = None,
    per_panel_letters: bool = False,
) -> set[str]:
    """Draw one analysis (methods x ordered bins) across a row of 3 axes.

    `letter` labels the row once, at its first (leftmost) axis, matching the
    DPFunc convention where a-b-c-d name analyses, not ontology x analysis.
    """
    methods = present_methods(frame)
    keep = [
        name for name in bins
        if frame.loc[frame["bin"] == name, "bin_n"].to_numpy().max(initial=0) > 0
    ]
    x = np.arange(len(keep), dtype=float)
    low_n_bins: set[str] = set()

    for column_index, ontology in enumerate(ONTOLOGY_ORDER):
        ax = axes_row[column_index]
        for method in methods:
            values, sds, counts = _series(frame, method, ontology, keep)
            color = method_color(method)
            has_sd = np.isfinite(sds).any()
            ax.errorbar(
                x, values,
                yerr=np.where(np.isfinite(sds), sds, 0.0) if has_sd else None,
                marker=MARKERS.get(method, "o"), markersize=4.0,
                color=color, linewidth=1.3, capsize=2.0 if has_sd else 0,
                elinewidth=0.8,
                markeredgecolor="#222222", markeredgewidth=0.4,
                zorder=4 if method == "deepgreengo" else 3,
                alpha=1.0 if method == "deepgreengo" else 0.85,
            )
            for name, count in zip(keep, counts):
                if 0 < count < MIN_BIN_N:
                    low_n_bins.add(name)

        _, _, counts = _series(frame, methods[0], ontology, keep)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{name} (n={count})" for name, count in zip(keep, counts)],
            rotation=38, ha="right",
        )
        ax.set_xlim(-0.35, len(keep) - 0.65)
        ax.set_title(ONTOLOGY_SHORT[ontology])
        if show_xlabel:
            ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", visible=True)
        if per_panel_letters:
            label_panel(ax, chr(97 + panel_offset + column_index))
    axes_row[0].set_ylabel(ylabel)
    if letter is not None:
        label_panel(axes_row[0], letter)
    return low_n_bins


def plot_binned(
    frame: pd.DataFrame,
    bins: list[str],
    ylabel: str,
    xlabel: str,
    stem: str,
    out: Path,
    caption_note: str,
) -> None:
    """One panel per ontology; methods as lines across ordered bins."""
    methods = present_methods(frame)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.0), sharey=True)
    low_n_bins = _draw_binned_row(
        axes, frame, bins, ylabel, xlabel, panel_offset=0, per_panel_letters=True,
    )

    handles = [
        Line2D([0], [0], color=method_color(method),
               marker=MARKERS.get(method, "o"), markersize=4.5, linewidth=1.3,
               markeredgecolor="#222222", markeredgewidth=0.4,
               label=METHOD_LABEL.get(method, method))
        for method in methods
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 4),
        bbox_to_anchor=(0.5, -0.16), frameon=False,
        handletextpad=0.5, columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.10, bottom=0.42, wspace=0.16)
    savefig(fig, out / stem, MAIN)
    return sorted(low_n_bins)


def _ic_aupr_summary(frame: pd.DataFrame, methods: list[str]) -> str:
    """Area under each IC-weighted PR curve, trapezoidal, per method/ontology.

    Text-panel companion to the curves, in the style of DPFunc Fig. 2b's
    numeric annotation block -- not a lettered panel.
    """
    lines = []
    for method in methods:
        parts = []
        for ontology in ONTOLOGY_ORDER:
            subset = frame[
                (frame.method == method) & (frame.ontology == ontology)
            ].sort_values("threshold")
            recall = pd.to_numeric(subset["ic_weighted_recall"], errors="coerce").to_numpy()
            precision = pd.to_numeric(subset["ic_weighted_precision"], errors="coerce").to_numpy()
            valid = np.isfinite(recall) & np.isfinite(precision)
            if valid.sum() < 2:
                continue
            order = np.argsort(recall[valid])
            area = float(np.trapezoid(precision[valid][order], recall[valid][order]))
            parts.append(f"IC_AUPR_{ONTOLOGY_SHORT[ontology]}={area:.2f}")
        if parts:
            lines.append(f"{METHOD_LABEL.get(method, method)}\n  " + "  ".join(parts))
    return "\n".join(lines)


def _draw_pr_row(axes_row, frame: pd.DataFrame, panel_offset: int, show_xlabel: bool = True,
                  letter: str | None = None) -> None:
    """Draw IC-weighted PR curves across a row of 3 axes (one per ontology)."""
    methods = present_methods(frame)
    for column_index, ontology in enumerate(ONTOLOGY_ORDER):
        ax = axes_row[column_index]
        for method in methods:
            subset = frame[
                (frame.method == method) & (frame.ontology == ontology)
            ].sort_values("threshold")
            recall = pd.to_numeric(subset["ic_weighted_recall"], errors="coerce")
            precision = pd.to_numeric(subset["ic_weighted_precision"], errors="coerce")
            valid = recall.notna() & precision.notna()
            if not valid.any():
                continue
            ax.plot(
                recall[valid], precision[valid],
                color=method_color(method),
                linewidth=1.5 if method == "deepgreengo" else 1.1,
                zorder=4 if method == "deepgreengo" else 3,
                alpha=1.0 if method == "deepgreengo" else 0.85,
            )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title(ONTOLOGY_SHORT[ontology])
        if show_xlabel:
            ax.set_xlabel("IC-weighted recall")
    axes_row[0].set_ylabel("IC-weighted precision")
    if letter is not None:
        label_panel(axes_row[0], letter)


def plot_ic_weighted_pr(frame: pd.DataFrame, out: Path) -> None:
    """IC-weighted precision-recall curves, one panel per ontology."""
    methods = present_methods(frame)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.0), sharey=True)
    for column_index in range(3):
        label_panel(axes[column_index], chr(97 + column_index))
    _draw_pr_row(axes, frame, panel_offset=0)

    handles = [
        Line2D([0], [0], color=method_color(method), linewidth=1.5,
               label=METHOD_LABEL.get(method, method))
        for method in methods
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 4),
        bbox_to_anchor=(0.5, -0.16), frameon=False,
        handletextpad=0.5, columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.10, bottom=0.32, wspace=0.16)
    savefig(fig, out / "stratified_ic_weighted_pr", MAIN)


def plot_combined(
    homology: pd.DataFrame,
    ic: pd.DataFrame,
    depth: pd.DataFrame,
    curves: pd.DataFrame,
    out: Path,
) -> None:
    """Single main-text figure, DPFunc-style: one letter per ANALYSIS, not
    per ontology. Row a = homology-Fmax, b = IC-AUPRC, c = depth-AUPRC, each
    with 3 ontology sub-panels sharing one label. Row d = IC-weighted PR
    curves (3 ontology sub-panels, one label) plus a 4th, unlettered panel of
    the per-method/per-ontology area-under-curve numbers, mirroring the
    numeric annotation block DPFunc prints beside its PR curves.
    """
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 4 * 2.9))
    gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, 0.85], hspace=0.85, wspace=0.16)

    row0 = [fig.add_subplot(gs[0, c]) for c in range(3)]
    row1 = [fig.add_subplot(gs[1, c]) for c in range(3)]
    row2 = [fig.add_subplot(gs[2, c]) for c in range(3)]
    row3 = [fig.add_subplot(gs[3, c]) for c in range(3)]
    for axes_row in (row0, row1, row2):
        for ax in axes_row[1:]:
            ax.sharey(axes_row[0])
    for ax in row3[1:]:
        ax.sharey(row3[0])
    text_ax = fig.add_subplot(gs[3, 3])
    text_ax.axis("off")

    _draw_binned_row(
        row0, homology, HOMOLOGY_BINS,
        "CAFA F$_{max}$", "Max identity to training set", panel_offset=0, letter="a",
    )
    _draw_binned_row(
        row1, ic, IC_BINS,
        "Term-centric AUPRC", "Term information content", panel_offset=0, letter="b",
    )
    _draw_binned_row(
        row2, depth, DEPTH_BINS,
        "Term-centric AUPRC", "GO term depth from root", panel_offset=0, letter="c",
    )
    _draw_pr_row(row3, curves, panel_offset=0, letter="d")

    methods = present_methods(homology)
    text_ax.text(
        0.0, 1.0, _ic_aupr_summary(curves, methods),
        transform=text_ax.transAxes, va="top", ha="left", fontsize=6.0,
        linespacing=1.6, family="monospace",
    )

    handles = [
        Line2D([0], [0], color=method_color(method),
               marker=MARKERS.get(method, "o"), markersize=4.5, linewidth=1.3,
               markeredgecolor="#222222", markeredgewidth=0.4,
               label=METHOD_LABEL.get(method, method))
        for method in methods
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 4),
        bbox_to_anchor=(0.5, -0.03), frameon=False,
        handletextpad=0.5, columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.11)
    savefig(fig, out / "stratified_combined", MAIN, formats=("pdf", "svg", "tiff"))


def build_captions(low_n: dict[str, list[str]], manifest: dict) -> str:
    def counts_for(key: str) -> str:
        return "; ".join(
            f"{ONTOLOGY_SHORT[ontology]} {values[key]}"
            for ontology, values in manifest["ontologies"].items()
        )

    flagged = ""
    if low_n.get("homology"):
        flagged = (
            f" Bins with fewer than {MIN_BIN_N} test proteins "
            f"({', '.join(low_n['homology'])}) are shown for completeness but are "
            "too small to support a claim; the 40-60% bin contains only 5 proteins."
        )

    return f"""Unless noted, higher values are better. DeepGreenGO error bars are mean +/- s.d. over five independent training seeds. The similarity and annotation baselines are deterministic given the split, so they produce a single value with no error bar; the asymmetry reflects the methods, not selective reporting.

stratified_homology_fmax
Protein-centric CAFA Fmax as a function of the maximum sequence identity between each test protein and the locked DeepGreenGO training set, computed by BLAST. Bins are defined on the proteins alone, never on any model's predictions, so no method is favoured by the binning. The "no hit" bin (612 of 754 test proteins) contains proteins with no detectable homolog in that locked training set, where project-trained homology-transfer baselines have nothing to transfer from. For externally pretrained methods, these bins are descriptive and do not measure overlap with each method's original training corpus; they therefore cannot by themselves establish external-training leakage.{flagged} Bin sizes: {counts_for('homology_bin_counts')}.

stratified_homology_micro_aupr
Pooled micro-AUPR as a function of the maximum sequence identity between each test protein and the locked DeepGreenGO training set. Unlike Fmax, micro-AUPR evaluates the ranking of protein-term scores without selecting a decision threshold. The identity bins, sample sizes, and external-pretraining caveat are the same as in stratified_homology_fmax; these bins do not measure similarity to each externally pretrained method's own training corpus. Bin sizes: {counts_for('homology_bin_counts')}.

stratified_ic_auprc
Term-centric AUPRC as a function of GO term information content, IC = -log2(frequency) measured on the training labels only. Higher IC means a rarer, more informative term. Terms never seen in training have undefined IC and are excluded. Term counts per bin: {counts_for('ic_bin_term_counts')}.

stratified_depth_auprc
Term-centric AUPRC as a function of GO term depth, the shortest is_a/part_of path from the ontology root. Deeper terms are more specific and harder to predict. Term counts per bin: {counts_for('depth_bin_term_counts')}. Bins empty in every ontology are omitted rather than drawn as gaps.

stratified_ic_weighted_pr
Information-content-weighted precision-recall curves across a 0.01-0.99 threshold sweep. Each protein-term decision is weighted by the term's information content, so recovering a rare, specific annotation counts for more than recovering a near-universal one. This down-weights the shallow, high-frequency terms that dominate unweighted precision-recall.

stratified_combined (main text)
Composite of the four analyses above as one figure: (a-c) CAFA Fmax by max training-set sequence identity; (d-f) term-centric AUPRC by GO term information content; (g-i) term-centric AUPRC by GO term depth from root; (j-l) IC-weighted precision-recall curves. Columns are MF, BP, CC. See the four entries above for the full methodological description of each row; this panel arrangement is provided for the main text, the individual panels above for supplementary reference at larger size.
"""


def main() -> None:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    output = (args.output or workspace / "plots" / "stratified").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = workspace / "results"
    apply_style()

    manifest = json.loads((results / "stratified_manifest.json").read_text())
    homology = pd.read_csv(results / "stratified_homology.csv")
    homology_aupr = pd.read_csv(results / "stratified_homology_aupr.csv")
    ic = pd.read_csv(results / "stratified_ic.csv")
    depth = pd.read_csv(results / "stratified_depth.csv")
    curves = pd.read_csv(results / "ic_weighted_pr.csv")

    low_n = {}
    low_n["homology"] = plot_binned(
        homology, HOMOLOGY_BINS,
        "CAFA F$_{max}$", "Max identity to training set",
        "stratified_homology_fmax", output, "",
    )
    plot_binned(
        homology_aupr, HOMOLOGY_BINS,
        "Micro-AUPR", "Max identity to training set",
        "stratified_homology_micro_aupr", output, "",
    )
    low_n["ic"] = plot_binned(
        ic, IC_BINS,
        "Term-centric AUPRC", "Term information content",
        "stratified_ic_auprc", output, "",
    )
    low_n["depth"] = plot_binned(
        depth, DEPTH_BINS,
        "Term-centric AUPRC", "GO term depth from root",
        "stratified_depth_auprc", output, "",
    )
    plot_ic_weighted_pr(curves, output)
    plot_combined(homology, ic, depth, curves, output)

    (output / "captions.txt").write_text(
        build_captions(low_n, manifest), encoding="utf-8"
    )
    print(f"Methods plotted: {', '.join(present_methods(homology))}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
