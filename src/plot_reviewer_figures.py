#!/usr/bin/env python3
"""Figures answering specific BMC Bioinformatics reviewer requests.

Mapping (reviewer comment -> figure):

  R1.3, R2.3  redundancy / data leakage between train and test
              -> figure_leakage_residual_identity
              The split is homology-controlled but NOT leakage-free, and the
              figure states the residual similarity rather than claiming none.

  R2.6        loss-function ablation (BCE vs focal)
              -> figure_loss_from_search
              A controlled loss ablation does NOT exist on the locked split;
              only the random search varied the loss. This figure reports that
              evidence honestly as observational, and the script prints what
              would be needed for the controlled version the reviewer asked for.

  R1.10       micro- and macro-AUROC reported as identical
              -> printed verification, not a figure

Run:
  python src/plot_reviewer_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_style import (
    CATEGORICAL_PALETTE,
    DOUBLE_COLUMN_IN,
    MAIN,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    SUPPLEMENTARY,
    apply_style,
    assert_palette_locked,
    label_panel,
    provenance,
    savefig,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = REPO / "preprocessing/data_arc_rebuild_2026_07_14/pdb_splits/threshold_30"
DEFAULT_TRIALS = REPO / "plots/arc_tuning_cafa/trial_metrics.csv"
TEST_TOTAL = 754

# Bin edges for residual identity. 30% is the clustering threshold the split
# was built at, so anything at or above it is the residual the reviewers asked
# to see quantified.
IDENTITY_BINS = [0, 30, 40, 60, 100]
IDENTITY_LABELS = ["<30%", "30-40%", "40-60%", ">=60%"]


def max_identity(tsv: Path) -> dict[str, float]:
    best: dict[str, float] = {}
    if not tsv.is_file():
        raise SystemExit(f"BLAST test-vs-train table not found: {tsv}")
    with tsv.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            try:
                value = float(fields[2])
            except ValueError:
                continue
            best[fields[0]] = max(best.get(fields[0], 0.0), value)
    return best


def figure_leakage(tsv: Path, out: Path, tier: str = MAIN) -> dict:
    """Residual train-test sequence identity after homology-aware splitting."""
    best = max_identity(tsv)
    values = np.array(list(best.values()), dtype=float)
    no_hit = TEST_TOTAL - values.size
    counts = [int(((values >= lo) & (values < hi)).sum())
              for lo, hi in zip(IDENTITY_BINS[:-1], IDENTITY_BINS[1:])]
    counts[-1] = int((values >= IDENTITY_BINS[-2]).sum())

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 2.6))

    ax = axes[0]
    ax.hist(values, bins=np.arange(0, 105, 5), color=CATEGORICAL_PALETTE[0],
            edgecolor="#111111", linewidth=.4)
    ax.axvline(30, color="#b00000", linewidth=1.0, linestyle="-")
    ax.text(31, ax.get_ylim()[1] * .92, "30% clustering\nthreshold", fontsize=5.4,
            color="#b00000", va="top")
    ax.set_xlabel("Maximum identity to any training sequence (%)", fontsize=7)
    ax.set_ylabel("Test proteins", fontsize=7)
    ax.set_title("Residual similarity", fontsize=8)
    label_panel(ax, "a")

    ax = axes[1]
    labels = ["no hit"] + IDENTITY_LABELS
    heights = [no_hit] + counts
    colors = ["#bbbbbb", CATEGORICAL_PALETTE[0], CATEGORICAL_PALETTE[2],
              CATEGORICAL_PALETTE[7], CATEGORICAL_PALETTE[5]]
    bars = ax.bar(np.arange(len(labels)), heights, color=colors,
                  edgecolor="#111111", linewidth=.4, width=.7)
    for rect, height in zip(bars, heights):
        ax.text(rect.get_x() + rect.get_width() / 2, height + TEST_TOTAL * .015,
                f"{height}\n({height / TEST_TOTAL:.1%})", ha="center", va="bottom", fontsize=5.2)
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=6)
    ax.set_ylim(0, TEST_TOTAL * 1.15)
    ax.set_ylabel(f"Test proteins (of {TEST_TOTAL})", fontsize=7)
    ax.set_xlabel("Maximum identity to training set", fontsize=7)
    ax.set_title("Test-set composition", fontsize=8)
    label_panel(ax, "b")

    above = int((values >= 30).sum())
    fig.text(.5, -.13,
             f"The split clusters at 30% identity with 80% coverage, but it is not leakage-free "
             f"and is not described as such. {values.size} of {TEST_TOTAL} test proteins have any "
             f"BLAST hit to the training set, and {above} ({above / TEST_TOTAL:.1%}) retain at "
             f"least 30% identity to a training sequence; {counts[-1]} reach 60% or more. "
             f"Performance stratified by these bins is reported separately so results at low "
             f"identity can be read independently of the residual. "
             + provenance("src/plot_reviewer_figures.py", str(tsv.relative_to(REPO))),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "figure_leakage_residual_identity.png", tier)

    return {"test_total": TEST_TOTAL, "with_any_hit": int(values.size), "no_hit": int(no_hit),
            "at_or_above_30": above, "at_or_above_60": int((values >= 60).sum()),
            "median_max_identity": float(np.median(values)) if values.size else float("nan"),
            "bin_counts": dict(zip(IDENTITY_LABELS, counts))}


def figure_loss(trials_csv: Path, out: Path, tier: str = SUPPLEMENTARY) -> None:
    """BCE vs focal loss, as observed across the random-search trials.

    This is NOT the controlled ablation reviewer 2 asked for. Loss was one of
    nine hyperparameters drawn independently per trial, so BCE and focal trials
    differ in learning rate, dropout and the rest as well; the comparison is
    observational and confounded. It is included because it is the only
    loss evidence that exists on the locked split, and it is labelled as such.
    """
    if not trials_csv.is_file():
        print(f"NOTE: {trials_csv} absent; skipping loss figure.")
        return
    df = pd.read_csv(trials_csv)
    if "loss" not in df.columns:
        print("NOTE: trial table has no loss column; skipping loss figure.")
        return
    metric = "validation_macro_fmax" if df["validation_macro_fmax"].notna().any() else "validation_micro_fmax"
    losses = ["BCE", "Focal"]
    colors = {"BCE": CATEGORICAL_PALETTE[0], "Focal": CATEGORICAL_PALETTE[7]}

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), sharey=True)
    for panel, (ax, ontology) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ontology]
        data = [pd.to_numeric(sub[sub.loss == name][metric], errors="coerce").dropna().to_numpy()
                for name in losses]
        box = ax.boxplot(data, positions=[0, 1], widths=.55, patch_artist=True, showfliers=False,
                         medianprops=dict(color="#111111", linewidth=1.0),
                         boxprops=dict(linewidth=.5, edgecolor="#111111"),
                         whiskerprops=dict(linewidth=.6, color="#111111"),
                         capprops=dict(linewidth=.6, color="#111111"))
        for patch, name in zip(box["boxes"], losses):
            patch.set_facecolor(colors[name])
            patch.set_alpha(.5)
        for i, (name, values) in enumerate(zip(losses, data)):
            if not values.size:
                continue
            rng = np.random.default_rng(abs(hash((ontology, name))) % (2 ** 32))
            ax.scatter(i + rng.uniform(-.12, .12, values.size), values, s=10,
                       color=colors[name], edgecolor="#111111", linewidth=.3, zorder=4)
            ax.text(i, ax.get_ylim()[0], f"n={values.size}", ha="center", va="bottom",
                    fontsize=5, color="#555555")
        ax.set_xticks([0, 1], losses, fontsize=6.5)
        ax.set_title(ONTOLOGY_SHORT[ontology])
        label_panel(ax, chr(97 + panel))
    axes[0].set_ylabel("Validation macro-F$_{max}$", fontsize=7)
    fig.text(.5, -.12,
             "Observational comparison, not a controlled ablation. Loss was one of nine "
             "hyperparameters drawn independently in the 40-trial random search, so BCE and focal "
             "trials also differ in learning rate, dropout, hidden dimension and batch size; the "
             "difference between the two boxes cannot be attributed to the loss alone. Each point "
             "is one trial (single run, no seed replication). A controlled ablation would hold "
             "every other hyperparameter at the selected configuration and vary only the loss. "
             + provenance("src/plot_reviewer_figures.py", "plots/arc_tuning_cafa/trial_metrics.csv"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "figure_loss_bce_vs_focal_search.png", tier)

    print("\nBCE vs focal (validation macro-Fmax, observational):")
    print(df.groupby(["ontology", "loss"])[metric].agg(["mean", "std", "count"]).round(4).to_string())


def verify_auroc_distinct(ablation_csv: Path) -> None:
    """Reviewer 1 comment 10: micro- and macro-AUROC reported as identical."""
    if not ablation_csv.is_file():
        print(f"\nNOTE: {ablation_csv} absent; cannot verify micro/macro AUROC.")
        return
    df = pd.read_csv(ablation_csv)
    if not {"Micro_AUROC", "Macro_AUROC"}.issubset(df.columns):
        return
    identical = np.isclose(df["Micro_AUROC"], df["Macro_AUROC"], atol=1e-9).sum()
    print(f"\nReviewer 1.10 check — micro vs macro AUROC over {len(df)} ablation runs:")
    print(f"  rows where the two are numerically identical: {identical}")
    summary = df.groupby("ontology")[["Micro_AUROC", "Macro_AUROC"]].mean().round(4)
    summary["difference"] = (summary["Micro_AUROC"] - summary["Macro_AUROC"]).round(4)
    print(summary.to_string())
    if identical == 0:
        print("  -> The current pipeline produces clearly distinct values, so the identical "
              "figures in the submitted Table 2 came from the older evaluation, not from this "
              "code. Regenerate Table 2 from the current results.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--trials-csv", type=Path, default=DEFAULT_TRIALS)
    ap.add_argument("--ablation-csv", type=Path,
                    default=REPO / "plots/arc_tuning_cafa/ablations/ablation_test_metrics.csv")
    ap.add_argument("--output-dir", type=Path, default=REPO / "plots/figures/reviewer")
    args = ap.parse_args()

    apply_style()
    print("Palette fingerprint:", assert_palette_locked())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    stats = figure_leakage(args.split_dir / "blast_te_vs_tr.tsv", out, MAIN)
    print("\nResidual identity summary (R1.3 / R2.3):")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    figure_loss(args.trials_csv, out, SUPPLEMENTARY)
    verify_auroc_distinct(args.ablation_csv)

    print("\nStill outstanding for the response letter:")
    print("  R2.6 a CONTROLLED loss ablation (BCE vs focal at the selected configuration,")
    print("       5 seeds, 3 ontologies) does not exist on the locked split and must be run on ARC.")
    print(f"\nWrote reviewer figures to {out}")


if __name__ == "__main__":
    main()
