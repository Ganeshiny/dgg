#!/usr/bin/env python3
"""Make supplementary plots from the ARC hybrid tuning artifacts.

The script only reads JSON/CSV files; model checkpoints and graph caches are
not required.  Run once for each tuning root, for example:

  python src/plot_arc_tuning.py --tuning-root arc_tuning
  python src/plot_arc_tuning.py --tuning-root arc_tuning_cafa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_style import (
    DOUBLE_COLUMN_IN,
    ERROR_KIND_LABEL,
    MODEL_COLOR,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    VALIDATION_METRIC_HIGHER_IS_BETTER,
    VALIDATION_METRIC_LABEL,
    annotate_insufficient_data,
    apply_style,
    jitter,
    label_panel,
    mean_and_error,
    savefig,
)

# This script is single-model (Hybrid) tuning/QA diagnostics, not the
# multi-architecture ablation comparison (see plot_arc_ablations.py for that).
PRIMARY_COLOR = MODEL_COLOR["Hybrid"]
REPLICATE_COLOR = "#898781"  # neutral gray for a same-model CPU/GPU reproducibility replicate


def load_trials(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "hybrid_search").glob("trial_*/**/config.json")):
        trial_dir = p.parent
        metric_path = trial_dir / "validation_metrics.json"
        if not metric_path.exists():
            continue
        cfg = json.loads(p.read_text())
        met = json.loads(metric_path.read_text())
        rows.append({
            "trial": trial_dir.parent.name,
            "ontology": cfg.get("ontology", trial_dir.name),
            **{k: cfg.get(k) for k in ["learning_rate", "weight_decay", "dropout", "hidden_dim", "batch_size", "loss", "patience", "focal_gamma"]},
            **{k: met.get(k) for k in ["validation_micro_fmax", "validation_macro_fmax", "validation_micro_aupr", "validation_macro_aupr", "validation_micro_auroc", "validation_macro_auroc", "validation_smin"]},
        })
    return pd.DataFrame(rows)


def plot_landscape(df: pd.DataFrame, out: Path, metric: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.6), constrained_layout=True, sharey=True)
    sc = None
    for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ont].copy()
        if sub.empty:
            annotate_insufficient_data(ax); ax.set_title(ONTOLOGY_SHORT[ont]); label_panel(ax, chr(97 + panel)); continue
        sc = ax.scatter(sub.learning_rate, sub.weight_decay, c=sub[metric], cmap="viridis", s=26,
                         edgecolor="white", linewidth=.3, vmin=df[metric].min(), vmax=df[metric].max())
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ONTOLOGY_SHORT[ont]); ax.set_xlabel("Learning rate")
        label_panel(ax, chr(97 + panel))
    axes[0].set_ylabel("Weight decay")
    if sc is not None:
        fig.colorbar(sc, ax=axes, fraction=.025, pad=.02, label=VALIDATION_METRIC_LABEL.get(metric, metric))
    savefig(fig, out / "validation_hyperparameter_landscape.png")


def plot_top_trials(df: pd.DataFrame, out: Path, metric: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.0), sharey=False)
    for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ont].nlargest(10, metric).copy()
        if sub.empty:
            annotate_insufficient_data(ax); ax.set_title(ONTOLOGY_SHORT[ont]); label_panel(ax, chr(97 + panel)); continue
        sub = sub.sort_values(metric)
        ax.barh(sub.trial, sub[metric], color=PRIMARY_COLOR, edgecolor="#0b0b0b", linewidth=0.4)
        ax.set_title(f"{ONTOLOGY_SHORT[ont]}: top 10 trials")
        ax.set_xlabel(VALIDATION_METRIC_LABEL.get(metric, metric))
        label_panel(ax, chr(97 + panel))
    savefig(fig, out / "top_validation_trials.png")


def plot_seed_metrics(root: Path, out: Path, err_kind: str = "sd") -> None:
    """Mean +/- error of Hybrid test-set metrics across the 5 confirmation
    seeds, one panel per ontology. Every model-directory found under
    test_evaluation/ is loaded (not just the first match) and shown as its
    own series — e.g. "hybrid" (GPU) vs "hybrid_cpu" is a reproducibility
    replicate of the *same* model/seeds, not a different model, but which
    directory a bare glob()[0] returns is filesystem-order-dependent, so
    silently picking one is not reproducible either. Bounded 0-1 metrics
    (Fmax/AUPR/AUROC) and Smin (unbounded, lower-is-better) are always split
    into separate panels/figures — they were previously sharing one y-axis,
    which made every bounded metric render as a flat line at zero next to
    a much larger Smin value.
    """
    files = sorted((root / "test_evaluation").glob("*/per_seed_metrics.csv"))
    if not files:
        return
    frames = []
    for path in files:
        d = pd.read_csv(path)
        d["source"] = path.parent.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    sources = sorted(df["source"].unique())
    colors = {sources[0]: PRIMARY_COLOR}
    for extra in sources[1:]:
        colors[extra] = REPLICATE_COLOR

    bounded = [c for c in ["test_micro_fmax", "test_macro_fmax", "test_micro_aupr", "test_macro_aupr",
                            "test_micro_auroc", "test_macro_auroc", "test_micro_f1_at_validation_threshold",
                            "test_micro_fmax_diagnostic"] if c in df]
    smin = "test_smin" if "test_smin" in df else None
    if not bounded and not smin:
        print(f"plot_seed_metrics: no recognised metric columns in {files}, skipping")
        return

    def _panel(ax, sub: pd.DataFrame, metrics: list[str]) -> None:
        width = 0.8 / max(len(sources), 1)
        for i, source in enumerate(sources):
            src = sub[sub.source == source]
            xs = np.arange(len(metrics)) + (i - (len(sources) - 1) / 2) * width
            means, errs = [], []
            for m in metrics:
                mean, err, _ = mean_and_error(src[m].to_numpy(), err_kind) if m in src else (np.nan, np.nan, 0)
                means.append(mean); errs.append(err)
            ax.bar(xs, means, width=width * 0.9, yerr=errs, color=colors[source], edgecolor="#0b0b0b",
                   linewidth=0.4, error_kw=dict(elinewidth=0.8, capsize=2.0, ecolor="#0b0b0b"),
                   label=source if len(sources) > 1 else None)
        ax.set_xticks(np.arange(len(metrics)), [VALIDATION_METRIC_LABEL.get(m, m.replace("test_", "").replace("_", " ")) for m in metrics], rotation=30, ha="right")

    if bounded:
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.2), sharey=True)
        for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
            sub = df[df.ontology == ont]
            if sub.empty:
                annotate_insufficient_data(ax); ax.set_title(ONTOLOGY_SHORT[ont]); label_panel(ax, chr(97 + panel)); continue
            _panel(ax, sub, bounded)
            ax.set_title(ONTOLOGY_SHORT[ont]); label_panel(ax, chr(97 + panel))
        axes[0].set_ylabel(f"Test score ({ERROR_KIND_LABEL[err_kind]}, n≤5 seeds)")
        if len(sources) > 1:
            axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        savefig(fig, out / "test_metrics_mean_sd.png")

    if smin:
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.6), sharey=True)
        for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
            sub = df[df.ontology == ont]
            if sub.empty:
                annotate_insufficient_data(ax); ax.set_title(ONTOLOGY_SHORT[ont]); label_panel(ax, chr(97 + panel)); continue
            _panel(ax, sub, [smin])
            ax.set_title(ONTOLOGY_SHORT[ont]); label_panel(ax, chr(97 + panel))
            ax.text(0.03, 0.96, "lower is better", transform=ax.transAxes, fontsize=6, color="#898781", va="top")
        axes[0].set_ylabel(f"S$_{{min}}$ ({ERROR_KIND_LABEL[err_kind]}, n≤5 seeds)")
        if len(sources) > 1:
            axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        savefig(fig, out / "test_smin_mean_sd.png")


def plot_histories(root: Path, out: Path) -> None:
    frames = []
    for p in sorted((root / "five_seed_hybrid").glob("*/seed_*/history.csv")):
        d = pd.read_csv(p)
        if "validation_micro_fmax" not in d: continue
        d["ontology"] = p.parent.parent.name; d["seed"] = p.parent.name
        frames.append(d[["ontology", "seed", "epoch", "validation_micro_fmax"]])
    if not frames: return
    df = pd.concat(frames)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.6), sharey=True)
    for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ont]
        if sub.empty:
            annotate_insufficient_data(ax); label_panel(ax, chr(97 + panel)); continue
        for _, seed in sub.groupby("seed"):
            ax.plot(seed.epoch, seed.validation_micro_fmax, color=PRIMARY_COLOR, alpha=.25, linewidth=0.8)
        g = sub.groupby("epoch").validation_micro_fmax.agg(["mean", "std"])
        ax.plot(g.index, g["mean"], color=PRIMARY_COLOR, linewidth=1.8, label="mean")
        ax.fill_between(g.index, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), color=PRIMARY_COLOR, alpha=.15, linewidth=0)
        ax.set_title(ONTOLOGY_SHORT[ont]); ax.set_xlabel("Epoch")
        label_panel(ax, chr(97 + panel))
    axes[0].set_ylabel("Validation micro-F$_{max}$")
    savefig(fig, out / "five_seed_validation_curves.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--err", choices=["sd", "sem", "ci95"], default="sd")
    args = ap.parse_args()
    apply_style()
    root = args.tuning_root.resolve()
    out = (args.output_dir or Path("plots") / root.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = load_trials(root)
    if df.empty: raise SystemExit(f"No trial metrics found under {root / 'hybrid_search'}")
    df.to_csv(out / "trial_metrics.csv", index=False)
    metric = "validation_macro_fmax" if df["validation_macro_fmax"].notna().any() else "validation_micro_fmax"
    plot_landscape(df, out, metric); plot_top_trials(df, out, metric)
    plot_seed_metrics(root, out, args.err); plot_histories(root, out)
    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()
