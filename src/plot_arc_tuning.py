#!/usr/bin/env python3
"""Publication-ready diagnostics for ARC Hybrid hyperparameter tuning.

The landscape is a parallel-coordinates view of the complete seeded random
search, not a two-dimensional projection. All uncertainty bands explicitly
state their meaning, and the archived CC late-epoch downturn is retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

from plot_style import (
    DOUBLE_COLUMN_IN,
    ERROR_KIND_LABEL,
    MODEL_COLOR,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    VALIDATION_METRIC_LABEL,
    apply_style,
    colorblind_audit,
    label_panel,
    mean_and_error,
    savefig,
)

PRIMARY_COLOR = MODEL_COLOR["Hybrid"]
REPLICATE_COLOR = "#6b6a66"
TEST_LABELS = {
    "test_micro_f1_at_validation_threshold": "Micro F1 (validation threshold)",
    "test_micro_fmax": "Micro Fmax",
    "test_macro_fmax": "Macro Fmax",
    "test_micro_aupr": "Micro AUPR",
    "test_macro_aupr": "Macro AUPR",
    "test_micro_auroc": "Micro AUROC",
    "test_macro_auroc": "Macro AUROC",
    "test_micro_fmax_diagnostic": "Micro Fmax diagnostic",
}
PARAMS = [
    ("learning_rate", "LR", "log"),
    ("weight_decay", "WD", "log"),
    ("dropout", "Dropout", "linear"),
    ("hidden_dim", "Hidden", "linear"),
    ("batch_size", "Batch", "linear"),
    ("gradient_clip", "Clip", "linear"),
    ("patience", "Patience", "linear"),
    ("focal_gamma", "Focal gamma", "linear"),
    ("loss", "Loss", "categorical"),
]


def load_trials(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "hybrid_search").glob("trial_*/**/config.json")):
        metric_path = p.parent / "validation_metrics.json"
        if not metric_path.exists():
            continue
        cfg = json.loads(p.read_text())
        met = json.loads(metric_path.read_text())
        rows.append({
            "trial": p.parent.parent.name,
            "ontology": cfg.get("ontology", p.parent.name),
            **{k: cfg.get(k) for k in [
                "learning_rate", "weight_decay", "dropout", "hidden_dim",
                "batch_size", "gradient_clip", "loss", "patience", "focal_gamma",
            ]},
            **{k: met.get(k) for k in [
                "validation_micro_fmax", "validation_macro_fmax",
                "validation_micro_aupr", "validation_macro_aupr",
                "validation_micro_auroc", "validation_macro_auroc",
                "validation_smin",
            ]},
        })
    return pd.DataFrame(rows)


def _finite_limits(values: np.ndarray, lower: float | None = None) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    lo, hi = float(values.min()), float(values.max())
    span = max(hi - lo, 0.02)
    lo = max(0.0, lo - 0.05 * span) if lower is None else lower
    hi = hi + 0.05 * span
    return lo, hi


def _encode_parameter(series: pd.Series, mode: str) -> tuple[np.ndarray, list[str]]:
    if mode == "categorical":
        values = series.fillna("NA").astype(str)
        cats = sorted(values.unique())
        lookup = {value: i / max(len(cats) - 1, 1) for i, value in enumerate(cats)}
        return values.map(lookup).to_numpy(float), cats
    values = pd.to_numeric(series, errors="coerce")
    if mode == "log":
        values = np.log10(values.where(values > 0))
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.full(len(values), 0.5), [f"{lo:g}"]
    return ((values - lo) / (hi - lo)).to_numpy(float), [f"{lo:g}", f"{hi:g}"]


def _format_value(row: pd.Series, key: str) -> str:
    value = row.get(key)
    if pd.isna(value):
        return "NA"
    if key in {"learning_rate", "weight_decay"}:
        return f"{float(value):.1e}".replace("e-0", "e-").replace("e+0", "e+")
    if key in {"dropout", "gradient_clip"}:
        return f"{float(value):g}"
    if key == "focal_gamma":
        return f"gamma={float(value):g}"
    return str(value)


def plot_landscape(df: pd.DataFrame, out: Path, metric: str) -> None:
    """Parallel coordinates for all tuned parameters, coloured by actual score."""
    score = pd.to_numeric(df[metric], errors="coerce")
    valid = df.loc[score.notna()].copy()
    score = pd.to_numeric(valid[metric], errors="coerce")
    lo, hi = float(score.min()), float(score.max())
    norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.3), sharey=True)
    for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = valid[valid.ontology == ont].reset_index(drop=True)
        if sub.empty:
            ax.text(.5, .5, "insufficient data", transform=ax.transAxes, ha="center")
            ax.set_title(ONTOLOGY_SHORT[ont])
            label_panel(ax, chr(97 + panel))
            continue
        x = np.arange(len(PARAMS))
        encoded = []
        tick_labels = []
        for key, label, mode in PARAMS:
            values, labels = _encode_parameter(sub[key], mode)
            encoded.append(values)
            tick_labels.append(labels)
        encoded = np.asarray(encoded).T
        for row_index in range(len(sub)):
            ax.plot(x, encoded[row_index], color=cmap(norm(float(sub.iloc[row_index][metric]))),
                    alpha=.42, linewidth=.55, zorder=2)
        ax.set_xticks(x, [label for _, label, _ in PARAMS], rotation=48, ha="right")
        ax.set_ylim(-.04, 1.04)
        ax.set_yticks([0, .5, 1], ["low", "mid", "high"])
        ax.set_title(f"{ONTOLOGY_SHORT[ont]} (n={len(sub)})")
        ax.grid(axis="y", alpha=.35)
        for position, labels in enumerate(tick_labels):
            if labels:
                ax.text(position, 1.045, labels[-1], ha="center", va="bottom", fontsize=4.8, rotation=55)
                ax.text(position, -.07, labels[0], ha="center", va="top", fontsize=4.8, rotation=55)
        label_panel(ax, chr(97 + panel))
    axes[0].set_ylabel("Normalised hyperparameter value")
    fig.suptitle("Seeded random-search hyperparameter landscape", y=1.04, fontsize=9, fontweight="bold")
    fig.text(.5, -.03, "40-trial seeded random search per ontology; each line is one trial. "
             "Axes are independently normalised; endpoint labels show actual values. "
             "Colour is validation Macro-Fmax.", ha="center", fontsize=6)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=.018, pad=.02,
                 label=f"{VALIDATION_METRIC_LABEL.get(metric, metric)} (actual min={lo:.3f}, max={hi:.3f})")
    savefig(fig, out / "validation_hyperparameter_landscape.png")


def plot_top_trials(df: pd.DataFrame, out: Path, metric: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 4.2), sharey=False)
    for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ont].nlargest(10, metric).copy()
        if sub.empty:
            ax.text(.5, .5, "insufficient data", transform=ax.transAxes, ha="center")
            label_panel(ax, chr(97 + panel))
            continue
        sub = sub.sort_values(metric)
        labels = [
            f"lr={_format_value(row, 'learning_rate')}, wd={_format_value(row, 'weight_decay')}, "
            f"drop={_format_value(row, 'dropout')}, h={_format_value(row, 'hidden_dim')}, "
            f"b={_format_value(row, 'batch_size')}, {_format_value(row, 'loss')}"
            for _, row in sub.iterrows()
        ]
        ax.barh(np.arange(len(sub)), sub[metric], color=PRIMARY_COLOR, edgecolor="#111111", linewidth=.35)
        ax.set_yticks(np.arange(len(sub)), labels, fontsize=5.4)
        ax.set_title(f"{ONTOLOGY_SHORT[ont]}: top 10")
        ax.set_xlabel(VALIDATION_METRIC_LABEL.get(metric, metric))
        ax.grid(axis="x", alpha=.35)
        label_panel(ax, chr(97 + panel))
    fig.text(.5, -.02, "Each bar is one independent trial (one run per trial; no within-trial error bars). "
             "Labels show the sampled hyperparameters.", ha="center", fontsize=6)
    savefig(fig, out / "top_validation_trials.png")


def _bar_panel(ax: plt.Axes, sub: pd.DataFrame, sources: list[str], metrics: list[str],
               colors: dict[str, str]) -> None:
    width = .8 / max(len(sources), 1)
    for index, source in enumerate(sources):
        src = sub[sub.source == source]
        xs = np.arange(len(metrics)) + (index - (len(sources) - 1) / 2) * width
        means, errs = [], []
        for metric in metrics:
            mean, err, _ = mean_and_error(src[metric].to_numpy(), "sd") if metric in src else (np.nan, np.nan, 0)
            means.append(mean)
            errs.append(err)
        ax.bar(xs, means, width=width * .9, yerr=errs, color=colors[source],
               edgecolor="#111111", linewidth=.35,
               error_kw=dict(elinewidth=.7, capsize=2, ecolor="#111111"),
               label=source if len(sources) > 1 else None)
    ax.set_xticks(np.arange(len(metrics)), [TEST_LABELS.get(m, m) for m in metrics],
                  rotation=32, ha="right")


def plot_seed_metrics(root: Path, out: Path) -> None:
    files = sorted((root / "test_evaluation").glob("*/per_seed_metrics.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(path).assign(source=path.parent.name) for path in files], ignore_index=True)
    sources = sorted(df.source.unique())
    colors = {sources[0]: PRIMARY_COLOR, **{source: REPLICATE_COLOR for source in sources[1:]}}
    auroc = [m for m in ["test_micro_auroc", "test_macro_auroc"] if m in df]
    fmax_aupr = [m for m in [
        "test_micro_f1_at_validation_threshold", "test_micro_fmax",
        "test_macro_fmax", "test_micro_aupr", "test_macro_aupr",
        "test_micro_fmax_diagnostic",
    ] if m in df]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_IN, 5.2), sharey="row")
    for row, metrics in enumerate((auroc, fmax_aupr)):
        for panel, (ax, ont) in enumerate(zip(axes[row], ONTOLOGY_ORDER)):
            sub = df[df.ontology == ont]
            if sub.empty:
                ax.text(.5, .5, "insufficient data", transform=ax.transAxes, ha="center")
                continue
            _bar_panel(ax, sub, sources, metrics, colors)
            ax.set_title(ONTOLOGY_SHORT[ont])
            label_panel(ax, chr(97 + row * 3 + panel))
        if metrics:
            axes[row, 0].set_ylabel(f"{'AUROC family' if row == 0 else 'Fmax/AUPR family'}\nmean ± SD (n≤5 seeds)")
    fig.text(.5, 1.01, "(a) AUROC-family metrics", ha="center", fontsize=8, fontweight="bold")
    fig.text(.5, .49, "(b) Fmax/AUPR-family metrics", ha="center", fontsize=8, fontweight="bold")
    fig.text(.5, -.035, "Error bars = SD across at most five confirmation seeds. "
             "Micro Fmax diagnostic checks the independent test-swept implementation; "
             "micro F1 at the validation threshold is the fixed-threshold sanity/leakage check. "
             "They are complementary, not duplicate metrics.", ha="center", fontsize=6)
    if len(sources) > 1:
        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    savefig(fig, out / "test_metrics_mean_sd.png")

    smin = "test_smin" if "test_smin" in df else None
    if smin:
        global_max = float(np.nanmax(df[smin].to_numpy(dtype=float)))
        ymax = max(global_max * 1.10, 1.0)
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.8), sharey=True)
        for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
            sub = df[df.ontology == ont]
            _bar_panel(ax, sub, sources, [smin], colors)
            ax.set_title(ONTOLOGY_SHORT[ont])
            ax.set_ylim(0, ymax)
            ax.text(.01, 1.03, "lower is better", transform=ax.transAxes,
                    fontsize=6, color="#555555", va="bottom", clip_on=False)
            label_panel(ax, chr(97 + panel))
        axes[0].set_ylabel(f"Smin mean ± SD (n≤5 seeds)")
        fig.text(.5, -.035, f"Error bars = SD across at most five seeds; shared y-axis maximum = {ymax:.1f} "
                 "(global maximum plus 10% headroom).", ha="center", fontsize=6)
        if len(sources) > 1:
            axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        savefig(fig, out / "test_smin_mean_sd.png")


def plot_histories(root: Path, out: Path) -> None:
    frames = []
    for path in sorted((root / "five_seed_hybrid").glob("*/seed_*/history.csv")):
        data = pd.read_csv(path)
        if "validation_micro_fmax" not in data:
            continue
        data["ontology"] = path.parent.parent.name
        data["seed"] = path.parent.name
        frames.append(data[["ontology", "seed", "epoch", "validation_micro_fmax"]])
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    all_values = df.validation_micro_fmax.to_numpy(dtype=float)
    ymin, ymax = _finite_limits(all_values, lower=0.0)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.9), sharey=True)
    for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ont]
        for _, seed in sub.groupby("seed"):
            ax.plot(seed.epoch, seed.validation_micro_fmax, color=PRIMARY_COLOR, alpha=.30,
                    linewidth=.75, label="_seed")
        grouped = sub.groupby("epoch").validation_micro_fmax.agg(["mean", "std"])
        ax.fill_between(grouped.index, grouped["mean"] - grouped["std"].fillna(0),
                        grouped["mean"] + grouped["std"].fillna(0), color=PRIMARY_COLOR,
                        alpha=.14, linewidth=0, label="_sd")
        ax.plot(grouped.index, grouped["mean"], color=PRIMARY_COLOR, linewidth=2.0, label="_mean")
        ax.set_ylim(ymin, ymax)
        ax.set_title(ONTOLOGY_SHORT[ont])
        ax.set_xlabel("Epoch")
        label_panel(ax, chr(97 + panel))
        if ont == "cellular_component" and len(sub):
            late = grouped.tail(3)
            if len(late) >= 2 and late["mean"].iloc[-1] < late["mean"].iloc[0]:
                ax.annotate("late-epoch drop retained", xy=(late.index[-1], late["mean"].iloc[-1]),
                            xytext=(-45, 12), textcoords="offset points", fontsize=5.5,
                            arrowprops=dict(arrowstyle="->", linewidth=.6))
    axes[0].set_ylabel("Validation micro-Fmax")
    handles = [
        Line2D([0], [0], color=PRIMARY_COLOR, lw=.8, alpha=.30, label="individual seed"),
        Line2D([0], [0], color=PRIMARY_COLOR, lw=2.0, label="mean"),
        plt.Rectangle((0, 0), 1, 1, facecolor=PRIMARY_COLOR, alpha=.14, label="± SD"),
    ]
    axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.text(.5, -.035, f"Shared y-axis across MF/BP/CC: [{ymin:.3f}, {ymax:.3f}]. "
             "The shaded region is ± SD across seeds; the CC late-epoch downturn is shown, not smoothed.", ha="center", fontsize=6)
    savefig(fig, out / "five_seed_validation_curves.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--err", choices=["sd", "sem", "ci95"], default="sd")
    args = ap.parse_args()
    apply_style()
    print("Colour audit:", colorblind_audit())
    root = args.tuning_root.resolve()
    out = (args.output_dir or Path("plots") / root.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = load_trials(root)
    if df.empty:
        raise SystemExit(f"No trial metrics found under {root / 'hybrid_search'}")
    df.to_csv(out / "trial_metrics.csv", index=False)
    metric = "validation_macro_fmax" if df["validation_macro_fmax"].notna().any() else "validation_micro_fmax"
    plot_landscape(df, out, metric)
    plot_top_trials(df, out, metric)
    plot_seed_metrics(root, out)
    plot_histories(root, out)
    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()
