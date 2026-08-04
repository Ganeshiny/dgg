#!/usr/bin/env python3
"""Hyperparameter-search and confirmation-run figures for the ARC pipeline.

Search provenance (read from src/generate_hybrid_trials.py, not assumed):
a seeded RANDOM search of 40 trials, search seed 20260714 — not a Cartesian
grid and not Bayesian. Learning rate and weight decay are drawn log-uniformly
from [1e-5, 3e-3] and [1e-7, 1e-2]; dropout, hidden dim, batch size, gradient
clip, patience, loss and focal gamma are categorical draws. Each trial is run
once per ontology, so trial-level bars have no error bars by construction.

Run:
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
from matplotlib.ticker import MaxNLocator

from plot_style import (
    DOUBLE_COLUMN_IN,
    ERROR_CAPTION,
    ERROR_KIND_LABEL,
    MODEL_COLOR,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    SUPPLEMENTARY,
    VALIDATION_METRIC_LABEL,
    annotate_insufficient_data,
    apply_style,
    assert_palette_locked,
    label_panel,
    label_horizontal_bars,
    label_vertical_bars,
    mean_and_error,
    provenance,
    report_colorblind_audit,
    savefig,
)

PRIMARY = MODEL_COLOR["Hybrid"]
REPLICATE = "#898781"

SEARCH_CAPTION = ("Seeded random search, 40 trials (search seed 20260714); learning rate and "
                  "weight decay log-uniform over [1e-5, 3e-3] and [1e-7, 1e-2], remaining "
                  "hyperparameters categorical. Not a grid and not Bayesian.")

# The hyperparameters actually varied by the search, in the order they are
# drawn. focal_gamma is conditional on loss == "Focal" and is therefore
# undefined for BCE trials; it is plotted but flagged as conditional.
TUNED_PARAMS = ["learning_rate", "weight_decay", "dropout", "hidden_dim",
                "batch_size", "gradient_clip", "patience", "loss", "focal_gamma"]
LOG_PARAMS = {"learning_rate", "weight_decay"}
PARAM_LABEL = {
    "learning_rate": "Learning\nrate", "weight_decay": "Weight\ndecay",
    "dropout": "Dropout", "hidden_dim": "Hidden\ndim", "batch_size": "Batch\nsize",
    "gradient_clip": "Grad\nclip", "patience": "Patience", "loss": "Loss",
    "focal_gamma": "Focal\ngamma",
}


def load_trials(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "hybrid_search").glob("trial_*/**/config.json")):
        trial_dir = path.parent
        metric_path = trial_dir / "validation_metrics.json"
        if not metric_path.exists():
            continue
        cfg = json.loads(path.read_text())
        met = json.loads(metric_path.read_text())
        rows.append({
            "trial": trial_dir.parent.name,
            "ontology": cfg.get("ontology", trial_dir.name),
            **{k: cfg.get(k) for k in TUNED_PARAMS},
            **{k: met.get(k) for k in [
                "validation_micro_fmax", "validation_macro_fmax",
                "validation_micro_aupr", "validation_macro_aupr",
                "validation_micro_auroc", "validation_macro_auroc", "validation_smin"]},
        })
    return pd.DataFrame(rows)


def _fmt(value, name: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if name in LOG_PARAMS:
        return f"{value:.1e}".replace("e-0", "e-")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def trial_label(row: pd.Series) -> str:
    """Compact hyperparameter signature replacing the opaque trial_XXX id."""
    parts = [f"lr={_fmt(row.learning_rate, 'learning_rate')}",
             f"wd={_fmt(row.weight_decay, 'weight_decay')}",
             f"do={_fmt(row.dropout, 'dropout')}",
             f"h={_fmt(row.hidden_dim, 'hidden_dim')}",
             f"bs={_fmt(row.batch_size, 'batch_size')}"]
    loss = row.get("loss")
    if loss == "Focal":
        parts.append(f"Focal γ={_fmt(row.get('focal_gamma'), 'focal_gamma')}")
    else:
        parts.append("BCE")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 1. Five-seed validation curves
# ---------------------------------------------------------------------------
def plot_histories(root: Path, out: Path, tier: str = SUPPLEMENTARY) -> None:
    """Mean is drawn only where every seed is still running.

    Seeds early-stop at different epochs, so a naive mean over 'whatever is
    left' silently becomes a single seed's noisy trace at the tail, with an SD
    band that collapses to zero. That is what produced the apparent
    late-epoch drop in CC (and a spike in MF): at the final epochs only one of
    five seeds contributes. The mean is therefore truncated at the last epoch
    where all five seeds are present; beyond it the individual traces continue
    and the region is shaded and labelled so the thinning is visible rather
    than smoothed over.
    """
    frames = []
    for path in sorted((root / "five_seed_hybrid").glob("*/seed_*/history.csv")):
        d = pd.read_csv(path)
        if "validation_micro_fmax" not in d:
            continue
        d["ontology"] = path.parent.parent.name
        d["seed"] = path.parent.name
        frames.append(d[["ontology", "seed", "epoch", "validation_micro_fmax"]])
    if not frames:
        return
    df = pd.concat(frames)

    lo = float(df.validation_micro_fmax.min())
    hi = float(df.validation_micro_fmax.max())
    pad = (hi - lo) * .06
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), sharey=True)
    n_seeds_total = df.groupby("ontology")["seed"].nunique().max()
    for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ont]
        if sub.empty:
            annotate_insufficient_data(ax)
            label_panel(ax, chr(97 + panel))
            continue
        for _, seed_df in sub.groupby("seed"):
            ax.plot(seed_df.epoch, seed_df.validation_micro_fmax,
                    color=PRIMARY, alpha=.30, linewidth=.7, zorder=2)
        per_epoch = sub.groupby("epoch").validation_micro_fmax.agg(["mean", "std", "count"])
        full = per_epoch[per_epoch["count"] == sub.seed.nunique()]
        if not full.empty:
            cutoff = int(full.index.max())
            ax.plot(full.index, full["mean"], color=PRIMARY, linewidth=1.9, zorder=4)
            ax.fill_between(full.index, full["mean"] - full["std"].fillna(0),
                            full["mean"] + full["std"].fillna(0),
                            color=PRIMARY, alpha=.18, linewidth=0, zorder=3)
            tail = per_epoch[per_epoch.index > cutoff]
            if not tail.empty:
                ax.axvspan(cutoff, float(per_epoch.index.max()), color="#bbbbbb",
                           alpha=.22, linewidth=0, zorder=1)
                ax.text(cutoff, hi + pad * .3, f"  <{sub.seed.nunique()} seeds",
                        fontsize=5.0, color="#555555", va="top", ha="left")
        ax.set_title(ONTOLOGY_SHORT[ont])
        ax.set_xlabel("Epoch")
        label_panel(ax, chr(97 + panel))
    axes[0].set_ylabel("Validation micro-F$_{max}$")
    axes[0].set_ylim(lo - pad, hi + pad)
    handles = [
        Line2D([0], [0], color=PRIMARY, alpha=.30, linewidth=.7, label="Individual seed (n=5)"),
        Line2D([0], [0], color=PRIMARY, linewidth=1.9, label="Mean of all 5 seeds"),
        Line2D([0], [0], color=PRIMARY, alpha=.18, linewidth=6, label="± s.d."),
        Line2D([0], [0], color="#bbbbbb", alpha=.5, linewidth=6, label="Fewer than 5 seeds"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, .95),
               frameon=False, fontsize=6.2)
    fig.text(.5, -.10,
             "Shared y-axis across panels; all three show the same metric on one scale. "
             "The mean and s.d. are drawn only while all five seeds are still training — seeds "
             "early-stop at different epochs (e.g. CC at 18/20/20/23/28), so the shaded tail is "
             "a shrinking subset and its apparent drop is an averaging artifact, not instability. "
             + provenance("src/plot_arc_tuning.py", f"{root.name}/five_seed_hybrid/*/seed_*/history.csv"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "five_seed_validation_curves.png", tier)


# ---------------------------------------------------------------------------
# 2 + 3. Test metrics, split by metric family; Smin separate with shared limits
# ---------------------------------------------------------------------------
AUROC_FAMILY = ["test_micro_auroc", "test_macro_auroc"]
FMAX_AUPR_FAMILY = ["test_micro_fmax", "test_macro_fmax", "test_micro_aupr",
                    "test_macro_aupr", "test_micro_f1_at_validation_threshold"]

DIAGNOSTIC_NOTE = (
    "'micro f1 at validation threshold' is the honest operating point: the decision threshold is "
    "selected on validation and applied unchanged to test. 'micro fmax' scans thresholds on the "
    "test set itself and is therefore an optimistic upper bound; the gap between the two is the "
    "cost of threshold transfer. The former 'micro fmax diagnostic' column is dropped here — it is "
    "numerically identical to 'micro fmax' (same test-side threshold scan), not an independent check."
)


def _load_seed_metrics(root: Path) -> pd.DataFrame | None:
    files = sorted((root / "test_evaluation").glob("*/per_seed_metrics.csv"))
    if not files:
        return None
    frames = []
    for path in files:
        d = pd.read_csv(path)
        d["source"] = path.parent.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _grouped_bars(ax, sub: pd.DataFrame, metrics: list[str], sources: list[str],
                  colors: dict, err_kind: str) -> None:
    width = .8 / max(len(sources), 1)
    for i, source in enumerate(sources):
        src = sub[sub.source == source]
        xs = np.arange(len(metrics)) + (i - (len(sources) - 1) / 2) * width
        means, errs = [], []
        for m in metrics:
            if m in src:
                mean, err, _ = mean_and_error(src[m].to_numpy(), err_kind)
            else:
                mean, err = np.nan, np.nan
            means.append(mean)
            errs.append(err)
        bars = ax.bar(xs, means, width=width * .9, yerr=errs, color=colors[source],
                      edgecolor="#111111", linewidth=.4,
                      error_kw=dict(elinewidth=.7, capsize=2, ecolor="#111111"),
                      label=source if len(sources) > 1 else None)
        label_vertical_bars(ax, bars, means, errs, fontsize=4.8, rotation=90)
    ax.set_xticks(np.arange(len(metrics)),
                  [m.replace("test_", "").replace("_", " ") for m in metrics],
                  rotation=32, ha="right")


def plot_seed_metrics(root: Path, out: Path, err_kind: str = "sd",
                      tier: str = SUPPLEMENTARY) -> None:
    df = _load_seed_metrics(root)
    if df is None:
        return
    sources = sorted(df["source"].unique())
    colors = {sources[0]: PRIMARY}
    for extra in sources[1:]:
        colors[extra] = REPLICATE

    auroc = [c for c in AUROC_FAMILY if c in df]
    fmax = [c for c in FMAX_AUPR_FAMILY if c in df]
    if auroc or fmax:
        rows = [g for g in (fmax, auroc) if g]
        fig, axes = plt.subplots(len(rows), 3, figsize=(DOUBLE_COLUMN_IN, 2.5 * len(rows)),
                                 squeeze=False)
        panel = 0
        for r, metrics in enumerate(rows):
            # Each family gets its own y-scale; AUROC sits near 0.5-0.85 and
            # would flatten the Fmax/AUPR family if forced onto one axis.
            vals = pd.concat([df[m] for m in metrics]).to_numpy(float)
            vals = vals[np.isfinite(vals)]
            lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0., 1.)
            pad = max((hi - lo) * .12, .02)
            for c, ont in enumerate(ONTOLOGY_ORDER):
                ax = axes[r][c]
                sub = df[df.ontology == ont]
                if sub.empty:
                    annotate_insufficient_data(ax)
                else:
                    ax.set_ylim(max(0., lo - pad), min(1., hi + pad))
                    _grouped_bars(ax, sub, metrics, sources, colors, err_kind)
                if r == 0:
                    ax.set_title(ONTOLOGY_SHORT[ont])
                label_panel(ax, chr(97 + panel))
                panel += 1
            axes[r][0].set_ylabel("F$_{max}$ / AUPR family" if metrics is rows[0] else "AUROC family")
        if len(sources) > 1:
            axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6.2)
        fig.text(.5, -.06, f"{ERROR_CAPTION[err_kind]}. Metric families are on separate axes "
                 f"with independent y-limits — AUROC occupies a much higher range and would "
                 f"flatten the F$_{{max}}$/AUPR family on a shared scale. {DIAGNOSTIC_NOTE}",
                 ha="center", fontsize=5.2, wrap=True)
        savefig(fig, out / "test_metrics_mean_sd.png", tier)

    if "test_smin" in df:
        vals = df["test_smin"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        ymax = float(vals.max()) * 1.10 if vals.size else 1.0
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.4), sharey=True)
        for panel, (ax, ont) in enumerate(zip(axes, ONTOLOGY_ORDER)):
            sub = df[df.ontology == ont]
            if sub.empty:
                annotate_insufficient_data(ax)
            else:
                ax.set_ylim(0, ymax)
                _grouped_bars(ax, sub, ["test_smin"], sources, colors, err_kind)
            ax.set_title(ONTOLOGY_SHORT[ont])
            label_panel(ax, chr(97 + panel))
        axes[0].set_ylabel("S$_{min}$")
        if len(sources) > 1:
            axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6.2)
        fig.text(.5, -.08, f"{ERROR_CAPTION[err_kind]}. All three panels share one y-axis "
                 f"(0 to {ymax:.1f}, the global maximum plus 10% headroom) so cross-ontology "
                 f"S$_{{min}}$ magnitudes are directly comparable. "
                 + provenance("src/plot_arc_tuning.py", f"{root.name}/test_evaluation/*/per_seed_metrics.csv"),
                 ha="center", fontsize=5.2, wrap=True)
        savefig(fig, out / "test_smin_mean_sd.png", tier)


# ---------------------------------------------------------------------------
# 4. Top trials, labelled by hyperparameters rather than opaque ids
# ---------------------------------------------------------------------------
def plot_top_trials(df: pd.DataFrame, out: Path, metric: str,
                    tier: str = SUPPLEMENTARY, top: int = 10) -> None:
    """Ontologies stacked vertically so the value axis is not squeezed.

    Side-by-side panels gave each subplot roughly a third of the page width,
    of which the hyperparameter labels consumed most — leaving a value axis so
    narrow its ticks overprinted each other. One full-width row per ontology
    gives the labels their own margin and the bars the remaining width.
    """
    fig, axes = plt.subplots(len(ONTOLOGY_ORDER), 1,
                             figsize=(DOUBLE_COLUMN_IN, 2.05 * len(ONTOLOGY_ORDER)))
    for panel, (ax, ont) in enumerate(zip(np.atleast_1d(axes), ONTOLOGY_ORDER)):
        sub = df[df.ontology == ont].nlargest(top, metric).copy()
        if sub.empty:
            annotate_insufficient_data(ax)
            label_panel(ax, chr(97 + panel))
            continue
        sub = sub.sort_values(metric)
        labels = [trial_label(r) for _, r in sub.iterrows()]
        bars = ax.barh(np.arange(len(sub)), sub[metric], color=PRIMARY,
                       edgecolor="#111111", linewidth=.4)
        ax.set_yticks(np.arange(len(sub)), labels, fontsize=5.2)
        ax.set_title(f"{ONTOLOGY_SHORT[ont]}: top {top} trials", loc="left")
        if panel == len(ONTOLOGY_ORDER) - 1:
            ax.set_xlabel(VALIDATION_METRIC_LABEL.get(metric, metric))
        lo = float(sub[metric].min()) * .97
        ax.set_xlim(lo, float(sub[metric].max()) * 1.01)
        label_horizontal_bars(ax, bars, sub[metric].to_numpy(float), fontsize=5.0)
        # A narrow window puts default ticks close enough to overprint; cap
        # the count and let matplotlib choose round values inside it.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.tick_params(axis="x", labelsize=5.4)
        label_panel(ax, chr(97 + panel))
    fig.text(.5, -.07, f"{SEARCH_CAPTION} Each trial was run once per ontology, so no error bars "
             f"are possible at trial level; seed-level dispersion is quantified separately in the "
             f"five-seed confirmation runs. Bars are labelled by hyperparameters rather than trial "
             f"id. x-axis starts near the minimum shown to resolve differences between top trials. "
             + provenance("src/plot_arc_tuning.py", "hybrid_search/trial_*/*/validation_metrics.json"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "top_validation_trials.png", tier)


# ---------------------------------------------------------------------------
# 5. Search landscape: parallel coordinates over ALL tuned hyperparameters
# ---------------------------------------------------------------------------
def _encode(df: pd.DataFrame, param: str) -> tuple[np.ndarray, list, list]:
    """Return numeric positions in [0,1] plus tick positions/labels."""
    series = df[param]
    if series.dropna().map(lambda v: isinstance(v, (int, float, np.number))).all() and series.notna().any():
        values = pd.to_numeric(series, errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.full(len(series), .5), [], []
        if param in LOG_PARAMS:
            values = np.log10(np.where(values > 0, values, np.nan))
            finite = values[np.isfinite(values)]
        lo, hi = float(finite.min()), float(finite.max())
        span = hi - lo or 1.0
        scaled = (values - lo) / span
        # Only the endpoints are labelled: three labels per axis collided with
        # the neighbouring axis at this figure width.
        ticks = np.linspace(0, 1, 2)
        if param in LOG_PARAMS:
            labels = [f"$10^{{{lo + t * span:.1f}}}$" for t in ticks]
        else:
            labels = [f"{lo + t * span:g}" for t in ticks]
        return scaled, list(ticks), labels
    categories = sorted(series.dropna().astype(str).unique())
    index = {c: i for i, c in enumerate(categories)}
    span = max(len(categories) - 1, 1)
    scaled = series.astype(str).map(lambda v: index.get(v, np.nan)).to_numpy(float) / span
    return scaled, list(np.arange(len(categories)) / span), categories


def plot_parallel_coordinates(df: pd.DataFrame, out: Path, metric: str,
                              tier: str = SUPPLEMENTARY) -> None:
    params = [p for p in TUNED_PARAMS if p in df and df[p].notna().any()]
    values = pd.to_numeric(df[metric], errors="coerce")
    vmin, vmax = float(values.min()), float(values.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(len(ONTOLOGY_ORDER), 1,
                             figsize=(DOUBLE_COLUMN_IN, 2.35 * len(ONTOLOGY_ORDER)))
    for panel, (ax, ont) in enumerate(zip(np.atleast_1d(axes), ONTOLOGY_ORDER)):
        sub = df[df.ontology == ont]
        if sub.empty:
            annotate_insufficient_data(ax)
            label_panel(ax, chr(97 + panel))
            continue
        encoded, ticks_by_param = {}, {}
        for p in params:
            scaled, ticks, labels = _encode(sub, p)
            encoded[p] = scaled
            ticks_by_param[p] = (ticks, labels)
        order = np.argsort(pd.to_numeric(sub[metric], errors="coerce").to_numpy(float))
        for row in order:  # best trials drawn last, on top
            y = [encoded[p][row] for p in params]
            score = float(pd.to_numeric(sub[metric], errors="coerce").to_numpy(float)[row])
            if not np.isfinite(score):
                continue
            ax.plot(np.arange(len(params)), y, color=cmap(norm(score)),
                    alpha=.75, linewidth=.85, solid_capstyle="round")
        for i, p in enumerate(params):
            ax.axvline(i, color="#cccccc", linewidth=.6, zorder=0)
            ticks, labels = ticks_by_param[p]
            for t, lab in zip(ticks, labels):
                ax.text(i - .04, t, lab, fontsize=4.2, color="#555555",
                        ha="right", va="center")
        ax.set_xticks(np.arange(len(params)), [PARAM_LABEL.get(p, p) for p in params], fontsize=5.6)
        ax.set_yticks([])
        ax.set_ylim(-.08, 1.08)
        ax.set_xlim(-.6, len(params) - .4)
        ax.set_title(ONTOLOGY_SHORT[ont])
        ax.grid(False)
        label_panel(ax, chr(97 + panel))
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Horizontal colourbar below the stack: a vertical bar spanning three tall
    # panels overlapped the middle panel's axis labels.
    fig.tight_layout()
    cax = fig.add_axes([0.25, -0.005, 0.5, 0.012])
    fig.colorbar(mappable, cax=cax, orientation="horizontal",
                 label=f"{VALIDATION_METRIC_LABEL.get(metric, metric)} "
                       f"(scale clipped to observed {vmin:.3f}–{vmax:.3f})")
    cax.tick_params(labelsize=5.4)
    cax.xaxis.label.set_size(6)
    fig.text(.5, -.06, f"{SEARCH_CAPTION} Each line is one trial across all "
             f"{len(params)} tuned hyperparameters; colour is validation "
             f"{VALIDATION_METRIC_LABEL.get(metric, metric)} with the scale clipped to the observed "
             f"range rather than 0–1, so mid-tier trials remain distinguishable. focal_gamma is "
             f"conditional on the Focal loss and is undefined for BCE trials. "
             + provenance("src/plot_arc_tuning.py", "hybrid_search/trial_*/*/config.json"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "validation_hyperparameter_parallel_coordinates.png", tier)


def plot_param_small_multiples(df: pd.DataFrame, out: Path, metric: str,
                               tier: str = SUPPLEMENTARY) -> None:
    """One 1-D scatter per tuned hyperparameter — the readable companion."""
    params = [p for p in TUNED_PARAMS if p in df and df[p].notna().any()]
    ncols = 3
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_COLUMN_IN, 1.85 * nrows),
                             squeeze=False)
    values_all = pd.to_numeric(df[metric], errors="coerce")
    lo, hi = float(values_all.min()), float(values_all.max())
    pad = (hi - lo) * .08
    # Ontology is encoded by MARKER SHAPE in one neutral ink, never by the
    # categorical palette: those hues denote models everywhere else in the
    # figure set, and reusing them here would make a colour mean two things.
    ont_marker = {"molecular_function": "o", "biological_process": "s", "cellular_component": "^"}
    for i, p in enumerate(params):
        ax = axes[i // ncols][i % ncols]
        for ont in ONTOLOGY_ORDER:
            sub = df[df.ontology == ont]
            if sub.empty:
                continue
            y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(float)
            raw = sub[p]
            if raw.dropna().map(lambda v: isinstance(v, (int, float, np.number))).all():
                x = pd.to_numeric(raw, errors="coerce").to_numpy(float)
            else:
                cats = sorted(raw.dropna().astype(str).unique())
                index = {c: k for k, c in enumerate(cats)}
                x = raw.astype(str).map(lambda v: index.get(v, np.nan)).to_numpy(float)
                ax.set_xticks(range(len(cats)), cats, fontsize=4.8)
            ax.scatter(x, y, s=7, alpha=.7, linewidth=.3, facecolor="none",
                       edgecolor="#333333", marker=ont_marker[ont],
                       label=ONTOLOGY_SHORT[ont] if i == 0 else None)
        if p in LOG_PARAMS:
            ax.set_xscale("log")
        ax.set_xlabel(PARAM_LABEL.get(p, p).replace("\n", " "), fontsize=6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(labelsize=5.2)
        if i % ncols == 0:
            ax.set_ylabel(VALIDATION_METRIC_LABEL.get(metric, metric), fontsize=6)
    for j in range(len(params), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.legend(loc="upper left", bbox_to_anchor=(1.0, .95), frameon=False, fontsize=6.2,
               title="Ontology", title_fontsize=6.2)
    fig.text(.5, -.04, f"{SEARCH_CAPTION} Marginal effect of each tuned hyperparameter on "
             f"validation {VALIDATION_METRIC_LABEL.get(metric, metric)}; one point per trial. "
             f"Shared y-limits across all panels. "
             + provenance("src/plot_arc_tuning.py", "hybrid_search/trial_*/*/config.json"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "validation_hyperparameter_small_multiples.png", tier)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--err", choices=["sd", "sem", "ci95"], default="sd")
    ap.add_argument("--tier", choices=["main", "supplementary"], default="supplementary")
    args = ap.parse_args()

    apply_style()
    print("Palette fingerprint:", assert_palette_locked())
    report_colorblind_audit()
    root = args.tuning_root.resolve()
    out = (args.output_dir or Path("plots") / root.name).resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = load_trials(root)
    if df.empty:
        raise SystemExit(f"No trial metrics found under {root / 'hybrid_search'}")
    df.to_csv(out / "trial_metrics.csv", index=False)
    metric = ("validation_macro_fmax" if df["validation_macro_fmax"].notna().any()
              else "validation_micro_fmax")
    print(f"Selection metric for landscape/top-trial figures: {metric}")

    plot_top_trials(df, out, metric, args.tier)
    plot_parallel_coordinates(df, out, metric, args.tier)
    plot_param_small_multiples(df, out, metric, args.tier)
    plot_seed_metrics(root, out, args.err, args.tier)
    plot_histories(root, out, args.tier)
    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()
