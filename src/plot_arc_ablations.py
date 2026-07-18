#!/usr/bin/env python3
"""Publication figures for the ARC input-modality ablation (full / seq_only /
struct_only, x5 models, x3 ontologies, x5 seeds).

For each metric this writes two figures showing the same aggregated data:

  dynamite_<metric>.png   bar + error bar ("dynamite" plot), as requested.
  strip_<metric>.png      individual seed points + mean +/- error overlay.

The strip plot is the recommended default for n=5 seeds: a bar can look
identical whether it summarises 5 tightly clustered runs or 5 wildly variable
ones, and it hides the actual sample size. Krzywinski & Altman's Nature
Methods "Error bars" column (and the wider "beyond bar charts" literature)
argue against bar+error-bar for exactly this reason. Both are generated so
you can choose per panel; see --style to restrict to one.

Usage:
  python src/plot_arc_ablations.py
  python src/plot_arc_ablations.py --metrics Micro_Fmax Smin --style strip
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
    BIN_AXIS_LABEL,
    DOUBLE_COLUMN_IN,
    ERROR_KIND_LABEL,
    METRIC_HIGHER_IS_BETTER,
    METRIC_LABEL,
    METRIC_ORDER,
    MODEL_COLOR,
    MODEL_ORDER,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    VARIANT_HATCH,
    VARIANT_LABEL,
    VARIANT_ORDER,
    annotate_insufficient_data,
    apply_style,
    jitter,
    label_panel,
    mean_and_error,
    savefig,
)

EXPECTED_SEEDS = 5
VARIANT_MARKER = {"full": "o", "seq_only": "^", "struct_only": "s"}


def read_results(root: Path, logs: Path) -> pd.DataFrame:
    """Load per-seed test metrics from materialised result folders, falling
    back to SLURM stdout logs for (ontology, model, variant, seed) combos
    whose result folder was never downloaded from the cluster. Both sources
    are needed in practice: as of this rewrite, only molecular_function has
    materialised result folders locally, but all three ontologies are fully
    present across logs/arc_ablation_*.out.

    The SLURM log's trailing filename number is the array-task index (0-224),
    not the real seed - the JSON payload a completed task prints has no seed
    field at all. Task index and seed are in a fixed 1:1 correspondence per
    (ontology, model, input) triple, but that mapping isn't recoverable from
    the log alone, so a log-derived row can't be matched against a specific
    folder-derived seed. Verified concretely: array task 45's logged "test"
    metrics for molecular_function/Hybrid/full are byte-identical to
    seed_1103's test_metrics.json - same run, two labels. Deduplicating by
    (ontology, model, input, seed) therefore silently double-counted every
    triple that has folder data (each folder seed re-appeared under its
    array-index alias and was treated as a 6th, 7th, ... replicate) while
    leaving genuinely folder-less triples untouched - inflating n from 225
    to 300 in practice. Deduplicating at the (ontology, model, input) level
    instead is the honest fix: trust the folder if any exists for a triple,
    otherwise take all 5 of that triple's log entries.
    """
    rows = []
    for p in root.rglob("test_metrics.json"):
        rel = p.relative_to(root).parts
        if len(rel) >= 5:
            rows.append({"ontology": rel[0], "model": rel[1], "input": rel[2], "seed": rel[3], **json.loads(p.read_text())})
    covered = {(r["ontology"], r["model"], r["input"]) for r in rows}
    for p in sorted(logs.glob("arc_ablation_*.out")):
        for line in reversed(p.read_text(errors="ignore").splitlines()):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not {"input_modality", "model", "ontology", "test"}.issubset(item):
                continue
            key = (item["ontology"], item["model"], item["input_modality"])
            if key not in covered:
                pseudo_seed = f"log_{p.stem.rsplit('_', 1)[-1]}"
                rows.append({"ontology": item["ontology"], "model": item["model"], "input": item["input_modality"], "seed": pseudo_seed, **item["test"]})
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["seed"] = df["seed"].astype(str)
    return df


def coverage_table(df: pd.DataFrame, expected_seeds: int = EXPECTED_SEEDS) -> pd.DataFrame:
    rows = []
    for ontology in ONTOLOGY_ORDER:
        for model in MODEL_ORDER:
            for variant in VARIANT_ORDER:
                found = df[(df.ontology == ontology) & (df.model == model) & (df.input == variant)]["seed"].nunique()
                rows.append({
                    "ontology": ontology, "model": model, "input_modality": variant,
                    "seeds_found": found, "seeds_expected": expected_seeds,
                    "complete": found >= expected_seeds,
                })
    return pd.DataFrame(rows)


def report_coverage(coverage: pd.DataFrame) -> None:
    missing = coverage[~coverage["complete"]]
    if missing.empty:
        print(f"Coverage: all {len(coverage)} (ontology, model, input) cells have >= {EXPECTED_SEEDS} seeds.")
        return
    print(f"Coverage: {len(missing)}/{len(coverage)} (ontology, model, input) cells have FEWER than "
          f"{missing['seeds_expected'].iloc[0]} seeds. Affected figures will mark these with 'n=' labels "
          f"or, if a whole ontology is empty, an 'insufficient data' panel:")
    for _, row in missing.iterrows():
        print(f"  {row.ontology:20s} {row.model:10s} {row.input_modality:12s} seeds_found={row.seeds_found}")


def _bar_positions(n_groups: int, n_series: int, width: float = 0.25, gap: float = 1.05):
    x = np.arange(n_groups)
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * width * gap
    return x, offsets


def _n_label(ax, xi: float, n: int) -> None:
    if 0 < n < EXPECTED_SEEDS:
        ax.text(xi, 0.015, f"n={n}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=5.5, color="#898781")


def _legend(fig, out_path_stem: str, variant_glyphs) -> None:
    model_handles = [Patch(facecolor=MODEL_COLOR[m], edgecolor="none", label=m) for m in MODEL_ORDER]
    leg1 = fig.legend(handles=model_handles, title="Model", loc="upper left",
                       bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=7, title_fontsize=7)
    fig.add_artist(leg1)
    fig.legend(handles=variant_glyphs, title="Input", loc="upper left",
               bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7, title_fontsize=7)


def _empty_panel(ax, panel: int, ontology: str) -> None:
    annotate_insufficient_data(ax)
    ax.set_title(ONTOLOGY_SHORT[ontology])
    label_panel(ax, chr(97 + panel))


def plot_dynamite(df: pd.DataFrame, out: Path, metric: str, err_kind: str = "sd") -> None:
    higher_better = METRIC_HIGHER_IS_BETTER[metric]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.2), sharey=False)
    x, offsets = _bar_positions(len(MODEL_ORDER), len(VARIANT_ORDER))
    any_data = False
    for panel, (ax, ontology) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ontology]
        if sub.empty:
            _empty_panel(ax, panel, ontology)
            continue
        any_data = True
        for j, variant in enumerate(VARIANT_ORDER):
            means, errs, ns = [], [], []
            for model in MODEL_ORDER:
                vals = sub[(sub.model == model) & (sub.input == variant)][metric].to_numpy()
                m, e, n = mean_and_error(vals, err_kind)
                means.append(m); errs.append(e); ns.append(n)
            xi = x + offsets[j]
            ax.bar(xi, means, width=0.25 * 0.95, yerr=errs, color=[MODEL_COLOR[m] for m in MODEL_ORDER],
                   hatch=VARIANT_HATCH[variant], edgecolor="#0b0b0b", linewidth=0.5,
                   error_kw=dict(elinewidth=0.8, capsize=2.2, ecolor="#0b0b0b"))
            for xii, n in zip(xi, ns):
                _n_label(ax, xii, n)
        ax.set_xticks(x, MODEL_ORDER, rotation=30, ha="right")
        ax.set_title(ONTOLOGY_SHORT[ontology])
        label_panel(ax, chr(97 + panel))
        if not higher_better:
            ax.text(0.03, 0.96, "lower is better", transform=ax.transAxes, fontsize=6, color="#898781", va="top")
    if not any_data:
        plt.close(fig)
        raise SystemExit(f"No ablation data found for metric {metric!r} in any ontology")
    axes[0].set_ylabel(f"{METRIC_LABEL[metric]}\n({ERROR_KIND_LABEL[err_kind]}, n≤5 seeds)")
    variant_handles = [Patch(facecolor="white", edgecolor="black", hatch=VARIANT_HATCH[v], label=VARIANT_LABEL[v]) for v in VARIANT_ORDER]
    _legend(fig, "dynamite", variant_handles)
    savefig(fig, out / f"dynamite_{metric.lower()}.png")


def plot_strip(df: pd.DataFrame, out: Path, metric: str, err_kind: str = "sd") -> None:
    higher_better = METRIC_HIGHER_IS_BETTER[metric]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.2), sharey=False)
    x, offsets = _bar_positions(len(MODEL_ORDER), len(VARIANT_ORDER))
    any_data = False
    for panel, (ax, ontology) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ontology]
        if sub.empty:
            _empty_panel(ax, panel, ontology)
            continue
        any_data = True
        for j, variant in enumerate(VARIANT_ORDER):
            for i, model in enumerate(MODEL_ORDER):
                vals = sub[(sub.model == model) & (sub.input == variant)][metric].to_numpy()
                vals = vals[np.isfinite(vals)]
                xi = x[i] + offsets[j]
                if vals.size:
                    seed_for_jitter = abs(hash((ontology, model, variant))) % (2 ** 32)
                    xs = xi + jitter(vals.size, width=0.045, seed=seed_for_jitter)
                    ax.scatter(xs, vals, s=11, color=MODEL_COLOR[model], marker=VARIANT_MARKER[variant],
                               alpha=0.65, linewidth=0, zorder=3)
                mean, err, n = mean_and_error(vals, err_kind)
                if n:
                    ax.errorbar([xi], [mean], yerr=[err], fmt="D", markersize=3.6, markerfacecolor="white",
                                markeredgecolor="#0b0b0b", ecolor="#0b0b0b", elinewidth=0.9, capsize=2.2, zorder=5)
                _n_label(ax, xi, n)
        ax.set_xticks(x, MODEL_ORDER, rotation=30, ha="right")
        ax.set_title(ONTOLOGY_SHORT[ontology])
        label_panel(ax, chr(97 + panel))
        if not higher_better:
            ax.text(0.03, 0.96, "lower is better", transform=ax.transAxes, fontsize=6, color="#898781", va="top")
    if not any_data:
        plt.close(fig)
        raise SystemExit(f"No ablation data found for metric {metric!r} in any ontology")
    axes[0].set_ylabel(f"{METRIC_LABEL[metric]}\n(points = seeds; ◇ = {ERROR_KIND_LABEL[err_kind]})")
    variant_handles = [Line2D([0], [0], marker=VARIANT_MARKER[v], linestyle="", color="#52514e",
                               label=VARIANT_LABEL[v], markersize=5) for v in VARIANT_ORDER]
    _legend(fig, "strip", variant_handles)
    savefig(fig, out / f"strip_{metric.lower()}.png")


def plot_heatmap(df: pd.DataFrame, out: Path, metric: str = "Micro_Fmax") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.4), sharey=True)
    vmax = 100.0 if metric == "Smin" else 1.0
    image = None
    for panel, (ax, ontology) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ontology]
        table = sub.pivot_table(index="model", columns="input", values=metric, aggfunc="mean").reindex(index=MODEL_ORDER, columns=VARIANT_ORDER)
        image = ax.imshow(table.values, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(VARIANT_ORDER)), [VARIANT_LABEL[v] for v in VARIANT_ORDER], rotation=20, ha="right")
        ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
        ax.set_title(ONTOLOGY_SHORT[ontology])
        label_panel(ax, chr(97 + panel))
        ax.grid(False)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                if pd.notna(table.iloc[i, j]):
                    value = table.iloc[i, j]
                    color = "white" if value < 0.6 * vmax else "black"
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=6.5)
    fig.colorbar(image, ax=axes, fraction=.025, pad=.02, label=f"Mean {METRIC_LABEL[metric]} across seeds")
    savefig(fig, out / f"{metric.lower()}_model_input_heatmap.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations-root", type=Path, default=Path("arc_tuning_cafa/ablations/nominal_30_identity_80_coverage"))
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--output-dir", type=Path, default=Path("plots/arc_tuning_cafa/ablations"))
    ap.add_argument("--metrics", nargs="+", default=METRIC_ORDER, choices=METRIC_ORDER)
    ap.add_argument("--style", choices=["dynamite", "strip", "both"], default="both")
    ap.add_argument("--err", choices=["sd", "sem", "ci95"], default="sd",
                     help="Error bar type across seeds (default: sd, matching this repo's mean+/-SD convention).")
    args = ap.parse_args()

    apply_style()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = read_results(args.ablations_root.resolve(), args.logs_dir.resolve())
    if df.empty:
        raise SystemExit("No ablation result JSON or logs found")
    df.to_csv(out / "ablation_test_metrics.csv", index=False)

    coverage = coverage_table(df)
    coverage.to_csv(out / "ablation_coverage.csv", index=False)
    report_coverage(coverage)

    for metric in args.metrics:
        if metric not in df:
            print(f"Skipping {metric}: not present in loaded data")
            continue
        if args.style in ("dynamite", "both"):
            plot_dynamite(df, out, metric, args.err)
        if args.style in ("strip", "both"):
            plot_strip(df, out, metric, args.err)
    plot_heatmap(df, out, "Micro_Fmax")

    print(f"Loaded {len(df)} completed ablation runs")
    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()
