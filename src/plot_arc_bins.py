#!/usr/bin/env python3
"""Plot test performance across sequence-homology and information-content (IC)
bins, per ontology, per input-modality variant, one line per model.

Previously this put all 5 models x 3 input-modality variants (15 series) on
one panel and let pandas groupby sort the bin labels lexicographically, which
silently scrambled the x-axis (e.g. "40-60%" sorted before "<30%"). This
version orders bins by increasing homology/information-content explicitly
(plot_style.BIN_ORDER) and facets by variant as well as ontology, so each
panel holds at most 5 lines (one per model).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_style import (
    BIN_AXIS_LABEL,
    BIN_ORDER,
    DOUBLE_COLUMN_IN,
    METRIC_HIGHER_IS_BETTER,
    METRIC_LABEL,
    METRIC_ORDER,
    MODEL_COLOR,
    MODEL_ORDER,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    VARIANT_LABEL,
    VARIANT_ORDER,
    annotate_insufficient_data,
    apply_style,
    label_panel,
    savefig,
)


def _marker_sizes(n: pd.Series, size_by_n: bool) -> np.ndarray:
    if not size_by_n:
        return np.full(len(n), 14.0)
    values = n.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    denom = finite.max() if finite.size and finite.max() > 0 else 1.0
    return 10.0 + 22.0 * np.sqrt(np.nan_to_num(values) / denom)


def plot_bin_grid(df: pd.DataFrame, out: Path, bin_type: str, order: list[str], metric: str, size_by_n: bool) -> None:
    higher_better = METRIC_HIGHER_IS_BETTER[metric]
    fig, axes = plt.subplots(len(VARIANT_ORDER), len(ONTOLOGY_ORDER),
                              figsize=(DOUBLE_COLUMN_IN, 2.5 * len(VARIANT_ORDER)),
                              sharex=True, sharey="col")
    x = np.arange(len(order))
    any_data = False
    panel = 0
    for row, variant in enumerate(VARIANT_ORDER):
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row, col]
            sub = df[(df.ontology == ontology) & (df.input_modality == variant)]
            if sub.empty:
                annotate_insufficient_data(ax)
            else:
                for model in MODEL_ORDER:
                    g = sub[sub.model == model]
                    if g.empty:
                        continue
                    any_data = True
                    agg = g.groupby("bin", observed=False)[metric].mean().reindex(order)
                    n = g.groupby("bin", observed=False)["examples"].first().reindex(order)
                    ax.plot(x, agg.to_numpy(), color=MODEL_COLOR[model], linewidth=1.2, alpha=0.9, zorder=2)
                    ax.scatter(x, agg.to_numpy(), s=_marker_sizes(n, size_by_n), color=MODEL_COLOR[model], zorder=3, linewidth=0)
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            if col == 0:
                ax.set_ylabel(f"{VARIANT_LABEL[variant]}\n{METRIC_LABEL[metric]}", fontsize=7)
            if row == len(VARIANT_ORDER) - 1:
                ax.set_xticks(x, order, rotation=35, ha="right")
            label_panel(ax, chr(97 + panel))
            ax.set_xlim(-0.5, len(order) - 0.5)
            panel += 1
    if not any_data:
        plt.close(fig)
        print(f"Skipping {bin_type}_{metric.lower()}: no data for any (ontology, variant, model)")
        return
    fig.supxlabel(BIN_AXIS_LABEL[bin_type], fontsize=8)
    caption = "marker area ~ sqrt(test examples in bin)" if size_by_n else None
    if not higher_better:
        caption = f"{caption}; lower is better" if caption else "lower is better"
    if caption:
        fig.text(0.995, 0.002, caption, ha="right", va="bottom", fontsize=6, color="#898781")
    handles = [Line2D([0], [0], color=MODEL_COLOR[m], marker="o", linestyle="-", markersize=4, label=m) for m in MODEL_ORDER]
    fig.legend(handles=handles, title="Model", loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    savefig(fig, out / f"{bin_type}_{metric.lower()}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("plots/arc_tuning_cafa/bin_evaluation"))
    ap.add_argument("--metrics", nargs="+", default=METRIC_ORDER, choices=METRIC_ORDER)
    ap.add_argument("--no-size-by-n", dest="size_by_n", action="store_false", default=True,
                     help="Disable scaling marker area by sqrt(test examples in the bin).")
    args = ap.parse_args()
    apply_style()
    df = pd.read_csv(args.bin_csv)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "bin_metrics.csv", index=False)

    found_bin_types = set(df["bin_type"].dropna().unique())
    for bin_type in [b for b in BIN_ORDER if b in found_bin_types]:
        order = BIN_ORDER[bin_type]
        sub_bt = df[df.bin_type == bin_type].copy()
        sub_bt["bin"] = pd.Categorical(sub_bt["bin"], categories=order, ordered=True)
        missing = sorted(set(order) - set(sub_bt["bin"].dropna().unique().astype(str)))
        if missing:
            print(f"{bin_type}: bin(s) with no rows in the data, will show as gaps: {missing}")
        for metric in args.metrics:
            if metric not in sub_bt:
                continue
            plot_bin_grid(sub_bt, out, bin_type, order, metric, args.size_by_n)
    unknown = found_bin_types - set(BIN_ORDER)
    if unknown:
        print(f"Skipping unrecognised bin_type(s) with no defined ordering: {sorted(unknown)}")
    print(f"Wrote bin plots to {out}")


if __name__ == "__main__":
    main()
