#!/usr/bin/env python3
"""Publication figures for the ARC baseline benchmark.

Reads results/benchmark_metrics.csv (+ bootstrap CIs and the coverage audit)
from an arc_benchmark workspace and renders BMC/Nature-ready comparisons of
DeepGreenGO against the sequence-, structure- and frequency-based baselines.

Integrity gate: a method whose stored predictions are empty is NOT plotted as
a legitimate zero. BLAST currently trips this — its raw hit file contains
7,108 alignments that re-transfer to 2,619 nonzero scores, yet the stored
prediction arrays are all zero and its Fmax is recorded as 0.000. Plotting
that as a real score would state that BLAST cannot annotate these proteins,
which the raw hits directly contradict. Such methods are drawn in a hatched
"not evaluated" slot with the reason printed and captioned.

  python src/plot_benchmark.py
  python src/plot_benchmark.py --workspace arc_benchmark/nominal_30_identity_80_coverage
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
    CATEGORICAL_PALETTE,
    DOUBLE_COLUMN_IN,
    MAIN,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    SUPPLEMENTARY,
    annotate_insufficient_data,
    apply_style,
    assert_palette_locked,
    label_panel,
    provenance,
    report_colorblind_audit,
    savefig,
)

# Method display order: our model first, then frequency, sequence, structure.
METHOD_ORDER = ["deepgreengo", "naive", "blast", "blast_max",
                "diamond", "diamond_max", "foldseek", "foldseek_max"]
METHOD_LABEL = {
    "deepgreengo": "DeepGreenGO", "naive": "Naive frequency",
    "blast": "BLAST (top-10)", "blast_max": "BLAST (max ident.)",
    "diamond": "DIAMOND (top-10)", "diamond_max": "DIAMOND (max ident.)",
    "foldseek": "Foldseek (top-10)", "foldseek_max": "Foldseek (max ident.)",
}
# Family colour: one hue per evidence type, so the comparison reads as
# "our model vs frequency vs sequence homology vs structure homology"
# rather than eight unrelated categories.
METHOD_FAMILY = {
    "deepgreengo": "model", "naive": "frequency",
    "blast": "sequence", "blast_max": "sequence",
    "diamond": "sequence", "diamond_max": "sequence",
    "foldseek": "structure", "foldseek_max": "structure",
}
FAMILY_COLOR = {
    "model": CATEGORICAL_PALETTE[3],      # green
    "frequency": CATEGORICAL_PALETTE[2],  # yellow
    "sequence": CATEGORICAL_PALETTE[0],   # blue
    "structure": CATEGORICAL_PALETTE[4],  # violet
}
FAMILY_LABEL = {"model": "This work", "frequency": "Frequency prior",
                "sequence": "Sequence homology", "structure": "Structure homology"}

METRICS = {
    "cafa_fmax": ("CAFA F$_{max}$", True),
    "cafa_smin": ("CAFA S$_{min}$", False),
    "micro_aupr": ("Micro-AUPR", True),
    "protein_coverage_any_score": ("Protein coverage", True),
}


def load(workspace: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    metrics = pd.read_csv(workspace / "results/benchmark_metrics.csv")
    audit_path = workspace / "results/coverage_and_mapping_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else None
    return metrics, audit


def detect_invalid(metrics: pd.DataFrame, workspace: Path) -> dict[str, str]:
    """Flag methods whose stored predictions are empty for every ontology.

    A zero-coverage method is only a legitimate result if it genuinely found
    nothing. Cross-check against the raw hit file: if raw hits exist, the
    zero is a broken artifact and the method must not be scored.
    """
    invalid: dict[str, str] = {}
    for method, group in metrics.groupby("method"):
        coverage = group["protein_coverage_any_score"].fillna(0)
        if coverage.max() > 0:
            continue
        raw = workspace / f"raw/{method.replace('_max', '')}_hits.tsv"
        if raw.exists() and raw.stat().st_size > 0:
            lines = sum(1 for _ in raw.open())
            invalid[method] = (f"stored predictions are empty for all ontologies, but "
                               f"{raw.name} contains {lines:,} alignments — stale artifact, "
                               f"re-run required")
        else:
            invalid[method] = "no predictions and no raw hits found"
    return invalid


def _methods_present(metrics: pd.DataFrame) -> list[str]:
    present = set(metrics["method"].unique())
    ordered = [m for m in METHOD_ORDER if m in present]
    return ordered + sorted(present - set(ordered))


def plot_metric_panels(metrics: pd.DataFrame, invalid: dict[str, str], out: Path,
                       tier: str = MAIN) -> None:
    """CAFA Fmax with bootstrap CIs, one panel per ontology."""
    methods = _methods_present(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.0), sharey=True)
    y = np.arange(len(methods))
    for panel, (ax, ontology) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = metrics[metrics.ontology == ontology].set_index("method")
        values, lows, highs, colors, hatches = [], [], [], [], []
        for method in methods:
            if method in invalid or method not in sub.index:
                values.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                colors.append("#dddddd"); hatches.append("///")
                continue
            row = sub.loc[method]
            value = float(row["cafa_fmax"])
            values.append(value)
            lows.append(value - float(row.get("cafa_fmax_ci_low", value)))
            highs.append(float(row.get("cafa_fmax_ci_high", value)) - value)
            colors.append(FAMILY_COLOR[METHOD_FAMILY.get(method, "sequence")])
            hatches.append("")
        ax.barh(y, [0 if not np.isfinite(v) else v for v in values],
                xerr=[np.nan_to_num(lows), np.nan_to_num(highs)],
                color=colors, hatch=hatches, edgecolor="#111111", linewidth=.45, height=.7,
                error_kw=dict(elinewidth=.8, capsize=2, ecolor="#111111"))
        for i, (method, value) in enumerate(zip(methods, values)):
            if method in invalid:
                ax.text(.02, i, "not evaluated", va="center", ha="left", fontsize=5,
                        style="italic", color="#777777")
            elif np.isfinite(value):
                # Offset past the CI cap, not the bar end, or the label sits on it.
                tip = value + (highs[i] if np.isfinite(highs[i]) else 0.0)
                ax.text(tip + .014, i, f"{value:.3f}", va="center", ha="left", fontsize=5.2)
        ax.set_yticks(y, [METHOD_LABEL.get(m, m) for m in methods], fontsize=6)
        ax.invert_yaxis()
        ax.set_title(ONTOLOGY_SHORT[ontology])
        ax.set_xlabel("CAFA F$_{max}$", fontsize=7)
        ax.set_xlim(0, max(.05, float(np.nanmax(values)) * 1.30))
        label_panel(ax, chr(97 + panel))
    handles = [Patch(facecolor=FAMILY_COLOR[f], edgecolor="#111111", label=FAMILY_LABEL[f])
               for f in ("model", "frequency", "sequence", "structure")]
    handles.append(Patch(facecolor="#dddddd", edgecolor="#111111", hatch="///",
                         label="Not evaluated"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, .95),
               frameon=False, fontsize=6.2, title="Evidence type", title_fontsize=6.5)
    note = ("Error bars are 1,000-replicate paired protein bootstrap 95% CIs. "
            "Bars are coloured by evidence type, not by individual method. ")
    if invalid:
        note += ("Hatched slots are methods excluded from scoring: "
                 + "; ".join(f"{METHOD_LABEL.get(m, m)} ({r})" for m, r in invalid.items()) + ". ")
    fig.text(.5, -.10, note + provenance("src/plot_benchmark.py", "results/benchmark_metrics.csv"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "benchmark_cafa_fmax.png", tier)


def plot_metric_grid(metrics: pd.DataFrame, invalid: dict[str, str], out: Path,
                     tier: str = SUPPLEMENTARY) -> None:
    """Every headline metric: rows are metrics, columns are ontologies."""
    methods = [m for m in _methods_present(metrics) if m not in invalid]
    keys = [k for k in METRICS if k in metrics.columns]
    fig, axes = plt.subplots(len(keys), 3, figsize=(DOUBLE_COLUMN_IN, 1.85 * len(keys)),
                             squeeze=False, sharex="row")
    x = np.arange(len(methods))
    panel = 0
    for row, key in enumerate(keys):
        label, higher = METRICS[key]
        values_all = pd.to_numeric(metrics[key], errors="coerce")
        top = float(values_all.max()) * 1.12 if np.isfinite(values_all.max()) else 1.0
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row][col]
            sub = metrics[metrics.ontology == ontology].set_index("method")
            values = [float(sub.loc[m, key]) if m in sub.index else np.nan for m in methods]
            ax.bar(x, values, color=[FAMILY_COLOR[METHOD_FAMILY.get(m, "sequence")] for m in methods],
                   edgecolor="#111111", linewidth=.4, width=.72)
            ax.set_ylim(0, top)
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            if col == 0:
                ax.set_ylabel(label, fontsize=6.5)
            if row == len(keys) - 1:
                ax.set_xticks(x, [METHOD_LABEL.get(m, m) for m in methods],
                              rotation=35, ha="right", fontsize=5.2)
            else:
                ax.set_xticks(x, [""] * len(methods))
            if not higher:
                ax.text(.02, .93, "lower is better", transform=ax.transAxes,
                        fontsize=5, color="#777777")
            label_panel(ax, chr(97 + panel))
            panel += 1
    handles = [Patch(facecolor=FAMILY_COLOR[f], edgecolor="#111111", label=FAMILY_LABEL[f])
               for f in ("model", "frequency", "sequence", "structure")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, .95),
               frameon=False, fontsize=6.2, title="Evidence type", title_fontsize=6.5)
    excluded = ", ".join(METHOD_LABEL.get(m, m) for m in invalid) or "none"
    fig.text(.5, -.04,
             f"Rows share a y-axis across ontologies so cross-ontology magnitudes are comparable. "
             f"Coverage is the fraction of the 754 test proteins receiving any non-zero score; a "
             f"method with high F$_{{max}}$ but low coverage annotates few proteins confidently "
             f"rather than annotating all of them well. Excluded methods: {excluded}. "
             + provenance("src/plot_benchmark.py", "results/benchmark_metrics.csv"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "benchmark_metric_grid.png", tier)


def plot_coverage_vs_fmax(metrics: pd.DataFrame, invalid: dict[str, str], out: Path,
                          tier: str = SUPPLEMENTARY) -> None:
    """Coverage against Fmax — separates 'accurate' from 'accurate and complete'."""
    methods = [m for m in _methods_present(metrics) if m not in invalid]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 2.5), sharex=True, sharey=True)
    for panel, (ax, ontology) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = metrics[metrics.ontology == ontology].set_index("method")
        for method in methods:
            if method not in sub.index:
                continue
            row = sub.loc[method]
            ax.scatter(float(row["protein_coverage_any_score"]), float(row["cafa_fmax"]),
                       s=26, color=FAMILY_COLOR[METHOD_FAMILY.get(method, "sequence")],
                       edgecolor="#111111", linewidth=.4, zorder=3)
            ax.annotate(METHOD_LABEL.get(method, method).split(" (")[0],
                        (float(row["protein_coverage_any_score"]), float(row["cafa_fmax"])),
                        xytext=(3, 3), textcoords="offset points", fontsize=4.6, color="#333333")
        ax.set_title(ONTOLOGY_SHORT[ontology])
        ax.set_xlabel("Protein coverage", fontsize=7)
        ax.set_xlim(-.05, 1.08)
        label_panel(ax, chr(97 + panel))
    axes[0].set_ylabel("CAFA F$_{max}$", fontsize=7)
    fig.text(.5, -.10,
             "Upper-right is better: high F$_{max}$ at high coverage. Homology-transfer baselines "
             "sit at low coverage because they only score proteins with a qualifying hit, so their "
             "F$_{max}$ describes a favourable subset rather than the full test set. "
             + provenance("src/plot_benchmark.py", "results/benchmark_metrics.csv"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "benchmark_coverage_vs_fmax.png", tier)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path,
                    default=Path("arc_benchmark/nominal_30_identity_80_coverage"))
    ap.add_argument("--output-dir", type=Path, default=Path("plots/figures/benchmark"))
    ap.add_argument("--tier", choices=["main", "supplementary"], default="main")
    args = ap.parse_args()

    apply_style()
    print("Palette fingerprint:", assert_palette_locked())
    report_colorblind_audit()
    workspace = args.workspace.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    metrics, audit = load(workspace)
    invalid = detect_invalid(metrics, workspace)
    if invalid:
        print("\nINTEGRITY: methods excluded from scoring")
        for method, reason in invalid.items():
            print(f"  {method}: {reason}")
        (out / "excluded_methods.json").write_text(json.dumps(invalid, indent=2) + "\n")
    else:
        print("All methods have non-empty predictions.")

    valid = metrics[~metrics.method.isin(invalid)]
    print("\nCAFA Fmax by method (valid methods only):")
    print(valid.pivot_table(index="method", columns="ontology_short",
                            values="cafa_fmax").round(4).to_string())

    plot_metric_panels(metrics, invalid, out, MAIN if args.tier == "main" else SUPPLEMENTARY)
    plot_metric_grid(metrics, invalid, out)
    plot_coverage_vs_fmax(metrics, invalid, out)
    metrics.to_csv(out / "benchmark_metrics_plotted.csv", index=False)
    print(f"\nWrote benchmark figures to {out}")


if __name__ == "__main__":
    main()
