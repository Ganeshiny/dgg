#!/usr/bin/env python3
"""Publication figures for the ARC baseline benchmark.

Reads results/benchmark_metrics.csv (+ bootstrap CIs and the coverage audit)
from an arc_benchmark workspace and renders BMC/Nature-ready baseline-only
comparisons.  The plotted method set is deliberately locked to CAFA naive,
BLAST, DIAMOND, Foldseek, DeepFRI, DPFunc, and HEAL; other evaluated methods
remain in the source results but cannot leak into these figures.

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

# This is both the display order and the baseline-only allowlist. Keep the two
# transfer summaries for each search method and both DeepFRI modes.
METHOD_ORDER = ["naive", "blast", "blast_max", "diamond", "diamond_max",
                "foldseek", "foldseek_max", "deepfri_sequence",
                "deepfri_structure", "dpfunc", "heal"]
REQUESTED_METHODS = frozenset(METHOD_ORDER)
METHOD_LABEL = {
    "naive": "CAFA naive",
    "blast": "BLAST (top-10)", "blast_max": "BLAST (max ident.)",
    "diamond": "DIAMOND (top-10)", "diamond_max": "DIAMOND (max ident.)",
    "foldseek": "Foldseek (top-10)", "foldseek_max": "Foldseek (max ident.)",
    "deepfri_sequence": "DeepFRI (sequence)",
    "deepfri_structure": "DeepFRI (structure)",
    "dpfunc": "DPFunc", "heal": "HEAL",
}
# Family colour: one hue per evidence type, so the comparison reads as
# frequency vs homology transfer vs external deep-learning baselines.
METHOD_FAMILY = {
    "naive": "frequency",
    "blast": "sequence", "blast_max": "sequence",
    "diamond": "sequence", "diamond_max": "sequence",
    "foldseek": "structure", "foldseek_max": "structure",
    "deepfri_sequence": "deep_learning", "deepfri_structure": "deep_learning",
    "dpfunc": "deep_learning", "heal": "deep_learning",
}
FAMILY_COLOR = {
    "frequency": CATEGORICAL_PALETTE[2],  # yellow
    "sequence": CATEGORICAL_PALETTE[0],   # blue
    "structure": CATEGORICAL_PALETTE[4],  # violet
    "deep_learning": CATEGORICAL_PALETTE[3],  # green
}
FAMILY_LABEL = {"frequency": "CAFA frequency prior",
                "sequence": "Sequence homology", "structure": "Structure homology",
                "deep_learning": "External deep learning"}
FAMILY_ORDER = ("frequency", "sequence", "structure", "deep_learning")

METRICS = {
    "cafa_fmax": ("CAFA F$_{max}$", True),
    "cafa_smin": ("CAFA S$_{min}$", False),
    "micro_aupr": ("Micro-AUPR", True),
    "protein_coverage_any_score": ("Protein coverage", True),
}


def load(workspace: Path):
    metrics = pd.read_csv(workspace / "results/benchmark_metrics.csv")
    audit_path = workspace / "results/coverage_and_mapping_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else None
    boot_path = workspace / "results/bootstrap_metrics.csv"
    bootstrap = pd.read_csv(boot_path) if boot_path.exists() else None
    return metrics, audit, bootstrap


def select_requested_methods(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return only the locked baseline set, preserving the input row order."""
    if frame is None:
        return None
    if "method" not in frame.columns:
        raise ValueError("Benchmark table has no 'method' column")
    return frame.loc[frame["method"].isin(REQUESTED_METHODS)].copy()


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
    return [m for m in METHOD_ORDER if m in present]


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
               for f in FAMILY_ORDER]
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


def plot_bootstrap_boxplots(bootstrap: pd.DataFrame, invalid: dict[str, str], out: Path,
                            tier: str = MAIN) -> None:
    """Boxplots over the 1,000 paired protein bootstrap replicates.

    A boxplot needs a real distribution behind it. These two metrics have one
    — 1,000 resamples each — so the box shows the actual sampling
    uncertainty rather than a summary of a summary. The remaining metrics in
    the results table are single point estimates with no replicates, and are
    deliberately NOT drawn as boxes anywhere in this figure set.
    """
    methods = [m for m in _methods_present(bootstrap) if m not in invalid]
    keys = [k for k in ("cafa_fmax", "cafa_smin") if k in bootstrap.columns]
    fig, axes = plt.subplots(len(keys), 3, figsize=(DOUBLE_COLUMN_IN, 2.35 * len(keys)),
                             squeeze=False)
    positions = np.arange(len(methods))
    panel = 0
    for row, key in enumerate(keys):
        label, higher = METRICS.get(key, (key, True))
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row][col]
            sub = bootstrap[bootstrap.ontology == ontology]
            data, colors = [], []
            for method in methods:
                values = pd.to_numeric(sub[sub.method == method][key], errors="coerce")
                values = values[np.isfinite(values)].to_numpy()
                data.append(values if values.size else np.array([np.nan]))
                colors.append(FAMILY_COLOR[METHOD_FAMILY.get(method, "sequence")])
            box = ax.boxplot(data, positions=positions, widths=.62, patch_artist=True,
                             showfliers=False, whis=(2.5, 97.5),
                             medianprops=dict(color="#111111", linewidth=1.0),
                             boxprops=dict(linewidth=.5, edgecolor="#111111"),
                             whiskerprops=dict(linewidth=.6, color="#111111"),
                             capprops=dict(linewidth=.6, color="#111111"))
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(.85)
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
            if row == len(keys) - 1:
                ax.set_xticks(positions, [METHOD_LABEL.get(m, m) for m in methods],
                              rotation=35, ha="right", fontsize=5.2)
            else:
                ax.set_xticks(positions, [""] * len(methods))
            if not higher:
                # Above the axes: inside the panel this sat on the top whisker.
                ax.text(.0, 1.015, "lower is better", transform=ax.transAxes,
                        fontsize=5.2, color="#777777", va="bottom", ha="left")
            label_panel(ax, chr(97 + panel))
            panel += 1
    handles = [Patch(facecolor=FAMILY_COLOR[f], edgecolor="#111111", label=FAMILY_LABEL[f])
               for f in FAMILY_ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, .95),
               frameon=False, fontsize=6.2, title="Evidence type", title_fontsize=6.5)
    fig.text(.5, -.05,
             "Boxes span the interquartile range of 1,000 paired protein bootstrap replicates, "
             "the line is the median, and whiskers reach the 2.5th and 97.5th percentiles (a 95% "
             "interval); outliers are not drawn separately because every replicate is already "
             "represented. Each ontology is on its own y-scale. "
             + provenance("src/plot_benchmark.py", "results/bootstrap_metrics.csv"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "benchmark_bootstrap_boxplots.png", tier)


def plot_metric_grid(metrics: pd.DataFrame, invalid: dict[str, str], out: Path,
                     bootstrap: pd.DataFrame | None = None,
                     tier: str = SUPPLEMENTARY) -> None:
    """Every headline metric: rows are metrics, columns are ontologies.

    Error bars are drawn only where replicates actually exist. cafa_fmax and
    cafa_smin are bootstrapped, so they get 95% intervals; micro-AUPR and
    coverage are single point estimates in the results table and are left
    bare, with the absence stated in the caption rather than implied by a
    missing whisker.
    """
    methods = [m for m in _methods_present(metrics) if m not in invalid]
    keys = [k for k in METRICS if k in metrics.columns]
    has_ci = set()
    if bootstrap is not None:
        has_ci = {k for k in keys if k in bootstrap.columns}
    fig, axes = plt.subplots(len(keys), 3, figsize=(DOUBLE_COLUMN_IN, 1.85 * len(keys)),
                             squeeze=False, sharex="row")
    x = np.arange(len(methods))
    panel = 0
    for row, key in enumerate(keys):
        label, higher = METRICS[key]
        values_all = pd.to_numeric(metrics[key], errors="coerce")
        top = float(values_all.max()) * 1.18 if np.isfinite(values_all.max()) else 1.0
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row][col]
            sub = metrics[metrics.ontology == ontology].set_index("method")
            values = [float(sub.loc[m, key]) if m in sub.index else np.nan for m in methods]
            yerr = None
            if key in has_ci:
                lows, highs = [], []
                bsub = bootstrap[bootstrap.ontology == ontology]
                for method, value in zip(methods, values):
                    reps = pd.to_numeric(bsub[bsub.method == method][key], errors="coerce")
                    reps = reps[np.isfinite(reps)].to_numpy()
                    if reps.size and np.isfinite(value):
                        lows.append(max(value - np.percentile(reps, 2.5), 0))
                        highs.append(max(np.percentile(reps, 97.5) - value, 0))
                    else:
                        lows.append(0.0); highs.append(0.0)
                yerr = [lows, highs]
            ax.bar(x, values, color=[FAMILY_COLOR[METHOD_FAMILY.get(m, "sequence")] for m in methods],
                   edgecolor="#111111", linewidth=.4, width=.72, yerr=yerr,
                   error_kw=dict(elinewidth=.7, capsize=1.8, ecolor="#111111"))
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
               for f in FAMILY_ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, .95),
               frameon=False, fontsize=6.2, title="Evidence type", title_fontsize=6.5)
    excluded = ", ".join(METHOD_LABEL.get(m, m) for m in invalid) or "none"
    ci_note = ("CAFA F$_{max}$ and S$_{min}$ carry 95% bootstrap intervals (1,000 paired protein "
               "resamples). Micro-AUPR and coverage are single point estimates in the results "
               "table with no replicates, so they are shown without error bars rather than with "
               "invented ones. ")
    fig.text(.5, -.05, ci_note +
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
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path,
                    default=repo / "arc_benchmark/nominal_30_identity_80_coverage")
    ap.add_argument("--output-dir", type=Path, default=repo / "plots/figures/benchmark")
    ap.add_argument("--tier", choices=["main", "supplementary"], default="main")
    ap.add_argument(
        "--allow-missing", action="store_true",
        help="Render requested baselines that are present instead of failing if any are missing.",
    )
    args = ap.parse_args()

    apply_style()
    print("Palette fingerprint:", assert_palette_locked())
    report_colorblind_audit()
    workspace = args.workspace.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    metrics, audit, bootstrap = load(workspace)
    available = set(metrics["method"].astype(str))
    missing = [method for method in METHOD_ORDER if method not in available]
    if missing and not args.allow_missing:
        raise SystemExit(
            "Requested baseline results are missing from results/benchmark_metrics.csv: "
            + ", ".join(missing)
            + ". Complete/evaluate those baselines on ARC, or use --allow-missing "
              "for a partial diagnostic render."
        )
    if missing:
        print("WARNING: partial baseline render; missing: " + ", ".join(missing))
    removed = sorted(available - REQUESTED_METHODS)
    if removed:
        print("Baseline-only filter removed: " + ", ".join(removed))
    metrics = select_requested_methods(metrics)
    if audit is not None and "method" in audit.columns:
        audit = select_requested_methods(audit)
    bootstrap = select_requested_methods(bootstrap)
    if metrics.empty:
        raise SystemExit("None of the requested baseline methods are present in benchmark_metrics.csv")
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
    if bootstrap is not None:
        plot_bootstrap_boxplots(bootstrap, invalid, out, MAIN)
    else:
        print("NOTE: results/bootstrap_metrics.csv absent; skipping boxplot figure.")
    plot_metric_grid(metrics, invalid, out, bootstrap)
    plot_coverage_vs_fmax(metrics, invalid, out)
    metrics.to_csv(out / "benchmark_metrics_plotted.csv", index=False)
    print(f"\nWrote benchmark figures to {out}")


if __name__ == "__main__":
    main()
