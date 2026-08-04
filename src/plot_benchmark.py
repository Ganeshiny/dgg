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
from matplotlib.colors import to_rgb

from plot_style import (
    DOUBLE_COLUMN_IN,
    MAIN,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    SUPPLEMENTARY,
    annotate_insufficient_data,
    apply_style,
    label_panel,
    label_horizontal_bars,
    label_vertical_bars,
    provenance,
    savefig,
)

# This is both the display order and the locked comparison allowlist. Retain
# DeepGreenGO as the focal model, plus only the requested baseline methods.
METHOD_ORDER = ["deepgreengo", "naive", "blast", "blast_max", "diamond", "diamond_max",
                "foldseek", "foldseek_max", "deepfri_sequence",
                "deepfri_structure", "dpfunc", "heal"]
REQUESTED_METHODS = frozenset(METHOD_ORDER)
METHOD_LABEL = {
    "deepgreengo": "DeepGreenGO",
    "naive": "CAFA naive",
    "blast": "BLAST (top-10)", "blast_max": "BLAST (max ident.)",
    "diamond": "DIAMOND (top-10)", "diamond_max": "DIAMOND (max ident.)",
    "foldseek": "Foldseek (top-10)", "foldseek_max": "Foldseek (max ident.)",
    "deepfri_sequence": "DeepFRI (sequence)",
    "deepfri_structure": "DeepFRI (structure)",
    "dpfunc": "DPFunc", "heal": "HEAL",
}
# Per-method palette selected by farthest-point sampling in CIELAB space.
METHOD_COLOR = {
    "deepgreengo": "#006D2C", "naive": "#DAA520",
    "blast": "#0000CD", "blast_max": "#DC143C",
    "diamond": "#4682B4", "diamond_max": "#DA70D6",
    "foldseek": "#191970", "foldseek_max": "#32CD32",
    "deepfri_sequence": "#40E0D0", "deepfri_structure": "#8B4513",
    "dpfunc": "#DB7093", "heal": "#4169E1",
}

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values to CIELAB for a method-palette separation audit."""
    rgb = np.asarray(rgb, dtype=float)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    xyz = linear @ np.asarray([
        [.4124564, .2126729, .0193339],
        [.3575761, .7151522, .1191920],
        [.1804375, .0721750, .9503041],
    ])
    xyz = xyz / np.asarray([.95047, 1.0, 1.08883])
    epsilon, kappa = 216 / 24389, 24389 / 27
    transformed = np.where(xyz > epsilon, xyz ** (1 / 3), (kappa * xyz + 16) / 116)
    return np.stack([
        116 * transformed[..., 1] - 16,
        500 * (transformed[..., 0] - transformed[..., 1]),
        200 * (transformed[..., 1] - transformed[..., 2]),
    ], axis=-1)


def validate_method_palette(minimum_distance: float = 35.0) -> tuple[float, str, str]:
    points = _rgb_to_lab(np.asarray([to_rgb(METHOD_COLOR[m]) for m in METHOD_ORDER]))
    distances = [
        (float(np.linalg.norm(points[i] - points[j])), METHOD_ORDER[i], METHOD_ORDER[j])
        for i in range(len(points)) for j in range(i)
    ]
    minimum = min(distances)
    if minimum[0] < minimum_distance:
        raise ValueError(
            f"Method colors are too similar: {minimum[1]} vs {minimum[2]} "
            f"has CIELAB distance {minimum[0]:.1f} < {minimum_distance:.1f}"
        )
    return minimum

METRICS = {
    "cafa_fmax": ("F$_{max}$", True),
    "cafa_smin": ("S$_{min}$", False),
    "micro_aupr": ("Micro-AUPR", True),
    "macro_aupr": ("Macro-AUPR", True),
    "micro_auroc": ("Micro-AUROC", True),
    "macro_auroc": ("Macro-AUROC", True),
    "test_precision_at_validation_threshold": ("Precision", True),
    "test_recall_at_validation_threshold": ("Recall", True),
    "test_f1_at_validation_threshold": ("F1", True),
    "coverage_at_fmax": ("Coverage at F$_{max}$", True),
    "validation_threshold": ("Validation threshold", None),
    "validation_cafa_fmax": ("Validation F$_{max}$", True),
    "test_coverage_at_validation_threshold": ("Coverage at validation threshold", True),
    "mean_terms_per_protein_at_validation_threshold": ("Mean GO terms per protein", None),
    "protein_coverage_any_score": ("Protein coverage", True),
    "predicted_term_coverage": ("Predicted GO terms", None),
    "brier_score": ("Brier score", False),
    "expected_calibration_error": ("Expected calibration error", False),
}

ACCURACY_METRICS = tuple(list(METRICS)[:9])
OPERATING_METRICS = tuple(list(METRICS)[9:])
METRIC_STEMS = {
    "cafa_fmax": "fmax",
    "cafa_smin": "smin",
    "validation_cafa_fmax": "validation_fmax",
}


def load(workspace: Path):
    metrics = pd.read_csv(workspace / "results/benchmark_metrics.csv")
    audit_path = workspace / "results/coverage_and_mapping_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else None
    boot_path = workspace / "results/bootstrap_metrics.csv"
    bootstrap = pd.read_csv(boot_path) if boot_path.exists() else None
    return metrics, audit, bootstrap


def select_requested_methods(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return only DeepGreenGO and the locked baselines, preserving row order."""
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
    """Protein-centric Fmax with bootstrap CIs, one panel per ontology."""
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
            colors.append(METHOD_COLOR[method])
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
        ax.set_xlabel("F$_{max}$", fontsize=7)
        ax.set_xlim(0, max(.05, float(np.nanmax(values)) * 1.30))
        label_panel(ax, chr(97 + panel))
    handles = [Patch(facecolor=METHOD_COLOR[m], edgecolor="#111111", label=METHOD_LABEL[m])
               for m in methods]
    handles.append(Patch(facecolor="#dddddd", edgecolor="#111111", hatch="///",
                         label="Not evaluated"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, .95),
               frameon=False, fontsize=5.5, title="Method", title_fontsize=6)
    note = "Error bars are 1,000-replicate paired protein bootstrap 95% CIs. "
    if invalid:
        note += ("Hatched slots are methods excluded from scoring: "
                 + "; ".join(f"{METHOD_LABEL.get(m, m)} ({r})" for m, r in invalid.items()) + ". ")
    fig.text(.5, -.10, note + provenance("src/plot_benchmark.py", "results/benchmark_metrics.csv"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / "benchmark_fmax", tier)


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
    keys = [k for k in ("cafa_fmax", "cafa_smin", "micro_aupr", "macro_aupr") if k in bootstrap.columns]
    fig, axes = plt.subplots(len(keys), 3, figsize=(DOUBLE_COLUMN_IN, 2.35 * len(keys)),
                             squeeze=False)
    positions = np.arange(len(methods))
    panel = 0
    for row, key in enumerate(keys):
        label, _ = METRICS.get(key, (key, True))
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row][col]
            sub = bootstrap[bootstrap.ontology == ontology]
            data, colors = [], []
            for method in methods:
                values = pd.to_numeric(sub[sub.method == method][key], errors="coerce")
                values = values[np.isfinite(values)].to_numpy()
                data.append(values if values.size else np.array([np.nan]))
                colors.append(METHOD_COLOR[method])
            box = ax.boxplot(data, positions=positions, widths=.62, patch_artist=True,
                             showfliers=False, whis=(2.5, 97.5),
                             medianprops=dict(color="#111111", linewidth=1.0),
                             boxprops=dict(linewidth=.5, edgecolor="#111111"),
                             whiskerprops=dict(linewidth=.6, color="#111111"),
                             capprops=dict(linewidth=.6, color="#111111"))
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(.85)
            for position, values in zip(positions, data):
                finite = values[np.isfinite(values)]
                if finite.size:
                    median = float(np.median(finite))
                    ax.annotate(
                        f"{median:.3f}", (position, median), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.0, rotation=90, clip_on=False,
                    )
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
            if row == len(keys) - 1:
                ax.set_xticks(positions, [METHOD_LABEL.get(m, m) for m in methods],
                              rotation=35, ha="right", fontsize=5.2)
            else:
                ax.set_xticks(positions, [""] * len(methods))
            label_panel(ax, chr(97 + panel))
            panel += 1
    handles = [Patch(facecolor=METHOD_COLOR[m], edgecolor="#111111", label=METHOD_LABEL[m])
               for m in methods]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, .95),
               frameon=False, fontsize=5.5, title="Method", title_fontsize=6)
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
                     metric_keys: tuple[str, ...] = ACCURACY_METRICS,
                     stem: str = "benchmark_accuracy_metric_grid",
                     tier: str = SUPPLEMENTARY) -> None:
    """Every headline metric: rows are metrics, columns are ontologies.

    Error bars are drawn only where replicates actually exist. cafa_fmax and
    cafa_smin are bootstrapped, so they get 95% intervals; micro-AUPR and
    coverage are single point estimates in the results table and are left
    bare, with the absence stated in the caption rather than implied by a
    missing whisker.
    """
    methods = [m for m in _methods_present(metrics) if m not in invalid]
    keys = [k for k in metric_keys if k in metrics.columns]
    has_ci = set()
    if bootstrap is not None:
        has_ci = {k for k in keys if k in bootstrap.columns}
    fig, axes = plt.subplots(len(keys), 3, figsize=(DOUBLE_COLUMN_IN, 1.85 * len(keys)),
                             squeeze=False, sharex="row")
    x = np.arange(len(methods))
    panel = 0
    for row, key in enumerate(keys):
        label, _ = METRICS[key]
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
            ax.set_ylim(0, top)
            bars = ax.bar(x, values, color=[METHOD_COLOR[m] for m in methods],
                          edgecolor="#111111", linewidth=.4, width=.72, yerr=yerr,
                          error_kw=dict(elinewidth=.7, capsize=1.8, ecolor="#111111"))
            label_vertical_bars(
                ax, bars, values, yerr, fontsize=5.0, rotation=90,
            )
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            if col == 0:
                ax.set_ylabel(label, fontsize=6.5)
            if row == len(keys) - 1:
                ax.set_xticks(x, [METHOD_LABEL.get(m, m) for m in methods],
                              rotation=35, ha="right", fontsize=5.2)
            else:
                ax.set_xticks(x, [""] * len(methods))
            label_panel(ax, chr(97 + panel))
            panel += 1
    handles = [Patch(facecolor=METHOD_COLOR[m], edgecolor="#111111", label=METHOD_LABEL[m])
               for m in methods]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, .95),
               frameon=False, fontsize=5.5, title="Method", title_fontsize=6)
    excluded = ", ".join(METHOD_LABEL.get(m, m) for m in invalid) or "none"
    ci_note = ("Metrics with paired bootstrap draws carry percentile 95% intervals "
               "from 1,000 resamples; metrics without draws are shown as point "
               "estimates without invented error bars. ")
    fig.text(.5, -.05, ci_note +
             f"Rows share a y-axis across ontologies so cross-ontology magnitudes are comparable. "
             f"Coverage is the fraction of the 754 test proteins receiving any non-zero score; a "
             f"method with high F$_{{max}}$ but low coverage annotates few proteins confidently "
             f"rather than annotating all of them well. Excluded methods: {excluded}. "
             + provenance("src/plot_benchmark.py", "results/benchmark_metrics.csv"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / stem, tier)


def plot_individual_metric(
    metrics: pd.DataFrame,
    invalid: dict[str, str],
    out: Path,
    key: str,
    bootstrap: pd.DataFrame | None = None,
) -> None:
    """One three-ontology figure for one metric, with exact values printed."""
    methods = [m for m in _methods_present(metrics) if m not in invalid]
    label, _direction = METRICS[key]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.0), sharey=True)
    y = np.arange(len(methods))
    for panel, (ax, ontology) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        subset = metrics[metrics.ontology == ontology].set_index("method")
        values = np.asarray([
            float(subset.loc[method, key]) if method in subset.index else np.nan
            for method in methods
        ])
        errors = None
        if bootstrap is not None and key in bootstrap.columns:
            lower, upper = [], []
            bsub = bootstrap[bootstrap.ontology == ontology]
            for method, value in zip(methods, values):
                draws = pd.to_numeric(
                    bsub.loc[bsub.method == method, key], errors="coerce"
                ).dropna().to_numpy()
                if draws.size and np.isfinite(value):
                    lower.append(max(value - np.percentile(draws, 2.5), 0.0))
                    upper.append(max(np.percentile(draws, 97.5) - value, 0.0))
                else:
                    lower.append(0.0)
                    upper.append(0.0)
            errors = np.asarray([lower, upper])
        bars = ax.barh(
            y, values, xerr=errors, color=[METHOD_COLOR[m] for m in methods],
            edgecolor="#111111", linewidth=.45, height=.68,
            error_kw=dict(elinewidth=.7, capsize=1.8, ecolor="#111111"),
        )
        ax.set_yticks(y, [METHOD_LABEL[m] for m in methods], fontsize=5.8)
        ax.invert_yaxis()
        ax.set_title(ONTOLOGY_SHORT[ontology])
        ax.set_xlabel(label, fontsize=7)
        finite = values[np.isfinite(values)]
        maximum = float(finite.max()) if finite.size else 1.0
        ax.set_xlim(0, maximum * 1.22 if maximum > 0 else 1.0)
        label_horizontal_bars(ax, bars, values, errors, fontsize=5.5)
        label_panel(ax, chr(97 + panel))
    handles = [
        Patch(facecolor=METHOD_COLOR[m], edgecolor="#111111", label=METHOD_LABEL[m])
        for m in methods
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.0, .95),
        frameon=False, fontsize=5.5, title="Method", title_fontsize=6,
    )
    savefig(fig, out / f"metric_{METRIC_STEMS.get(key, key)}", SUPPLEMENTARY)

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
                       s=26, color=METHOD_COLOR[method],
                       edgecolor="#111111", linewidth=.4, zorder=3)
            x_value = float(row["protein_coverage_any_score"])
            y_value = float(row["cafa_fmax"])
            ax.annotate(
                        f"{METHOD_LABEL.get(method, method).split(' (')[0]}\n"
                        f"({x_value:.3f}, {y_value:.3f})",
                        (x_value, y_value),
                        xytext=(3, 3), textcoords="offset points", fontsize=5.0, color="#333333")
        ax.set_title(ONTOLOGY_SHORT[ontology])
        ax.set_xlabel("Protein coverage", fontsize=7)
        ax.set_xlim(-.05, 1.08)
        label_panel(ax, chr(97 + panel))
    axes[0].set_ylabel("F$_{max}$", fontsize=7)
    fig.text(.5, -.10,
             "Homology-transfer baselines only score proteins with a qualifying hit, so their "
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
        help="Render requested comparison methods that are present instead of failing if any are missing.",
    )
    args = ap.parse_args()

    apply_style()
    distance, first, second = validate_method_palette()
    print(f"Method palette minimum CIELAB separation: {distance:.1f} ({first} vs {second})")
    workspace = args.workspace.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    metrics, audit, bootstrap = load(workspace)
    available = set(metrics["method"].astype(str))
    missing = [method for method in METHOD_ORDER if method not in available]
    if missing and not args.allow_missing:
        raise SystemExit(
            "Requested comparison results are missing from results/benchmark_metrics.csv: "
            + ", ".join(missing)
            + ". Complete/evaluate those baselines on ARC, or use --allow-missing "
              "for a partial diagnostic render."
        )
    if missing:
        print("WARNING: partial comparison render; missing: " + ", ".join(missing))
    removed = sorted(available - REQUESTED_METHODS)
    if removed:
        print("Locked comparison filter removed: " + ", ".join(removed))
    metrics = select_requested_methods(metrics)
    if audit is not None and "method" in audit.columns:
        audit = select_requested_methods(audit)
    bootstrap = select_requested_methods(bootstrap)
    if metrics.empty:
        raise SystemExit("None of the requested comparison methods are present in benchmark_metrics.csv")
    invalid = detect_invalid(metrics, workspace)
    if invalid:
        print("\nINTEGRITY: methods excluded from scoring")
        for method, reason in invalid.items():
            print(f"  {method}: {reason}")
        (out / "excluded_methods.json").write_text(json.dumps(invalid, indent=2) + "\n")
    else:
        print("All methods have non-empty predictions.")

    valid = metrics[~metrics.method.isin(invalid)]
    print("\nFmax by method (valid methods only):")
    print(valid.pivot_table(index="method", columns="ontology_short",
                            values="cafa_fmax").round(4).to_string())

    plot_metric_panels(metrics, invalid, out, MAIN if args.tier == "main" else SUPPLEMENTARY)
    if bootstrap is not None:
        plot_bootstrap_boxplots(bootstrap, invalid, out, MAIN)
    else:
        print("NOTE: results/bootstrap_metrics.csv absent; skipping boxplot figure.")
    plot_metric_grid(
        metrics, invalid, out, bootstrap, ACCURACY_METRICS,
        "benchmark_accuracy_metric_grid",
    )
    plot_metric_grid(
        metrics, invalid, out, bootstrap, OPERATING_METRICS,
        "benchmark_calibration_coverage_grid",
    )
    for key in METRICS:
        if key in metrics.columns:
            plot_individual_metric(metrics, invalid, out, key, bootstrap)
    plot_coverage_vs_fmax(metrics, invalid, out)
    metrics.to_csv(out / "benchmark_metrics_plotted.csv", index=False)
    if bootstrap is not None:
        bootstrap.to_csv(out / "benchmark_bootstrap_plotted.csv", index=False)
    if audit is not None:
        audit.to_csv(out / "benchmark_coverage_mapping_audit.csv", index=False)
    print(f"\nWrote benchmark figures to {out}")


if __name__ == "__main__":
    main()
