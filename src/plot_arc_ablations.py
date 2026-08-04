#!/usr/bin/env python3
"""Publication figures and integrity audits for ARC input ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import patheffects as path_effects
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.pickle_compat import load_pickle_compat

from plot_style import (
    CONSTANT_PREDICTOR_CAVEAT,
    DOUBLE_COLUMN_IN,
    ERROR_CAPTION,
    ERROR_KIND_LABEL,
    SUPPLEMENTARY,
    assert_palette_locked,
    provenance,
    report_colorblind_audit,
    METRIC_LABEL,
    METRIC_ORDER,
    MODEL_COLOR,
    MODEL_MARKER,
    MODEL_ORDER,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    VARIANT_HATCH,
    VARIANT_LABEL,
    VARIANT_ORDER,
    annotate_insufficient_data,
    apply_style,
    colorblind_audit,
    jitter,
    label_panel,
    label_vertical_bars,
    mean_and_error,
    savefig,
)

EXPECTED_SEEDS = 5
VARIANT_MARKER = {"full": "o", "seq_only": "^", "struct_only": "s"}
DEFAULT_SUPPORT_ROOT = Path("preprocessing/data_arc_rebuild_2026_07_14/datasets/threshold_30")


def read_results(root: Path, logs: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("test_metrics.json"):
        relative = path.relative_to(root).parts
        if len(relative) >= 5:
            rows.append({
                "ontology": relative[0], "model": relative[1], "input": relative[2],
                "seed": relative[3], **json.loads(path.read_text()),
            })
    covered = {(row["ontology"], row["model"], row["input"]) for row in rows}
    # rglob, not glob: the SLURM logs were reorganised into logs/arc_ablation/
    # at some point, and a flat glob silently matched nothing — which would
    # drop every ontology that has no materialised result folder rather than
    # failing loudly.
    log_files = sorted(logs.rglob("arc_ablation_*.out"))
    if not log_files:
        print(f"NOTE: no arc_ablation_*.out logs found under {logs} (searched recursively).")
    for path in log_files:
        for line in reversed(path.read_text(errors="ignore").splitlines()):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not {"input_modality", "model", "ontology", "test"}.issubset(item):
                continue
            key = (item["ontology"], item["model"], item["input_modality"])
            if key not in covered:
                rows.append({
                    "ontology": item["ontology"], "model": item["model"],
                    "input": item["input_modality"],
                    "seed": f"log_{path.stem.rsplit('_', 1)[-1]}",
                    **item["test"],
                })
            break
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["seed"] = frame["seed"].astype(str)
    return frame


def _records(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        obj = load_pickle_compat(handle)
    if isinstance(obj, list):
        return obj
    if hasattr(obj, "protein_ids") and hasattr(obj, "labels") and hasattr(obj, "terms"):
        return [
            {"id": pid, "labels": [term for term, value in zip(obj.terms, row) if value > 0]}
            for pid, row in zip(obj.protein_ids, obj.labels)
        ]
    raise TypeError(f"Unsupported support dataset: {path}")


def audit_ablation_integrity(df: pd.DataFrame, support_root: Path, ablations_root: Path, out: Path) -> pd.DataFrame:
    rows = []
    for ontology in ONTOLOGY_ORDER:
        path = support_root / f"{ontology}_test.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing support dataset for {ontology}: {path}")
        records = _records(path)
        term_counts: dict[str, int] = {}
        for record in records:
            for term in set(record.get("labels", [])):
                term_counts[term] = term_counts.get(term, 0) + 1
        support_values = np.asarray(list(term_counts.values()), dtype=int)
        checkpoint_count = len(list((ablations_root / ontology).rglob("best_checkpoint.pt")))
        prediction_artifacts_available = checkpoint_count > 0
        row = {
            "ontology": ontology,
            "test_examples": len(records),
            "test_terms_with_positive_support": int((support_values > 0).sum()),
            "terms_with_support_le_5": int((support_values <= 5).sum()),
            "terms_with_support_le_10": int((support_values <= 10).sum()),
            "median_term_support": float(np.median(support_values)) if support_values.size else np.nan,
            "minimum_term_support": int(support_values.min()) if support_values.size else 0,
            "maximum_term_support": int(support_values.max()) if support_values.size else 0,
            "checkpoint_count": checkpoint_count,
            "prediction_artifacts_available": prediction_artifacts_available,
            "raw_predicted_term_counts_available": prediction_artifacts_available,
        }
        rows.append(row)
        focus = df[(df.ontology == ontology) & (df.model == "MLP") & (df.input == "struct_only")]
        if not focus.empty:
            print(
                f"{ontology} MLP/structure-only: mean Macro-AUPRC={focus.Macro_AUPRC.mean():.4f}, "
                f"mean Macro-AUROC={focus.Macro_AUROC.mean():.4f}; "
                f"{row['terms_with_support_le_5']} of {row['test_terms_with_positive_support']} "
                "positive-support terms have <=5 test examples."
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "ablation_integrity_audit.csv", index=False)
    (out / "ablation_integrity_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not audit["raw_predicted_term_counts_available"].all():
        print("WARNING: best checkpoints/prediction arrays are absent locally; raw per-protein "
              "predicted-term counts cannot be audited. Smin interpretation is marked accordingly.")
    print("Interpretation: MLP structure-only zeroes node features, so its output is a "
          "per-term bias/constant score; Macro-AUROC near 0.5 is chance and Macro-AUPRC "
          "tracks prevalence, not evidence of structural function-prediction capability.")
    return audit


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
    missing = coverage[~coverage.complete]
    if missing.empty:
        print(f"Coverage: all {len(coverage)} cells have >= {EXPECTED_SEEDS} seeds.")
    else:
        print(f"Coverage: {len(missing)}/{len(coverage)} cells have fewer than {EXPECTED_SEEDS} seeds.")
        for _, row in missing.iterrows():
            print(f"  {row.ontology:20s} {row.model:10s} {row.input_modality:12s} seeds_found={row.seeds_found}")


def _bar_positions(n_groups: int, n_series: int, width: float = .25, gap: float = 1.05):
    x = np.arange(n_groups)
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * width * gap
    return x, offsets


def _metric_limits(df: pd.DataFrame, metric: str, extra: float = 0.0) -> tuple[float, float]:
    values = pd.to_numeric(df[metric], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * .08, .02)
    return max(0.0, lo - pad), hi + pad + extra


def _n_label(ax, xi: float, n: int) -> None:
    if 0 < n < EXPECTED_SEEDS:
        ax.text(xi, .015, f"n={n}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.3, color="#555555")


def _legend(fig, variant_glyphs) -> None:
    model_handles = [
        Patch(facecolor=MODEL_COLOR[m], edgecolor="#111111", label=m) for m in MODEL_ORDER
    ]
    first = fig.legend(handles=model_handles, title="Model", loc="upper left",
                       bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6.5, title_fontsize=7)
    fig.add_artist(first)
    fig.legend(handles=variant_glyphs, title="Input", loc="upper left",
               bbox_to_anchor=(1.0, .55), frameon=False, fontsize=6.5, title_fontsize=7)


def _empty_panel(ax, panel: int, ontology: str) -> None:
    annotate_insufficient_data(ax)
    ax.set_title(ONTOLOGY_SHORT[ontology])
    label_panel(ax, chr(97 + panel))


def _metric_note(metric: str) -> str:
    """Caption text carrying the audited caveats for this metric.

    Each statement here is backed by a reproduced number in
    docs/figure_data_integrity.md; none are impressions of the plot.
    """
    parts = []
    if metric in {"Micro_Fmax", "Macro_Fmax", "Smin"}:
        parts.append("F$_{max}$/S$_{min}$ favour graph-aware Hybrid/Hybrid_JK.")
    if metric in {"Micro_AUROC", "Macro_AUROC", "Micro_AUPRC", "Macro_AUPRC"}:
        parts.append("AUROC/AUPRC rank the models differently from F$_{max}$/S$_{min}$; "
                     "see the paired metric-family figure.")
    if metric in {"Macro_AUPRC", "Micro_AUPRC"}:
        parts.append("AUPRC is calculated with the average-precision estimator.")
    if metric in {"Macro_AUPRC", "Macro_AUROC", "Micro_AUROC"}:
        parts.append(CONSTANT_PREDICTOR_CAVEAT + " Its Macro-AUROC is 0.500 (chance); the "
                     "~0.70 Micro-AUROC reflects the term-frequency prior, not structure.")
    if metric == "Smin":
        parts.append("Under structure-only, MLP/GAT/GCN reach exactly the predict-nothing S$_{min}$ "
                     "(the test set's total IC), while Hybrid/Hybrid_JK saturate above the 0.99 "
                     "threshold-scan bound and predict everything; both are degenerate, not tuned "
                     "trade-offs.")
    return " ".join(parts)


def annotate_constant_predictor(ax, x: float, y: float, metric: str) -> None:
    """Mark the MLP/structure-only cell wherever its value is an artifact."""
    if metric in {"Macro_AUPRC", "Macro_AUROC", "Micro_AUROC", "Smin"} and np.isfinite(y):
        ax.annotate("†", (x, y), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, fontweight="bold", color="#b00000")


def plot_dynamite(df: pd.DataFrame, out: Path, metric: str, err_kind: str = "sd",
                  tier: str = SUPPLEMENTARY) -> None:
    """Vertically faceted grouped bars; modalities are never additively stacked."""
    ymin, ymax = _metric_limits(df, metric, .08 if metric == "Micro_Fmax" else 0.0)
    fig, axes = plt.subplots(
        len(VARIANT_ORDER), len(ONTOLOGY_ORDER),
        figsize=(DOUBLE_COLUMN_IN, 2.05 * len(VARIANT_ORDER)),
        sharex=True, sharey=True, squeeze=False,
    )
    x = np.arange(len(MODEL_ORDER), dtype=float)
    panel = 0
    for row, variant in enumerate(VARIANT_ORDER):
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row, col]
            sub = df[(df.ontology == ontology) & (df.input == variant)]
            means, errors, counts = [], [], []
            for model in MODEL_ORDER:
                values = pd.to_numeric(
                    sub[sub.model == model][metric], errors="coerce"
                ).dropna().to_numpy(float)
                mean, error, count = mean_and_error(values, err_kind)
                means.append(mean)
                errors.append(error)
                counts.append(count)
            means_array = np.asarray(means, dtype=float)
            errors_array = np.asarray(errors, dtype=float)
            valid = np.isfinite(means_array)
            if not valid.any():
                ax.remove()
                continue
            ax.set_ylim(ymin, ymax)
            bars = ax.bar(
                x[valid], means_array[valid], width=.72,
                yerr=errors_array[valid],
                color=[MODEL_COLOR[model] for model, keep in zip(MODEL_ORDER, valid) if keep],
                edgecolor="#111111", linewidth=.45,
                error_kw=dict(elinewidth=.75, capsize=2, ecolor="#111111"),
            )
            label_vertical_bars(
                ax, bars, means_array[valid], errors_array[valid],
                fontsize=5.0, rotation=90,
            )
            for position, count in zip(x, counts):
                _n_label(ax, position, count)
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            if col == 0:
                ax.set_ylabel(f"{VARIANT_LABEL[variant]}\n{METRIC_LABEL[metric]}", fontsize=6.7)
            if row == len(VARIANT_ORDER) - 1:
                ax.set_xticks(x, MODEL_ORDER, rotation=30, ha="right", fontsize=6)
            label_panel(ax, chr(97 + panel))
            panel += 1
    fig.text(
        .5, -.025,
        f"Rows are input modalities and columns are ontologies. Bars show the mean with "
        f"{ERROR_KIND_LABEL[err_kind]} across five seeds. Bars are grouped/faceted, "
        f"not stacked, because model scores are not additive. {_metric_note(metric)}",
        ha="center", fontsize=5.4, wrap=True,
    )
    savefig(fig, out / f"dynamite_{metric.lower()}.png", tier)


def plot_strip(df: pd.DataFrame, out: Path, metric: str, err_kind: str = "sd",
               tier: str = SUPPLEMENTARY) -> None:
    ymin, ymax = _metric_limits(df, metric)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_IN, 3.35), sharey=True)
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
                    xs = xi + jitter(vals.size, width=.045, seed=seed_for_jitter)
                    ax.scatter(xs, vals, s=12, color=MODEL_COLOR[model],
                               marker=VARIANT_MARKER[variant], alpha=.62,
                               linewidth=.25, edgecolor="#111111", zorder=3)
                mean, err, n = mean_and_error(vals, err_kind)
                if n:
                    mean_x = xi + .075
                    ax.plot([xi, mean_x], [mean, mean], color="#777777", linewidth=.45, alpha=.7, zorder=4)
                    ax.errorbar([mean_x], [mean], yerr=[err], fmt="D", markersize=3.2,
                                markerfacecolor="#111111", markeredgecolor="#111111",
                                markeredgewidth=.6, ecolor="#111111", elinewidth=1.0,
                                capsize=2.2, zorder=5)
                    if model == "MLP" and variant == "struct_only":
                        annotate_constant_predictor(ax, mean_x, mean, metric)
                _n_label(ax, xi, n)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x, MODEL_ORDER, rotation=30, ha="right")
        ax.set_title(ONTOLOGY_SHORT[ontology])
        label_panel(ax, chr(97 + panel))
    if not any_data:
        plt.close(fig)
        raise SystemExit(f"No ablation data found for {metric!r}")
    axes[0].set_ylabel(f"{METRIC_LABEL[metric]}\n(points = seeds; black diamond = {ERROR_KIND_LABEL[err_kind]})")
    variant_handles = [
        Line2D([0], [0], marker=VARIANT_MARKER[v], linestyle="", color="#555555",
               label=VARIANT_LABEL[v], markersize=5) for v in VARIANT_ORDER
    ]
    _legend(fig, variant_handles)
    fig.text(.5, -.055, f"{ERROR_CAPTION[err_kind]}; points are individual seeds and the small black diamond is offset to its right. "
             f"† marks the MLP structure-only constant-predictor control. {_metric_note(metric)} "
             + provenance("src/plot_arc_ablations.py", "ablations/**/test_metrics.json + logs/arc_ablation_*.out"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / f"strip_{metric.lower()}.png", tier)


def plot_strip_faceted(df: pd.DataFrame, out: Path, metric: str, err_kind: str = "sd",
                       tier: str = SUPPLEMENTARY) -> None:
    """Ablation strip plot with input modality as ROWS instead of marker shapes.

    The overlaid version packs 15 model x modality groups into each panel and
    asks the reader to decode circle/triangle/square while also tracking hue.
    Giving modality its own row drops that to five groups per panel, lets model
    identity ride on the x-axis where it needs no legend at all, and makes the
    Full -> Sequence -> Structure comparison a straight vertical read down a
    shared y-axis.
    """
    ymin, ymax = _metric_limits(df, metric)
    fig, axes = plt.subplots(len(VARIANT_ORDER), len(ONTOLOGY_ORDER),
                             figsize=(DOUBLE_COLUMN_IN, 2.15 * len(VARIANT_ORDER)),
                             sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(MODEL_ORDER), dtype=float)
    panel = 0
    for row, variant in enumerate(VARIANT_ORDER):
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row][col]
            sub = df[(df.ontology == ontology) & (df.input == variant)]
            if sub.empty:
                annotate_insufficient_data(ax)
                label_panel(ax, chr(97 + panel))
                panel += 1
                continue
            for i, model in enumerate(MODEL_ORDER):
                vals = sub[sub.model == model][metric].to_numpy()
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    seed = abs(hash((ontology, model, variant))) % (2 ** 32)
                    ax.scatter(x[i] + jitter(vals.size, width=.085, seed=seed), vals,
                               s=13, color=MODEL_COLOR[model], marker="o", alpha=.55,
                               linewidth=.25, edgecolor="#111111", zorder=3)
                mean, err, n = mean_and_error(vals, err_kind)
                if n:
                    ax.errorbar([x[i] + .23], [mean], yerr=[err], fmt="_", markersize=9,
                                markeredgecolor="#111111", markeredgewidth=1.4,
                                ecolor="#111111", elinewidth=1.0, capsize=2.2, zorder=5)
                    if model == "MLP" and variant == "struct_only":
                        annotate_constant_predictor(ax, x[i] + .23, mean, metric)
                _n_label(ax, x[i], n)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(-.6, len(MODEL_ORDER) - .2)
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            if col == 0:
                ax.set_ylabel(f"{VARIANT_LABEL[variant]}\n{METRIC_LABEL[metric]}", fontsize=6.5)
            if row == len(VARIANT_ORDER) - 1:
                ax.set_xticks(x, MODEL_ORDER, rotation=30, ha="right", fontsize=6)
            label_panel(ax, chr(97 + panel))
            panel += 1
    fig.text(.5, -.03,
             f"Rows are input modality, columns are GO sub-ontology, and each panel shows the five "
             f"architectures on a shared y-axis — modality is a facet rather than a marker shape, so "
             f"no symbol decoding is required. Points are individual seeds; the horizontal bar is the "
             f"mean with {ERROR_CAPTION[err_kind].lower()}. † marks the MLP structure-only "
             f"constant-predictor control. {_metric_note(metric)} "
             + provenance("src/plot_arc_ablations.py", "ablations/**/test_metrics.json"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / f"faceted_{metric.lower()}.png", tier)


def plot_box_faceted(df: pd.DataFrame, out: Path, metric: str, err_kind: str = "sd",
                     tier: str = SUPPLEMENTARY) -> None:
    """Boxplot version of the faceted ablation, with the seeds drawn on top.

    Read the caveat before using this in the main text: each box summarises
    only n = 5 seeds, so its quartiles are estimated from five numbers and the
    box will look confident whether or not it deserves to. The individual
    seeds are therefore always overlaid, and they — not the box — are the
    evidence. Where a real distribution exists (the 1,000-replicate benchmark
    bootstrap) a boxplot stands on its own; here it does not.
    """
    ymin, ymax = _metric_limits(df, metric)
    fig, axes = plt.subplots(len(VARIANT_ORDER), len(ONTOLOGY_ORDER),
                             figsize=(DOUBLE_COLUMN_IN, 2.15 * len(VARIANT_ORDER)),
                             sharex=True, sharey=True, squeeze=False)
    positions = np.arange(len(MODEL_ORDER), dtype=float)
    panel = 0
    for row, variant in enumerate(VARIANT_ORDER):
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row][col]
            sub = df[(df.ontology == ontology) & (df.input == variant)]
            if sub.empty:
                annotate_insufficient_data(ax)
                label_panel(ax, chr(97 + panel))
                panel += 1
                continue
            data = []
            for model in MODEL_ORDER:
                values = sub[sub.model == model][metric].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                data.append(values if values.size else np.array([np.nan]))
            box = ax.boxplot(data, positions=positions, widths=.6, patch_artist=True,
                             showfliers=False,
                             medianprops=dict(color="#111111", linewidth=.9),
                             boxprops=dict(linewidth=.45, edgecolor="#111111"),
                             whiskerprops=dict(linewidth=.55, color="#111111"),
                             capprops=dict(linewidth=.55, color="#111111"))
            for patch, model in zip(box["boxes"], MODEL_ORDER):
                patch.set_facecolor(MODEL_COLOR[model])
                patch.set_alpha(.45)
            for i, (model, values) in enumerate(zip(MODEL_ORDER, data)):
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                seed = abs(hash((ontology, model, variant))) % (2 ** 32)
                ax.scatter(positions[i] + jitter(values.size, width=.10, seed=seed), values,
                           s=11, color=MODEL_COLOR[model], edgecolor="#111111",
                           linewidth=.3, zorder=4)
                if model == "MLP" and variant == "struct_only":
                    annotate_constant_predictor(ax, positions[i], float(np.nanmean(values)), metric)
            ax.set_ylim(ymin, ymax)
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            if col == 0:
                ax.set_ylabel(f"{VARIANT_LABEL[variant]}\n{METRIC_LABEL[metric]}", fontsize=6.5)
            if row == len(VARIANT_ORDER) - 1:
                ax.set_xticks(positions, MODEL_ORDER, rotation=30, ha="right", fontsize=6)
            label_panel(ax, chr(97 + panel))
            panel += 1
    fig.text(.5, -.03,
             f"Rows are input modality, columns are ontology. Boxes show the median and "
             f"interquartile range across n = 5 training seeds and whiskers the full range; "
             f"because five points is a thin basis for quartiles, every seed is plotted on top "
             f"and the points are the evidence. † marks the MLP structure-only constant-predictor "
             f"control. {_metric_note(metric)} "
             + provenance("src/plot_arc_ablations.py", "ablations/**/test_metrics.json"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / f"box_{metric.lower()}.png", tier)


def plot_metric_family_composite(df: pd.DataFrame, out: Path,
                                 top_metric: str = "Micro_Fmax",
                                 bottom_metric: str = "Micro_AUROC",
                                 err_kind: str = "sd", tier: str = SUPPLEMENTARY) -> None:
    """One figure carrying the metric-family contradiction.

    Rows are the two metric families, columns are ontologies, layout and colour
    identical between rows, and a single shared legend. The point is that the
    model ranking inverts between rows; splitting these across two figures
    forces the reader to hold one ranking in memory while reading the other.
    """
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_IN, 4.9), squeeze=False)
    x, offsets = _bar_positions(len(MODEL_ORDER), len(VARIANT_ORDER))
    panel = 0
    for row, metric in enumerate((top_metric, bottom_metric)):
        ymin, ymax = _metric_limits(df, metric)
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row][col]
            sub = df[df.ontology == ontology]
            if sub.empty:
                annotate_insufficient_data(ax)
                label_panel(ax, chr(97 + panel))
                panel += 1
                continue
            for j, variant in enumerate(VARIANT_ORDER):
                for i, model in enumerate(MODEL_ORDER):
                    vals = sub[(sub.model == model) & (sub.input == variant)][metric].to_numpy()
                    vals = vals[np.isfinite(vals)]
                    xi = x[i] + offsets[j]
                    if vals.size:
                        seed = abs(hash((ontology, model, variant))) % (2 ** 32)
                        ax.scatter(xi + jitter(vals.size, width=.045, seed=seed), vals,
                                   s=10, color=MODEL_COLOR[model], marker=VARIANT_MARKER[variant],
                                   alpha=.6, linewidth=.25, edgecolor="#111111", zorder=3)
                    mean, err, n = mean_and_error(vals, err_kind)
                    if n:
                        mean_x = xi + .075
                        ax.plot([xi, mean_x], [mean, mean], color="#777777",
                                linewidth=.4, alpha=.7, zorder=4)
                        ax.errorbar([mean_x], [mean], yerr=[err], fmt="D", markersize=3.0,
                                    markerfacecolor="#111111", markeredgecolor="#111111",
                                    markeredgewidth=.5, ecolor="#111111", elinewidth=.9,
                                    capsize=1.9, zorder=5)
                        if model == "MLP" and variant == "struct_only":
                            annotate_constant_predictor(ax, mean_x, mean, metric)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(x, MODEL_ORDER, rotation=30, ha="right", fontsize=6)
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology])
            label_panel(ax, chr(97 + panel))
            panel += 1
        axes[row][0].set_ylabel(METRIC_LABEL[metric], fontsize=7.5)
    variant_handles = [
        Line2D([0], [0], marker=VARIANT_MARKER[v], linestyle="", color="#555555",
               label=VARIANT_LABEL[v], markersize=5) for v in VARIANT_ORDER
    ]
    _legend(fig, variant_handles)
    fig.text(.5, -.045,
             f"Top row: {METRIC_LABEL[top_metric]}. Bottom row: {METRIC_LABEL[bottom_metric]}. "
             f"Identical layout and colour mapping in both rows — only the metric changes, and the "
             f"model ranking inverts with it, so neither family alone supports a single "
             f"'best model' claim. {ERROR_CAPTION[err_kind]}; points are seeds, small black diamonds the "
             f"mean. † marks the MLP structure-only constant-predictor control "
             f"(Macro-AUROC 0.500 = chance). "
             + provenance("src/plot_arc_ablations.py", "ablations/**/test_metrics.json"),
             ha="center", fontsize=5.2, wrap=True)
    savefig(fig, out / f"metric_family_{top_metric.lower()}_vs_{bottom_metric.lower()}.png", tier)


def _contrast_text(rgba) -> tuple[str, list]:
    """Pick black or white cell text and guarantee it clears WCAG on any fill.

    A filled label box was tried first and read as a coloured block sitting on
    top of the data. A thin outline in the opposite ink does the same job —
    it raises effective contrast on mid-tone viridis cells — without adding a
    second rectangle to every cell.
    """
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = float(.2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2])
    black_ratio = (luminance + .05) / .05
    white_ratio = 1.05 / (luminance + .05)
    color, ratio = ("white", white_ratio) if white_ratio >= black_ratio else ("black", black_ratio)
    if ratio < 4.5:
        outline = "#000000" if color == "white" else "#ffffff"
        return color, [path_effects.withStroke(linewidth=1.6, foreground=outline)]
    return color, []


def plot_heatmap(df: pd.DataFrame, out: Path, metric: str = "Micro_Fmax",
                 tier: str = SUPPLEMENTARY) -> None:
    values = pd.to_numeric(df[metric], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    vmin = max(0.0, float(values.min())) if values.size else 0.0
    vmax = min(1.0, float(values.max())) if values.size else 1.0
    if vmax <= vmin:
        vmax = vmin + .05
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(DOUBLE_COLUMN_IN, 3.55), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, .055), hspace=.42, wspace=.16)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    image = None
    for panel, (ax, ontology) in enumerate(zip(axes, ONTOLOGY_ORDER)):
        sub = df[df.ontology == ontology]
        table = sub.pivot_table(
            index="model", columns="input", values=metric, aggfunc="mean"
        ).reindex(index=MODEL_ORDER, columns=VARIANT_ORDER)
        image = ax.imshow(table.values, cmap=cmap, aspect="auto", norm=norm)
        ax.set_xticks(
            range(len(VARIANT_ORDER)), [VARIANT_LABEL[v] for v in VARIANT_ORDER],
            rotation=22, ha="right",
        )
        ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
        ax.set_title(ONTOLOGY_SHORT[ontology])
        label_panel(ax, chr(97 + panel))
        ax.grid(False)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.iloc[i, j]
                if pd.notna(value):
                    color, effects = _contrast_text(cmap(norm(float(value))))
                    ax.text(
                        j, i, f"{value:.3f}", ha="center", va="center",
                        color=color, fontweight="bold", fontsize=6.3,
                        path_effects=effects,
                    )
    cax = fig.add_subplot(grid[1, :])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.set_label(
        f"Mean {METRIC_LABEL[metric]} (observed range {vmin:.3f}–{vmax:.3f})",
        fontsize=6,
    )
    cax.tick_params(labelsize=5.4)
    fig.text(
        .5, -.025,
        "The colorbar occupies a dedicated grid row and cannot overlap heatmap cells or labels. "
        f"Exact cell values are also exported as a supplementary table. {_metric_note(metric)}",
        ha="center", fontsize=5.6, wrap=True,
    )
    savefig(fig, out / f"{metric.lower()}_model_input_heatmap.png", tier)


def export_ablation_tables(df: pd.DataFrame, coverage: pd.DataFrame, out: Path) -> None:
    """Export raw, summary, and heatmap-ready values for the supplement."""
    table_dir = out / "supplementary_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics = [metric for metric in METRIC_ORDER if metric in df]
    raw_columns = ["ontology", "model", "input", "seed", *metrics]
    raw = df[raw_columns].rename(columns={"Smin": "Smin_freq"})
    raw.to_csv(table_dir / "supp_table_ablation_seed_metrics.csv", index=False)
    long = df[raw_columns].melt(
        id_vars=["ontology", "model", "input", "seed"],
        value_vars=metrics, var_name="metric", value_name="value",
    )
    summary = long.groupby(
        ["ontology", "model", "input", "metric"], dropna=False, observed=True
    )["value"].agg(
        seeds="count", mean="mean", sd="std", median="median",
        q1=lambda values: values.quantile(.25),
        q3=lambda values: values.quantile(.75),
        minimum="min", maximum="max",
    ).reset_index()
    summary["metric"] = summary["metric"].replace({"Smin": "Smin_freq"})
    summary.to_csv(table_dir / "supp_table_ablation_metric_summary.csv", index=False)
    coverage.to_csv(table_dir / "supp_table_ablation_coverage.csv", index=False)
    summary[summary.metric == "Micro_Fmax"].to_csv(
        table_dir / "supp_table_ablation_heatmap_micro_fmax.csv", index=False
    )
    control = summary[
        (summary.model == "MLP") & (summary.input == "struct_only") &
        summary.metric.isin(["Macro_AUPRC", "Macro_AUROC"])
    ].copy()
    control["interpretation"] = np.where(
        control.metric == "Macro_AUROC",
        "approximately 0.5: chance ranking from a constant per-term predictor",
        "tracks term prevalence; tied-score trapezoidal PR estimate is upward-biased",
    )
    control.to_csv(table_dir / "supp_table_mlp_structure_constant_control.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations-root", type=Path,
                    default=Path("arc_tuning_cafa/ablations/nominal_30_identity_80_coverage"))
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=Path("plots/arc_tuning_cafa/ablations"))
    ap.add_argument("--archive", type=Path, default=None,
                    help="Consolidated per-seed table to fall back on when per-run artifacts are "
                         "incomplete locally (default: plots/arc_tuning_cafa/ablations/ablation_test_metrics.csv).")
    ap.add_argument("--metrics", nargs="+", default=METRIC_ORDER, choices=METRIC_ORDER)
    ap.add_argument("--style", choices=["dynamite", "strip", "both"], default="dynamite")
    ap.add_argument("--err", choices=["sd", "sem", "ci95"], default="sd")
    args = ap.parse_args()

    apply_style()
    print("Palette fingerprint:", assert_palette_locked())
    report_colorblind_audit()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ablations_root = args.ablations_root.resolve()
    df = read_results(ablations_root, args.logs_dir.resolve())
    used_archive = False
    complete_csv = out / "ablation_test_metrics.csv"
    expected_rows = len(ONTOLOGY_ORDER) * len(MODEL_ORDER) * len(VARIANT_ORDER) * EXPECTED_SEEDS
    # The archive is a property of the dataset, not of wherever this run happens
    # to write: looking for it only inside --output-dir meant pointing the script
    # at a fresh directory silently lost every ontology that lives only in the
    # archive.
    archive_csv = args.archive if args.archive else Path("plots/arc_tuning_cafa/ablations/ablation_test_metrics.csv")
    archive_csv = archive_csv.resolve()
    if len(df) < expected_rows and archive_csv.exists():
        archived = pd.read_csv(archive_csv)
        if len(archived) >= expected_rows:
            print(f"Using complete consolidated table {archive_csv} ({len(archived)} rows); "
                  f"local per-run JSON/log artifacts contain only {len(df)} rows.")
            df = archived
            used_archive = True
    if df.empty:
        raise SystemExit("No ablation result JSON, logs, or complete consolidated table found")
    if not used_archive and complete_csv.resolve() != archive_csv:
        df.to_csv(complete_csv, index=False)
    audit_ablation_integrity(df, args.support_root.resolve(), ablations_root, out)
    coverage = coverage_table(df)
    coverage.to_csv(out / "ablation_coverage.csv", index=False)
    report_coverage(coverage)
    export_ablation_tables(df, coverage, out)
    for metric in args.metrics:
        if metric not in df:
            print(f"Skipping {metric}: not present in loaded data")
            continue
        if args.style in ("dynamite", "both"):
            plot_dynamite(df, out, metric, args.err)
        if args.style in ("strip", "both"):
            plot_strip(df, out, metric, args.err)
    if "Micro_Fmax" in df:
        plot_heatmap(df, out, "Micro_Fmax")
    print(f"Loaded {len(df)} completed ablation runs")
    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()
