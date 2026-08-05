#!/usr/bin/env python3
"""Create compact BMC grids for the key ablation and bin-evaluation results."""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_style import (
    MAIN,
    SUPPLEMENTARY,
    BIN_ORDER,
    CATEGORICAL_PALETTE,
    METRIC_LABEL,
    MODEL_COLOR,
    MODEL_ORDER,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    apply_style,
    assert_palette_locked,
    check_min_font,
    savefig,
)

MM_PER_INCH = 25.4
BMC_WIDTH_IN = 170.0 / MM_PER_INCH
RESULTS_HEIGHT_IN = 185.0 / MM_PER_INCH
BINS_HEIGHT_IN = 190.0 / MM_PER_INCH

KEY_METRICS = ("Micro_Fmax", "Micro_AUPRC", "Macro_AUPRC")
KEY_MODELS = tuple(MODEL_ORDER)
BIN_MODELS = ("Hybrid", "Hybrid_JK")
BIN_INPUTS = ("full", "seq_only", "struct_only")
INPUT_LABEL = {
    "full": "Full",
    "seq_only": "Sequence only",
    "struct_only": "Structure only",
}
BIN_LABEL = {
    "no_hit": "No hit",
    "<30%": "<30%",
    "30-40%": "30-40%",
    "40-60%": "40-60%",
    ">=60%": ">=60%",
    "no_positive_terms": "No positive\nterms",
    "<2_bits": "<2 bits",
    "2-4_bits": "2-4 bits",
    "4-6_bits": "4-6 bits",
    ">=6_bits": ">=6 bits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("plots/arc_tuning_cafa/ablations/ablation_test_metrics.csv"),
    )
    parser.add_argument(
        "--bin-csv",
        type=Path,
        default=Path("plots/arc_tuning_cafa/bin_evaluation/bin_metrics.csv"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/figures/ablation_grids"),
    )
    parser.add_argument(
        "--allow-unverified-auprc",
        action="store_true",
        help="Render an explicitly watermarked draft when the corrected manifest is absent.",
    )
    parser.add_argument(
        "--grid",
        choices=("both", "metrics", "bins"),
        default="both",
        help="Render both grids, only the AUPRC/Fmax metrics grid, or only the Fmax bin grid.",
    )
    return parser.parse_args()


def validate_provenance(
    manifest_path: Path, allow_unverified: bool
) -> tuple[bool, str]:
    reason = ""
    if not manifest_path.is_file():
        reason = f"missing provenance manifest: {manifest_path}"
    else:
        manifest = json.loads(manifest_path.read_text())
        required = {
            "rows": 225,
            "auprc_estimator": "sklearn.metrics.average_precision_score",
            "smin_weighting": "training-frequency information content",
            "smin_zero_frequency_policy": "one-count floor",
        }
        mismatches = {
            key: (manifest.get(key), expected)
            for key, expected in required.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            reason = f"provenance mismatch: {mismatches}"
    if reason and not allow_unverified:
        raise RuntimeError(
            f"{reason}. Refusing to create a manuscript AUPRC grid from the "
            "pre-correction archive. Regenerate/synchronize ARC results first, "
            "or use --allow-unverified-auprc for a watermarked layout draft."
        )
    return not bool(reason), reason


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def mean_sd(frame: pd.DataFrame, group: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    result = frame.groupby(group, observed=True)[list(metrics)].agg(["mean", "std", "count"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    return result.reset_index()


def panel_letter(ax: plt.Axes, index: int) -> None:
    ax.text(
        -0.13,
        1.04,
        string.ascii_lowercase[index],
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def add_draft_notice(fig: plt.Figure, verified: bool, reason: str) -> None:
    if verified:
        return
    fig.text(
        0.5,
        0.997,
        "DRAFT -- AUPRC provenance not verified; do not use for publication",
        ha="center",
        va="top",
        fontsize=7,
        fontweight="bold",
        color="#b2182b",
    )
    fig.text(
        0.5, 0.978, "Source manifest missing or incompatible",
        ha="center", va="top", fontsize=7.0, color="#6f1d1b",
    )


def plot_key_metrics(
    archive: pd.DataFrame,
    output_dir: Path,
    verified: bool,
    reason: str,
) -> pd.DataFrame:
    require_columns(
        archive,
        {"ontology", "model", "input", "seed", *KEY_METRICS},
        Path("ablation archive"),
    )
    selected = archive[
        archive["model"].isin(KEY_MODELS) & archive["input"].eq("full")
    ].copy()
    summary = mean_sd(selected, ["ontology", "model", "input"], KEY_METRICS)

    expected = summary[[f"{metric}_count" for metric in KEY_METRICS]]
    if not (expected == 5).all().all():
        bad = summary.loc[~(expected == 5).all(axis=1), ["ontology", "model", *expected.columns]]
        raise RuntimeError(f"Expected five seeds for every full-input configuration:\n{bad}")

    apply_style()
    fig, axes = plt.subplots(
        len(KEY_METRICS),
        len(ONTOLOGY_ORDER),
        figsize=(BMC_WIDTH_IN, RESULTS_HEIGHT_IN),
        squeeze=False,
    )
    y = np.arange(len(KEY_MODELS))
    letters = 0

    for row, metric in enumerate(KEY_METRICS):
        row_values = summary[
            summary["ontology"].isin(ONTOLOGY_ORDER)
        ][[f"{metric}_mean", f"{metric}_std"]].to_numpy(dtype=float)
        finite = row_values[np.isfinite(row_values)]
        row_max = float(np.max(finite)) if finite.size else 1.0
        x_max = max(0.05, row_max * 1.22)

        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row, col]
            panel = (
                summary[summary["ontology"].eq(ontology)]
                .set_index("model")
                .reindex(KEY_MODELS)
            )
            values = panel[f"{metric}_mean"].to_numpy(dtype=float)
            errors = panel[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
            colors = [MODEL_COLOR[model] for model in KEY_MODELS]
            bars = ax.barh(
                y,
                values,
                xerr=errors,
                color=colors,
                edgecolor=["#333333" if model in BIN_MODELS else "#666666" for model in KEY_MODELS],
                linewidth=[1.0 if model in BIN_MODELS else 0.45 for model in KEY_MODELS],
                error_kw={"elinewidth": 0.7, "capsize": 1.8, "capthick": 0.7},
                height=0.67,
            )
            for index, (bar, value, error, model) in enumerate(
                zip(bars, values, errors, KEY_MODELS)
            ):
                if not np.isfinite(value):
                    continue
                ax.text(
                    min(value + error + x_max * 0.018, x_max * 0.96),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}",
                    ha="left",
                    va="center",
                    fontsize=7.0,
                    fontweight="bold" if model in BIN_MODELS else "normal",
                )
            ax.set_xlim(0, x_max)
            ax.set_yticks(y)
            if col == 0:
                ax.set_yticklabels(
                    ["Hybrid-JK" if model == "Hybrid_JK" else model for model in KEY_MODELS]
                )
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            ax.invert_yaxis()
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("Score" if row == len(KEY_METRICS) - 1 else "")
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology], pad=5)
            if col == 0:
                ax.text(
                    -0.42,
                    0.5,
                    METRIC_LABEL[metric],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
            panel_letter(ax, letters)
            letters += 1

    fig.text(
        0.52,
        0.012,
        "Bars show mean +/- s.d. across five independently trained seeds; full-input configurations.",
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    add_draft_notice(fig, verified, reason)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.94, bottom=0.07, hspace=0.38, wspace=0.18)

    warnings = check_min_font(fig, MAIN)
    if warnings:
        raise RuntimeError("; ".join(warnings))
    suffix = "" if verified else "_DRAFT_UNVERIFIED"
    savefig(
        fig,
        output_dir / f"ablation_key_metrics_grid{suffix}",
        tier=MAIN,
        formats=("png", "svg", "tiff"),
    )

    table_columns = ["ontology", "model", "input"]
    for metric in KEY_METRICS:
        table_columns.extend(
            [f"{metric}_mean", f"{metric}_std", f"{metric}_count"]
        )
    summary[table_columns].to_csv(
        output_dir / f"ablation_key_metrics_grid_values{suffix}.csv", index=False
    )
    return summary


def blend_with_white(color: str, strength: float) -> tuple[float, float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color))
    alpha = float(np.clip(strength, 0.12, 0.82))
    mixed = (1 - alpha) * np.ones(3) + alpha * rgb
    return (*mixed, 1.0)


def plot_bin_grid(
    bins: pd.DataFrame,
    output_dir: Path,
    verified: bool,
    reason: str,
) -> pd.DataFrame:
    required = {
        "ontology",
        "model",
        "input_modality",
        "checkpoint",
        "bin_type",
        "bin",
        "examples",
        "Micro_Fmax",
    }
    require_columns(bins, required, Path("bin archive"))
    selected = bins[
        bins["model"].isin(BIN_MODELS)
        & bins["input_modality"].isin(BIN_INPUTS)
        & bins["bin_type"].isin(BIN_ORDER)
    ].copy()
    if "evaluation_split" in selected:
        selected = selected[selected["evaluation_split"].eq("test")].copy()

    selected["replicate"] = selected["checkpoint"].astype(str)
    summary = (
        selected.groupby(
            ["ontology", "bin_type", "bin", "model", "input_modality"],
            observed=True,
        )
        .agg(
            mean=("Micro_Fmax", "mean"),
            sd=("Micro_Fmax", "std"),
            seeds=("replicate", "nunique"),
            examples=("examples", "max"),
        )
        .reset_index()
    )
    observed_seed_counts = summary.loc[summary["mean"].notna(), "seeds"]
    if not observed_seed_counts.empty and not observed_seed_counts.eq(5).all():
        bad = summary.loc[summary["mean"].notna() & summary["seeds"].ne(5)]
        raise RuntimeError(f"Expected five bin-evaluation seeds:\n{bad.head(20)}")

    configurations = [
        (model, input_name) for model in BIN_MODELS for input_name in BIN_INPUTS
    ]
    global_max = float(summary["mean"].max(skipna=True))
    global_max = max(global_max, 1e-9)

    apply_style()
    fig, axes = plt.subplots(
        2,
        len(ONTOLOGY_ORDER),
        figsize=(BMC_WIDTH_IN, BINS_HEIGHT_IN),
        squeeze=False,
    )
    letters = 0
    bin_types = ("homology", "ic")

    for row, bin_type in enumerate(bin_types):
        ordered_bins = BIN_ORDER[bin_type]
        for col, ontology in enumerate(ONTOLOGY_ORDER):
            ax = axes[row, col]
            panel = summary[
                summary["ontology"].eq(ontology)
                & summary["bin_type"].eq(bin_type)
            ]
            means = np.full((len(configurations), len(ordered_bins)), np.nan)
            sds = np.full_like(means, np.nan)
            support = {}
            for y_index, (model, input_name) in enumerate(configurations):
                cells = panel[
                    panel["model"].eq(model)
                    & panel["input_modality"].eq(input_name)
                ].set_index("bin")
                for x_index, bin_name in enumerate(ordered_bins):
                    if bin_name not in cells.index:
                        continue
                    cell = cells.loc[bin_name]
                    if isinstance(cell, pd.DataFrame):
                        cell = cell.iloc[0]
                    means[y_index, x_index] = float(cell["mean"])
                    sds[y_index, x_index] = float(cell["sd"])
                    support[bin_name] = int(cell["examples"])

            rgba = np.ones((*means.shape, 4), dtype=float)
            for y_index, (model, _) in enumerate(configurations):
                for x_index in range(len(ordered_bins)):
                    value = means[y_index, x_index]
                    if np.isfinite(value):
                        strength = 0.18 + 0.64 * value / global_max
                        rgba[y_index, x_index] = blend_with_white(
                            MODEL_COLOR[model], strength
                        )
                    else:
                        rgba[y_index, x_index] = (0.94, 0.94, 0.94, 1.0)

            ax.imshow(rgba, aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(ordered_bins)))
            labels = []
            for bin_name in ordered_bins:
                label = BIN_LABEL[bin_name]
                if support.get(bin_name, 0) < 10:
                    label += "*"
                labels.append(label)
            ax.set_xticklabels(labels, rotation=32, ha="right", rotation_mode="anchor")
            ax.set_yticks(np.arange(len(configurations)))
            if col == 0:
                ax.set_yticklabels(
                    [
                        f"{'Hybrid-JK' if model == 'Hybrid_JK' else model} -- {INPUT_LABEL[input_name]}"
                        for model, input_name in configurations
                    ]
                )
                for tick, (model, _) in zip(ax.get_yticklabels(), configurations):
                    tick.set_color(MODEL_COLOR[model])
                    tick.set_fontweight("bold" if input_name == "full" else "normal")
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)

            ax.set_xticks(np.arange(-0.5, len(ordered_bins), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(configurations), 1), minor=True)
            ax.grid(which="minor", color="#ffffff", linewidth=1.1)
            ax.grid(which="major", visible=False)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color("#777777")

            for y_index in range(len(configurations)):
                for x_index in range(len(ordered_bins)):
                    value = means[y_index, x_index]
                    text = "--" if not np.isfinite(value) else f"{value:.3f}"
                    ax.text(
                        x_index,
                        y_index,
                        text,
                        ha="center",
                        va="center",
                        fontsize=5.4,
                        fontweight="bold" if configurations[y_index][1] == "full" else "normal",
                        color="#111111",
                    )
            if row == 0:
                ax.set_title(ONTOLOGY_SHORT[ontology], pad=5)
            if col == 0:
                row_label = "BLASTP identity bins" if bin_type == "homology" else "GO-term IC bins"
                ax.text(
                    -0.58,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
            panel_letter(ax, letters)
            letters += 1

    legend = [
        Patch(facecolor=MODEL_COLOR["Hybrid"], edgecolor="#555555", label="Hybrid"),
        Patch(facecolor=MODEL_COLOR["Hybrid_JK"], edgecolor="#555555", label="Hybrid-JK"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 0.985))
    fig.text(
        0.52,
        0.018,
        "Cells show mean Micro-F$_{max}$ across five seeds. * fewer than 10 proteins; "
        "-- metric undefined. Structure-only rows expose the principal input limitation.",
        ha="center",
        va="bottom",
        fontsize=6.0,
    )
    add_draft_notice(fig, verified, reason)
    fig.subplots_adjust(left=0.245, right=0.99, top=0.91, bottom=0.12, hspace=0.55, wspace=0.16)

    warnings = check_min_font(fig, SUPPLEMENTARY)
    if warnings:
        raise RuntimeError("; ".join(warnings))
    suffix = "" if verified else "_DRAFT_UNVERIFIED"
    savefig(
        fig,
        output_dir / f"ablation_bin_micro_fmax_grid{suffix}",
        tier=SUPPLEMENTARY,
        formats=("png", "svg", "tiff"),
    )
    summary.to_csv(
        output_dir / f"ablation_bin_micro_fmax_grid_values{suffix}.csv", index=False
    )
    return summary


def main() -> None:
    args = parse_args()
    manifest = args.manifest or args.ablation_csv.with_suffix(".manifest.json")
    needs_metrics = args.grid in {"both", "metrics"}
    if needs_metrics:
        verified, reason = validate_provenance(manifest, args.allow_unverified_auprc)
    else:
        # This grid uses only Micro-Fmax, whose definition was not affected by
        # the archived AUPRC estimator, so an existing bin CSV is sufficient.
        verified, reason = True, ""
    output_dir = args.output_dir
    if needs_metrics and not verified:
        output_dir = output_dir / "draft_unverified"
    output_dir.mkdir(parents=True, exist_ok=True)

    assert_palette_locked()
    if needs_metrics:
        archive = pd.read_csv(args.ablation_csv)
        plot_key_metrics(archive, output_dir, verified, reason)
    if args.grid in {"both", "bins"}:
        bins = pd.read_csv(args.bin_csv)
        plot_bin_grid(bins, output_dir, True, "")
    print(f"Wrote BMC ablation grids to {output_dir}")


if __name__ == "__main__":
    main()
