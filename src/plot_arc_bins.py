#!/usr/bin/env python3
"""Plot homology/IC-bin metrics with explicit support and uncertainty audits.

The script refuses to treat missing bins as absent data. It reconstructs raw
test-bin support from the split labels, reports one-class AUROC cells, masks
archived evaluator zeros that cannot be interpreted, and plots mean +/- SD
across the five checkpoint seeds without connecting unsupported points.
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_style import (
    ERROR_CAPTION,
    SUPPLEMENTARY,
    assert_palette_locked,
    provenance,
    report_colorblind_audit,
    BIN_AXIS_LABEL,
    BIN_ORDER,
    DOUBLE_COLUMN_IN,
    ERROR_KIND_LABEL,
    METRIC_HIGHER_IS_BETTER,
    METRIC_LABEL,
    METRIC_ORDER,
    MODEL_COLOR,
    MODEL_MARKER,
    MODEL_ORDER,
    ONTOLOGY_ORDER,
    ONTOLOGY_SHORT,
    VARIANT_LABEL,
    VARIANT_ORDER,
    annotate_insufficient_data,
    apply_style,
    colorblind_audit,
    label_panel,
    mean_and_error,
    savefig,
)

DEFAULT_DATA_ROOT = Path("preprocessing/data_arc_rebuild_2026_07_14/datasets")
DEFAULT_HOMOLOGY = Path("preprocessing/data_arc_rebuild_2026_07_14/pdb_splits/threshold_30/blast_te_vs_tr.tsv")
DEFAULT_VALIDATION_HOMOLOGY = Path("preprocessing/data_arc_rebuild_2026_07_14/pdb_splits/threshold_30/blast_va_vs_tr.tsv")


def _records(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, list):
        return obj
    if hasattr(obj, "protein_ids") and hasattr(obj, "labels") and hasattr(obj, "terms"):
        return [
            {"id": pid, "labels": [term for term, value in zip(obj.terms, row) if value > 0]}
            for pid, row in zip(obj.protein_ids, obj.labels)
        ]
    raise TypeError(f"Unsupported dataset pickle: {path}")


def _threshold_from_label(frame: pd.DataFrame) -> int:
    for value in frame.get("split_label", pd.Series(dtype=str)).dropna().astype(str):
        match = re.search(r"(?:threshold_|nominal_)(\d+)", value)
        if match:
            return int(match.group(1))
    return 30


def _homology_groups(path: Path, ids: list[str]) -> dict[str, str]:
    maximum = {protein: 0.0 for protein in ids}
    if path.is_file():
        with path.open() as handle:
            for line in handle:
                fields = line.rstrip().split("\t")
                if len(fields) >= 3 and fields[0] in maximum:
                    try:
                        maximum[fields[0]] = max(maximum[fields[0]], float(fields[2]))
                    except ValueError:
                        continue
    groups = {}
    for protein, value in maximum.items():
        groups[protein] = (
            "no_hit" if value <= 0 else "<30%" if value < 30 else
            "30-40%" if value < 40 else "40-60%" if value < 60 else ">=60%"
        )
    return groups


def _ic_groups(test: list[dict], train: list[dict]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for row in train:
        for term in set(row.get("labels", [])):
            counts[term] = counts.get(term, 0) + 1
    n_train = max(len(train), 1)
    ic = {term: -np.log2(count / n_train) for term, count in counts.items() if count > 0}
    result = {}
    for row in test:
        values = [ic[term] for term in set(row.get("labels", [])) if term in ic]
        value = max(values) if values else np.nan
        result[row["id"]] = (
            "no_positive_terms" if not np.isfinite(value) else "<2_bits" if value < 2 else
            "2-4_bits" if value < 4 else "4-6_bits" if value < 6 else ">=6_bits"
        )
    return result


def build_integrity_audit(dataset_root: Path, homology_path: Path, threshold: int,
                          evaluation_splits: tuple[str, ...] = ("test",),
                          validation_homology_path: Path | None = None) -> pd.DataFrame:
    rows = []
    homology_paths = {
        "test": Path(homology_path),
        "valid": Path(validation_homology_path or DEFAULT_VALIDATION_HOMOLOGY),
    }
    for evaluation_split in evaluation_splits:
        for ontology in ONTOLOGY_ORDER:
            train_path = dataset_root / f"threshold_{threshold}" / f"{ontology}_train.pkl"
            query_path = dataset_root / f"threshold_{threshold}" / f"{ontology}_{evaluation_split}.pkl"
            if not query_path.exists():
                raise FileNotFoundError(
                    f"Missing {evaluation_split} labels for {ontology}: {query_path}"
                )
            train = _records(train_path) if train_path.exists() else []
            query = _records(query_path)
            term_vocab = sorted(set(
                term for row in train + query for term in row.get("labels", [])
            ))
            groups_by_type = {
                "homology": _homology_groups(
                    homology_paths[evaluation_split], [row["id"] for row in query]
                ),
                "ic": _ic_groups(query, train),
            }
            for bin_type, group_map in groups_by_type.items():
                for group in BIN_ORDER[bin_type]:
                    selected = [row for row in query if group_map.get(row["id"]) == group]
                    matrix = np.asarray([
                        [term in set(row.get("labels", [])) for term in term_vocab]
                        for row in selected
                    ], dtype=int)
                    if not selected:
                        positive_terms = valid_terms = all_positive_terms = positive_assignments = 0
                    else:
                        counts = matrix.sum(axis=0)
                        positive_terms = int(np.sum(counts > 0))
                        valid_terms = int(np.sum((counts > 0) & (counts < len(selected))))
                        all_positive_terms = int(np.sum(counts == len(selected)))
                        positive_assignments = int(matrix.sum())
                    rows.append({
                        "evaluation_split": evaluation_split,
                        "ontology": ontology,
                        "bin_type": bin_type,
                        "bin": group,
                        "examples": len(selected),
                        "positive_examples": int(sum(bool(row.get("labels")) for row in selected)),
                        "positive_term_assignments": positive_assignments,
                        "positive_terms": positive_terms,
                        "valid_auroc_terms": valid_terms,
                        "all_positive_terms": all_positive_terms,
                        "zero_positive_terms": max(len(term_vocab) - positive_terms, 0),
                        "support_status": (
                            "empty" if not selected else "low_support" if len(selected) < 10 else "ok"
                        ),
                    })
    return pd.DataFrame(rows)


def _audit_status(row: pd.Series) -> str:
    if int(row.get("examples", 0)) == 0:
        return "no data"
    if int(row.get("positive_terms", 0)) == 0:
        return "no positive terms"
    if int(row.get("valid_auroc_terms", 0)) == 0:
        return "AUROC undefined: one-class terms"
    if int(row.get("examples", 0)) < 10:
        return "low support"
    return "ok"


def merge_and_validate(frame: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    audit = audit.copy()
    if "evaluation_split" not in frame:
        frame["evaluation_split"] = "test"
    if "evaluation_split" not in audit:
        audit["evaluation_split"] = "test"
    keys = ["evaluation_split", "ontology", "bin_type", "bin"]
    # bin_metrics.csv may itself be a previously audited archive. Remove only
    # derived audit columns before attaching the freshly reconstructed audit;
    # otherwise pandas creates duplicate examples_audit columns whose Series
    # cannot be used as a scalar validity check.
    derived = [
        column for column in audit.columns
        if column not in keys and column != "examples" and column in frame.columns
    ]
    derived.extend(column for column in ("examples_audit", "audit_status")
                   if column in frame.columns)
    frame = frame.drop(columns=sorted(set(derived)))
    merged = frame.merge(audit, on=keys, how="left", suffixes=("", "_audit"))
    if merged["examples_audit"].isna().any():
        missing = merged.loc[merged["examples_audit"].isna(), keys].drop_duplicates()
        raise RuntimeError(f"Missing raw-N audit rows for bin cells:\n{missing.to_string(index=False)}")
    merged["audit_status"] = merged.apply(_audit_status, axis=1)
    # The old evaluator returned 0 for a whole macro-AUROC bin when any term
    # was one-class. That value is an evaluator artifact, not a model score.
    bad_macro_auroc = (
        (merged["Macro_AUROC"].abs() <= 1e-12) &
        (merged["valid_auroc_terms"] > 0) &
        (merged["positive_terms"] > 0)
    )
    merged.loc[bad_macro_auroc, "Macro_AUROC"] = np.nan
    merged.loc[bad_macro_auroc, "audit_status"] = "archived AUROC evaluator bug; rerun required"
    merged.loc[merged["valid_auroc_terms"] == 0, "Macro_AUROC"] = np.nan
    merged.loc[merged["positive_terms"] == 0, ["Macro_AUPRC", "Macro_AUROC"]] = np.nan
    return merged


def _size_for_n(n: float, max_n: float) -> float:
    if not np.isfinite(n) or n <= 0:
        return 0.0
    return 22.0 + 58.0 * np.sqrt(n / max(max_n, 1.0))


def _metric_limits(frame: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    if metric == "Smin":
        return 0.0, max(1.0, hi * 1.10)
    pad = max((hi - lo) * .08, .02)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _marker_legend(max_n: float) -> list[Line2D]:
    refs = sorted(set([5.0, 10.0, max(10.0, min(max_n, 50.0))]))
    return [Line2D([0], [0], linestyle="", marker="o", color="#555555",
                   markersize=np.sqrt(_size_for_n(n, max_n)), label=f"n={int(n)}")
            for n in refs]


def load_audited_bins(bin_csv: Path, dataset_root: Path,
                      homology_path: Path | None = None,
                      validation_homology_path: Path | None = None,
                      logs_hint=None) -> pd.DataFrame | None:
    """Read bin metrics and attach split-specific raw-support audits."""
    bin_csv = Path(bin_csv)
    if not bin_csv.exists():
        print(f"NOTE: bin metrics not found at {bin_csv}; skipping bin figures.")
        return None
    frame = pd.read_csv(bin_csv)
    required = {"ontology", "model", "input_modality", "bin_type", "bin", "examples"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing required bin columns: {sorted(missing)}")
    if "evaluation_split" not in frame:
        frame["evaluation_split"] = "test"
    threshold = _threshold_from_label(frame)
    hom = Path(homology_path) if homology_path else DEFAULT_HOMOLOGY
    valid_hom = Path(validation_homology_path) if validation_homology_path else DEFAULT_VALIDATION_HOMOLOGY
    splits = tuple(frame["evaluation_split"].dropna().astype(str).unique())
    audit = build_integrity_audit(
        Path(dataset_root), Path(hom).resolve(), threshold, splits, Path(valid_hom).resolve()
    )
    merged = merge_and_validate(frame, audit)
    for bin_type, order in BIN_ORDER.items():
        mask = merged.bin_type == bin_type
        merged.loc[mask, "bin"] = pd.Categorical(
            merged.loc[mask, "bin"], categories=order, ordered=True)
    return merged


def export_bin_tables(frame: pd.DataFrame, out: Path) -> None:
    """Write raw support and tidy per-metric summaries for supplementary use."""
    table_dir = out / "supplementary_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_dir / "supp_table_bin_metrics_audited_raw.csv", index=False)
    metrics = [metric for metric in METRIC_ORDER if metric in frame]
    support_columns = [
        "evaluation_split", "ontology", "bin_type", "bin", "examples_audit",
        "positive_examples", "positive_term_assignments", "positive_terms",
        "valid_auroc_terms", "all_positive_terms", "zero_positive_terms",
        "support_status", "audit_status",
    ]
    support = frame[[c for c in support_columns if c in frame]].drop_duplicates()
    support.to_csv(table_dir / "supp_table_bin_support.csv", index=False)

    id_columns = [
        "evaluation_split", "ontology", "model", "input_modality", "bin_type", "bin",
        "examples_audit", "audit_status",
    ]
    long = frame.melt(
        id_vars=[c for c in id_columns if c in frame], value_vars=metrics,
        var_name="metric", value_name="value",
    )
    group_columns = [
        "evaluation_split", "ontology", "model", "input_modality", "bin_type", "bin",
        "metric", "examples_audit", "audit_status",
    ]
    summary = long.groupby(group_columns, dropna=False, observed=True)["value"].agg(
        seed_replicates="count", mean="mean", sd="std", median="median", minimum="min", maximum="max"
    ).reset_index()
    summary.to_csv(table_dir / "supp_table_bin_metric_summary.csv", index=False)
    summary[summary.metric == "Smin"].to_csv(
        table_dir / "supp_table_bin_smin.csv", index=False
    )


def plot_bin_grid(frame: pd.DataFrame, out: Path, bin_type: str, order: list[str],
                  metric: str, min_n: int, err_kind: str,
                  tier: str = SUPPLEMENTARY) -> bool:
    """Grouped dynamite panels with exact sample counts in x-axis labels.

    Models are grouped, not stacked: stacking would incorrectly imply that
    performance scores are additive. Empty bins and empty panels are omitted.
    Smin is deliberately table-only because the binwise values are dominated
    by scale/support and do not form an interpretable visual comparison.
    """
    if metric == "Smin":
        return False
    frame = frame.copy()
    if "evaluation_split" not in frame:
        frame["evaluation_split"] = "test"
    split_values = frame["evaluation_split"].dropna().astype(str).unique()
    if len(split_values) != 1:
        raise ValueError("plot_bin_grid expects exactly one evaluation split")
    evaluation_split = split_values[0]
    support_column = "examples_audit" if "examples_audit" in frame else "examples"
    panels = []
    for variant in VARIANT_ORDER:
        for ontology in ONTOLOGY_ORDER:
            sub = frame[(frame.ontology == ontology) & (frame.input_modality == variant)]
            usable_bins = []
            for label in order:
                cell = sub[sub["bin"].astype(str) == label]
                has_support = pd.to_numeric(cell[support_column], errors="coerce").fillna(0).gt(0).any()
                has_value = np.isfinite(pd.to_numeric(cell[metric], errors="coerce")).any()
                if has_support and has_value:
                    usable_bins.append(label)
            if usable_bins:
                panels.append((variant, ontology, sub, usable_bins))
    if not panels:
        return False

    ncols = min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(DOUBLE_COLUMN_IN, 2.45 * nrows),
        sharey=True, squeeze=False,
    )
    axes_flat = axes.ravel()
    ymin, ymax = _metric_limits(frame, metric)
    width = .78 / len(MODEL_ORDER)
    for panel_index, (variant, ontology, sub, usable_bins) in enumerate(panels):
        ax = axes_flat[panel_index]
        x = np.arange(len(usable_bins), dtype=float)
        counts = []
        for label in usable_bins:
            cell = sub[sub["bin"].astype(str) == label]
            values = pd.to_numeric(cell[support_column], errors="coerce").dropna()
            counts.append(int(values.iloc[0]) if len(values) else 0)
        for model_index, model in enumerate(MODEL_ORDER):
            group = sub[sub.model == model]
            stats = group.groupby("bin", observed=True)[metric].agg(["mean", "std", "count"])
            means = np.asarray([
                float(stats.loc[label, "mean"]) if label in stats.index else np.nan
                for label in usable_bins
            ])
            errors = np.asarray([
                float(stats.loc[label, "std"]) if label in stats.index and stats.loc[label, "count"] > 1 else 0.0
                for label in usable_bins
            ])
            valid = np.isfinite(means)
            positions = x + (model_index - (len(MODEL_ORDER) - 1) / 2) * width
            ax.bar(
                positions[valid], means[valid], width=width * .92,
                yerr=errors[valid], color=MODEL_COLOR[model], edgecolor="#111111",
                linewidth=.35, error_kw=dict(elinewidth=.65, capsize=1.5, ecolor="#111111"),
                label=model,
            )
            for pos, mean, count in zip(positions[valid], means[valid], np.asarray(counts)[valid]):
                if count < min_n:
                    ax.annotate("*", (pos, mean), xytext=(0, 3), textcoords="offset points",
                                ha="center", fontsize=6, fontweight="bold")
        labels = [f"{label.replace('_', ' ')}\n[n={count}]" for label, count in zip(usable_bins, counts)]
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-.55, len(usable_bins) - .45)
        ax.set_title(
            f"{ONTOLOGY_SHORT[ontology]} — {VARIANT_LABEL[variant]} ({evaluation_split})",
            fontsize=7,
        )
        label_panel(ax, chr(97 + panel_index))
        ax.set_ylabel(METRIC_LABEL[metric])
    for ax in axes_flat[len(panels):]:
        ax.remove()
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLOR[m], edgecolor="#111111", label=m)
        for m in MODEL_ORDER
    ]
    fig.legend(handles=handles, title="Model", loc="upper center", ncol=len(MODEL_ORDER),
               bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=6.2, title_fontsize=6.5)
    direction = "higher is better" if METRIC_HIGHER_IS_BETTER[metric] else "lower is better"
    fig.text(
        .5, -.015,
        f"Grouped bars show the mean with {ERROR_KIND_LABEL[err_kind]} across five seeds. "
        f"Exact {evaluation_split} examples are printed in brackets; '*' marks n<{min_n}. "
        f"Bins/panels without finite data are omitted; {direction}. Models are not stacked because "
        "performance scores are not additive.",
        ha="center", fontsize=5.6, wrap=True,
    )
    savefig(fig, out / f"{evaluation_split}_{bin_type}_{metric.lower()}.png", tier)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("plots/arc_tuning_cafa/bin_evaluation"))
    ap.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--homology-tsv", type=Path, default=DEFAULT_HOMOLOGY)
    ap.add_argument("--validation-homology-tsv", type=Path, default=DEFAULT_VALIDATION_HOMOLOGY)
    ap.add_argument("--metrics", nargs="+", default=METRIC_ORDER, choices=METRIC_ORDER)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--err", choices=["sd", "sem", "ci95"], default="sd")
    args = ap.parse_args()
    apply_style()
    print("Palette fingerprint:", assert_palette_locked())
    report_colorblind_audit()
    merged = load_audited_bins(
        args.bin_csv.resolve(), args.dataset_root.resolve(), args.homology_tsv.resolve(),
        args.validation_homology_tsv.resolve(),
    )
    if merged is None:
        return
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    export_bin_tables(merged, out)
    support_columns = [
        "evaluation_split", "ontology", "bin_type", "bin", "examples_audit",
        "positive_examples", "positive_term_assignments", "positive_terms",
        "valid_auroc_terms", "audit_status",
    ]
    merged[[c for c in support_columns if c in merged]].drop_duplicates().to_csv(
        out / "bin_integrity_audit.csv", index=False
    )
    for evaluation_split in merged["evaluation_split"].dropna().astype(str).unique():
        split_frame = merged[merged.evaluation_split == evaluation_split]
        for bin_type in [value for value in BIN_ORDER if value in set(split_frame.bin_type)]:
            subset = split_frame[split_frame.bin_type == bin_type].copy()
            for metric in args.metrics:
                if metric in subset and metric != "Smin":
                    plot_bin_grid(subset, out, bin_type, BIN_ORDER[bin_type], metric,
                                  args.min_n, args.err)
    print(f"Wrote audited bin tables and non-empty grouped-bar plots to {out}")


if __name__ == "__main__":
    main()
