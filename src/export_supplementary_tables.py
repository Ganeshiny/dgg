#!/usr/bin/env python3
"""Export publication-ready supplementary tables for the ARC benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = PROJECT_DIR / "arc_benchmark" / "nominal_30_identity_80_coverage"

# numpy 2.0 renamed trapz -> trapezoid; ARC's environment still has numpy < 2.
# See the identical note in plot_stratified.py.
_trapezoid = getattr(np, "trapezoid", None)
if _trapezoid is None:  # numpy < 2.0
    _trapezoid = np.trapz
ONTOLOGY_LABEL = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}
# Mirrors plot_baselines_only.EXCLUDED_FROM_PLOTS, which is the canonical
# definition; tests assert the two stay in sync.
EXCLUDED_FROM_TABLES = {"deepgoplus", "deepgose"}

EXTERNAL_PRETRAINED = {
    "deepfri_sequence", "deepfri_structure", "dpfunc", "deepgoplus",
    "deepgose", "transfun", "eggnog_mapper", "hayai", "gomap",
    # Graph-based comparators run from their released checkpoints. Without
    # these, comparison_audit() falls through to "requires manual review",
    # which would be wrong provenance for a publication table.
    "heal", "gat_go", "deepgraphgo",
}
SPLIT_TRAINED = {
    "deepgreengo", "naive", "blast", "blast_max", "diamond",
    "diamond_max", "foldseek", "foldseek_max",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def require_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required supplementary source table is missing: {path}")
    return pd.read_csv(path)


def ic_weighted_aupr(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, ontology), frame in curves.groupby(["method", "ontology"]):
        precision = pd.to_numeric(frame["ic_weighted_precision"], errors="coerce").to_numpy()
        recall = pd.to_numeric(frame["ic_weighted_recall"], errors="coerce").to_numpy()
        valid = np.isfinite(precision) & np.isfinite(recall)
        if valid.sum() < 2:
            area = np.nan
        else:
            order = np.argsort(recall[valid])
            area = float(_trapezoid(precision[valid][order], recall[valid][order]))
        rows.append({
            "method": method,
            "ontology": ontology,
            "ontology_short": ONTOLOGY_LABEL.get(str(ontology), str(ontology)),
            "ic_weighted_aupr": area,
            "threshold_points": int(valid.sum()),
        })
    return pd.DataFrame(rows).sort_values(["method", "ontology"])


def comparison_audit(methods: list[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        if method in SPLIT_TRAINED:
            training_relation = "trained/derived only from the locked project split"
            overlap_status = "not applicable beyond the locked split audit"
            bin_interpretation = "direct similarity to the method's available training/reference set"
        elif method == "interproscan":
            training_relation = "rule/domain-database annotation system; not retrained here"
            overlap_status = "external database coverage not audited as a training corpus"
            bin_interpretation = "descriptive only; bins use the DeepGreenGO training proteins"
        elif method in EXTERNAL_PRETRAINED:
            training_relation = "externally pretrained/released model or annotation pipeline"
            overlap_status = "not audited against the method's original training corpus"
            bin_interpretation = "descriptive only; cannot establish external-training homology leakage"
        else:
            training_relation = "comparison provenance requires manual review"
            overlap_status = "not audited"
            bin_interpretation = "descriptive only"
        rows.append({
            "method": method,
            "training_relation": training_relation,
            "external_training_overlap_status": overlap_status,
            "project_homology_bin_interpretation": bin_interpretation,
        })
    return pd.DataFrame(rows)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")
    print(f"[supplement] {path.name}: {len(frame)} rows")


def main() -> None:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    results = workspace / "results"
    output = (args.output_dir or workspace / "plots" / "supplementary_tables").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    metrics = require_csv(results / "benchmark_metrics.csv")
    homology = require_csv(results / "stratified_homology.csv")
    homology_aupr = require_csv(results / "stratified_homology_aupr.csv")
    ic = require_csv(results / "stratified_ic.csv")
    depth = require_csv(results / "stratified_depth.csv")
    curves = require_csv(results / "ic_weighted_pr.csv")

    # Keep the published tables consistent with the published figures. The
    # complete, unfiltered record stays in results/; only these presentation
    # tables drop the withheld comparators, which previously reappeared here
    # after being removed from every figure.
    def drop_excluded(frame: pd.DataFrame) -> pd.DataFrame:
        if "method" not in frame.columns:
            return frame
        return frame[~frame["method"].astype(str).isin(EXCLUDED_FROM_TABLES)].copy()

    metrics = drop_excluded(metrics)
    homology = drop_excluded(homology)
    homology_aupr = drop_excluded(homology_aupr)
    ic = drop_excluded(ic)
    depth = drop_excluded(depth)
    curves = drop_excluded(curves)
    metrics = metrics.copy()
    metrics["ontology_short"] = metrics["ontology"].map(ONTOLOGY_LABEL)

    performance_columns = [
        "method", "ontology", "ontology_short", "test_proteins", "test_terms",
        "cafa_fmax", "cafa_fmax_ci_low", "cafa_fmax_ci_high", "cafa_smin",
        "cafa_smin_ci_low", "cafa_smin_ci_high", "micro_aupr", "macro_aupr",
        "micro_auroc", "macro_auroc", "brier_score", "expected_calibration_error",
    ]
    performance = metrics[[c for c in performance_columns if c in metrics]].copy()
    write_table(performance, output / "supp_table_s1_overall_performance.csv")
    write_table(ic_weighted_aupr(curves), output / "supp_table_s2_ic_weighted_aupr.csv")

    coverage_columns = [
        "method", "ontology", "ontology_short", "protein_coverage_any_score",
        "predicted_term_coverage", "validation_threshold",
        "test_coverage_at_validation_threshold",
        "mean_terms_per_protein_at_validation_threshold",
    ]
    coverage = metrics[[c for c in coverage_columns if c in metrics]].copy()
    write_table(coverage, output / "supp_table_s3_prediction_coverage.csv")
    keys = ["ontology", "method", "bin", "bin_unit", "bin_n", "below_min_bin_size"]
    fmax = homology.rename(columns={
        "value": "fmax", "seed_mean": "fmax_seed_mean",
        "seed_sd": "fmax_seed_sd", "seed_n": "fmax_seed_n",
    })
    aupr = homology_aupr.rename(columns={
        "value": "micro_aupr", "seed_mean": "micro_aupr_seed_mean",
        "seed_sd": "micro_aupr_seed_sd", "seed_n": "micro_aupr_seed_n",
    })
    homology_table = fmax.merge(
        aupr, on=keys, how="outer", validate="one_to_one"
    )
    write_table(homology_table, output / "supp_table_s4_homology_fmax_aupr.csv")
    write_table(ic, output / "supp_table_s5_term_ic_auprc.csv")
    write_table(depth, output / "supp_table_s6_go_depth_auprc.csv")

    paired_path = results / "paired_differences_vs_deepgreengo.csv"
    if paired_path.is_file():
        write_table(pd.read_csv(paired_path), output / "supp_table_s7_paired_differences.csv")

    methods = sorted(set(metrics["method"].astype(str)))
    write_table(comparison_audit(methods), output / "supp_table_s8_comparison_audit.csv")
    tables = sorted(path.name for path in output.glob("supp_table_*.csv"))
    manifest = {
        "workspace": str(workspace),
        "reference_design": "DPFunc supplementary Figures S1-S3 and Tables S1-S3",
        "method_count": len(methods),
        "methods": methods,
        "caveat": (
            "Homology bins use similarity to the locked DeepGreenGO training set. "
            "They do not measure overlap with external methods' original training corpora."
        ),
        "tables": tables,
    }
    (output / "supplementary_table_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "README.txt").write_text(
        "Supplementary benchmark tables\n\n"
        "S1: overall accuracy and calibration; S2: IC-weighted AUPR; "
        "S3: prediction coverage; S4-S6: values underlying stratified figures; "
        "S7: paired differences when available; S8: comparison provenance and "
        "interpretation limits. Blank values are unavailable/undefined, never zero-filled.\n",
        encoding="utf-8",
    )
    print(f"[supplement] output={output}")


if __name__ == "__main__":
    main()
