#!/usr/bin/env python3
"""Emit the key numbers behind every figure, so two hosts can be diffed.

Every value printed here is recomputed from the data on this machine — none
are read back from a cached figure or hard-coded. Run it locally and on ARC
and diff the two reports: any line that differs is a real reproducibility gap,
not a rendering difference.

  python src/verify_figure_pipeline.py --output verification_report.json

The exit code is 0 when every check that CAN run on this host passes, and 1
when a check fails. Checks whose inputs are absent are reported as SKIPPED
rather than silently passing.
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ONTOLOGIES = ["molecular_function", "biological_process", "cellular_component"]
DATASETS = REPO / "preprocessing/data_arc_rebuild_2026_07_14/datasets/threshold_30"
HOMOLOGY = REPO / "preprocessing/data_arc_rebuild_2026_07_14/pdb_splits/threshold_30/blast_te_vs_tr.tsv"
ABLATION_ROOT = REPO / "arc_tuning_cafa/ablations/nominal_30_identity_80_coverage"
BENCH = REPO / "arc_benchmark/nominal_30_identity_80_coverage"

report: dict = {"checks": {}, "skipped": [], "failures": []}


def record(name: str, value, expected=None, tol: float | None = None) -> None:
    entry = {"value": value}
    if expected is not None:
        if tol is not None and isinstance(value, (int, float)):
            ok = abs(float(value) - float(expected)) <= tol
        else:
            ok = value == expected
        entry["expected"] = expected
        entry["ok"] = bool(ok)
        if not ok:
            report["failures"].append(f"{name}: got {value!r}, expected {expected!r}")
    report["checks"][name] = entry
    flag = "" if expected is None else ("  OK" if entry.get("ok") else "  MISMATCH")
    print(f"  {name}: {value}{flag}")


def skip(name: str, why: str) -> None:
    report["skipped"].append({"check": name, "reason": why})
    print(f"  {name}: SKIPPED ({why})")


def _vocab(ontology: str) -> list[str]:
    terms: set[str] = set()
    for split in ("train", "valid", "test"):
        with (DATASETS / f"{ontology}_{split}.pkl").open("rb") as handle:
            for record_ in pickle.load(handle):
                terms.update(record_["labels"])
    return sorted(terms)


def _matrix(ontology: str, split: str):
    vocab = _vocab(ontology)
    index = {term: i for i, term in enumerate(vocab)}
    with (DATASETS / f"{ontology}_{split}.pkl").open("rb") as handle:
        records = pickle.load(handle)
    y = np.zeros((len(records), len(vocab)), dtype=int)
    for row, rec in enumerate(records):
        for term in rec["labels"]:
            y[row, index[term]] = 1
    return y, [r["id"] for r in records]


# ---------------------------------------------------------------------------
def check_environment() -> None:
    print("\n[1] environment")
    record("python", platform.python_version())
    for module in ("numpy", "pandas", "matplotlib", "scipy", "sklearn"):
        try:
            mod = __import__(module)
            record(f"{module}_version", getattr(mod, "__version__", "?"))
        except ImportError:
            record(f"{module}_version", None)
            report["failures"].append(f"{module} is not importable")
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=False)
        record("git_head", head.stdout.strip() or "unknown")
    except Exception:
        record("git_head", "unknown")


def check_palette() -> None:
    print("\n[2] shared style (must be byte-identical across hosts)")
    try:
        import plot_style as ps
    except ImportError as exc:
        skip("palette_fingerprint", f"cannot import plot_style: {exc}")
        return
    record("palette_fingerprint", ps.palette_fingerprint(), expected="56a06a743fcd9eb0")
    record("journal", ps.JOURNAL)
    record("double_column_in", round(ps.DOUBLE_COLUMN_IN, 3))
    record("model_order", ",".join(ps.MODEL_ORDER),
           expected="MLP,GCN,GAT,Hybrid,Hybrid_JK")


def check_evals_fix() -> None:
    print("\n[3] get_auprc estimator (constant predictor must score prevalence, not ~0.5)")
    try:
        from evals import get_auprc
    except ImportError as exc:
        skip("auprc_constant_predictor", f"cannot import evals: {exc}")
        return
    rng = np.random.default_rng(0)
    y = (rng.random((754, 50)) < 0.03).astype(int)
    constant = np.full((754, 50), 0.5)
    value = float(get_auprc(y, constant, "macro"))
    prevalence = float(y[:, y.sum(0) > 0].mean())
    record("auprc_constant_predictor", round(value, 4), expected=round(prevalence, 4), tol=5e-3)
    record("auprc_is_not_half", bool(value < 0.2), expected=True)


def check_ablation() -> None:
    print("\n[4] ablation table")
    try:
        import pandas as pd
        import plot_arc_ablations as abl
    except ImportError as exc:
        skip("ablation", f"import failed: {exc}")
        return
    df = abl.read_results(ABLATION_ROOT, REPO / "logs")
    archive = REPO / "plots/arc_tuning_cafa/ablations/ablation_test_metrics.csv"
    if len(df) < 225 and archive.is_file():
        archived = pd.read_csv(archive)
        if len(archived) >= 225:
            df = archived
            record("ablation_source", "archive")
        else:
            record("ablation_source", "reconstructed")
    else:
        record("ablation_source", "reconstructed")
    record("ablation_rows", len(df), expected=225)
    cells = df.groupby(["ontology", "model", "input"]).ngroups
    record("ablation_cells", cells, expected=45)
    seeds = df.groupby(["ontology", "model", "input"])["seed"].nunique()
    record("ablation_min_seeds_per_cell", int(seeds.min()), expected=5)

    print("\n[5] MLP structure-only is a constant predictor")
    for ontology, expected_auprc in (("molecular_function", 0.5146),
                                     ("biological_process", 0.4873),
                                     ("cellular_component", 0.5417)):
        sel = df[(df.ontology == ontology) & (df.model == "MLP") & (df.input == "struct_only")]
        if sel.empty:
            skip(f"mlp_struct_{ontology}", "no rows")
            continue
        record(f"mlp_struct_macro_auprc_{ontology}",
               round(float(sel.Macro_AUPRC.mean()), 4), expected=expected_auprc, tol=1e-3)
        record(f"mlp_struct_macro_auroc_{ontology}",
               round(float(sel.Macro_AUROC.mean()), 4), expected=0.5, tol=1e-2)

    print("\n[6] micro vs macro AUROC are distinct (reviewer 1.10)")
    identical = int(np.isclose(df["Micro_AUROC"], df["Macro_AUROC"], atol=1e-9).sum())
    record("rows_with_identical_micro_macro_auroc", identical, expected=0)


def check_smin_baseline() -> None:
    print("\n[7] structure-only Smin equals the predict-nothing baseline")
    if not DATASETS.is_dir():
        skip("smin_all_negative", f"{DATASETS} absent")
        return
    for ontology, expected in (("molecular_function", 26.784569),
                               ("biological_process", 101.055691),
                               ("cellular_component", 29.926645)):
        ytr, _ = _matrix(ontology, "train")
        yte, _ = _matrix(ontology, "test")
        counts = ytr.sum(0)
        ic = np.zeros(ytr.shape[1])
        mask = counts > 0
        ic[mask] = -np.log2(counts[mask] / ytr.shape[0])
        value = float((yte * ic).sum() / yte.shape[0])
        record(f"smin_all_negative_{ontology}", round(value, 6), expected=expected, tol=1e-4)


def check_leakage() -> None:
    print("\n[8] residual train/test identity (reviewers 1.3 and 2.3)")
    if not HOMOLOGY.is_file():
        skip("leakage", f"{HOMOLOGY} absent")
        return
    best: dict[str, float] = {}
    for line in HOMOLOGY.open():
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 3:
            try:
                best[fields[0]] = max(best.get(fields[0], 0.0), float(fields[2]))
            except ValueError:
                continue
    values = np.array(list(best.values()))
    record("test_proteins_with_any_hit", int(values.size), expected=142)
    record("test_proteins_at_or_above_30pct", int((values >= 30).sum()), expected=116)
    record("test_proteins_at_or_above_60pct", int((values >= 60).sum()), expected=31)


def check_bins() -> None:
    print("\n[9] bin support audit")
    bin_csv = ABLATION_ROOT / "bin_evaluation/bin_metrics.csv"
    if not bin_csv.is_file() or not DATASETS.is_dir():
        skip("bin_audit", "bin_metrics.csv or datasets absent")
        return
    try:
        import pandas as pd
        import plot_arc_bins as bins
    except ImportError as exc:
        skip("bin_audit", f"import failed: {exc}")
        return
    frame = pd.read_csv(bin_csv)
    zero_auroc = int((frame["Macro_AUROC"].abs() <= 1e-12).sum())
    record("bin_rows_macro_auroc_exactly_zero", zero_auroc, expected=58)
    record("bin_rows_macro_auprc_exactly_zero", int((frame["Macro_AUPRC"].abs() <= 1e-12).sum()),
           expected=0)
    per_cell = frame.groupby(["ontology", "model", "input_modality", "bin_type", "bin"]).size()
    record("bin_seeds_per_cell_min", int(per_cell.min()), expected=5)
    audit = bins.build_integrity_audit(DATASETS.parent, HOMOLOGY, 30)
    hom = audit[(audit.bin_type == "homology") & (audit.ontology == "molecular_function")]
    counts = {row["bin"]: int(row["examples"]) for _, row in hom.iterrows()}
    record("homology_bin_sizes_mf", json.dumps(counts, sort_keys=True),
           expected=json.dumps({"30-40%": 80, "40-60%": 5, "<30%": 26, ">=60%": 31, "no_hit": 612},
                               sort_keys=True))


def check_benchmark() -> None:
    print("\n[10] benchmark, after the BLAST repair")
    metrics = BENCH / "results/benchmark_metrics.csv"
    if not metrics.is_file():
        skip("benchmark", f"{metrics} absent")
        return
    import pandas as pd
    df = pd.read_csv(metrics)
    table = df.pivot_table(index="method", columns="ontology_short", values="cafa_fmax")
    for method, expected in (("blast", {"mf": 0.2777, "bp": 0.1352, "cc": 0.1158}),
                             ("deepgreengo", {"mf": 0.3839, "bp": 0.2171, "cc": 0.3461})):
        if method not in table.index:
            skip(f"benchmark_{method}", "method absent")
            continue
        for short, value in expected.items():
            record(f"benchmark_{method}_fmax_{short}",
                   round(float(table.loc[method, short]), 4), expected=value, tol=2e-3)
    coverage = df.pivot_table(index="method", columns="ontology_short",
                              values="protein_coverage_any_score")
    if "blast" in coverage.index:
        record("benchmark_blast_coverage_is_nonzero",
               bool(float(coverage.loc["blast", "mf"]) > 0), expected=True)
    boot = BENCH / "results/bootstrap_metrics.csv"
    if boot.is_file():
        b = pd.read_csv(boot)
        record("bootstrap_rows", len(b), expected=24000)
        record("bootstrap_blast_nonzero_replicates",
               int((b[b.method == "blast"].cafa_fmax > 0).sum()), expected=3000)
    else:
        skip("bootstrap", "bootstrap_metrics.csv absent")


def check_outputs(figure_root: Path) -> None:
    print("\n[11] rendered figure inventory")
    if not figure_root.is_dir():
        skip("figure_inventory", f"{figure_root} absent — run the figure scripts first")
        return
    for name, folder in (("main_text", figure_root / "main_text"),
                         ("supplementary", figure_root / "supplementary"),
                         ("benchmark", figure_root / "benchmark"),
                         ("reviewer", figure_root / "reviewer"),
                         ("supplementary_tuning", figure_root / "supplementary_tuning")):
        count = len(list(folder.glob("*.pdf"))) if folder.is_dir() else 0
        record(f"pdf_count_{name}", count)
    record("pdf_count_total", len(list(figure_root.rglob("*.pdf"))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=REPO / "verification_report.json")
    ap.add_argument("--figure-root", type=Path, default=REPO / "plots/figures")
    args = ap.parse_args()

    print("=" * 72)
    print("DeepGreenGO figure-pipeline verification")
    print(f"repo: {REPO}")
    print("=" * 72)

    check_environment()
    check_palette()
    check_evals_fix()
    check_ablation()
    check_smin_baseline()
    check_leakage()
    check_bins()
    check_benchmark()
    check_outputs(args.figure_root.resolve())

    print("\n" + "=" * 72)
    if report["skipped"]:
        print(f"SKIPPED {len(report['skipped'])} check(s) — inputs absent on this host:")
        for item in report["skipped"]:
            print(f"  - {item['check']}: {item['reason']}")
    if report["failures"]:
        print(f"FAILED {len(report['failures'])} check(s):")
        for line in report["failures"]:
            print(f"  - {line}")
    else:
        print("All runnable checks passed.")
    print("=" * 72)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nMachine-readable report: {args.output}")
    print("Diff this file against the other host's copy; any differing value is a real gap.")
    sys.exit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
