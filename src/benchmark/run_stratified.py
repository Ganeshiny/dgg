#!/usr/bin/env python3
"""Compute stratified benchmark analyses and write tidy CSVs for plotting.

Usage
-----
python -m src.benchmark.run_stratified --workspace arc_benchmark/nominal_30_identity_80_coverage

Outputs, all under <workspace>/results/:
  stratified_homology.csv    Fmax per homology bin, per method, per ontology
  stratified_homology_aupr.csv  micro-AUPR per homology bin
  stratified_ic.csv          AUPRC per term-IC bin
  stratified_depth.csv       AUPRC per GO-depth bin
  ic_weighted_pr.csv         IC-weighted precision/recall sweep
  stratified_manifest.json   bin definitions, counts, provenance
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .core import ONTOLOGIES, ROOT_TERMS, load_label_npz, parse_obo
from .evaluate import propagate_scores_to_ancestors
from .stratified import (
    DEPTH_BINS,
    HOMOLOGY_BINS,
    IC_BINS,
    depth_bin_of,
    homology_bin_of,
    ic_bin_of,
    ic_weighted_pr_curve,
    information_content,
    load_scores,
    max_identity_by_protein,
    micro_auprc,
    protein_centric_fmax,
    seed_score_matrices,
    term_centric_auprc,
    term_depths,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HOMOLOGY = (
    PROJECT_DIR
    / "preprocessing/data_arc_rebuild_2026_07_14/pdb_splits/threshold_30/blast_te_vs_tr.tsv"
)

# Allowlist of methods the stratification will load if their predictions
# exist. A method absent from this list is silently never stratified, which
# is why HEAL was missing from plots/stratified and from the stratified
# supplementary tables even though it had completed successfully. Presentation
# filtering (for example withholding DeepGOPlus/DeepGO-SE from figures) is
# applied downstream, so this stays inclusive and the results/ tables remain
# the complete evidence record.
METHODS = [
    "deepgreengo", "naive",
    "blast", "blast_max",
    "diamond", "diamond_max",
    "foldseek", "foldseek_max",
    "interproscan", "deepfri_sequence", "deepfri_structure",
    "dpfunc", "deepgoplus", "deepgose", "transfun",
    "heal", "gat_go", "deepgraphgo",
    "eggnog_mapper", "hayai", "gomap",
]

# Below this many items a bin's metric is too unstable to interpret; it is
# still emitted with its true n so the figure can flag rather than hide it.
MIN_BIN_SIZE = 5


def resolve_data_root(workspace: Path, override: Path | None) -> Path:
    if override is not None:
        root = override.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"--data-root does not exist: {root}")
        return root
    manifest = json.loads((workspace / "benchmark_manifest.json").read_text())
    recorded = Path(manifest["data_root"])
    if recorded.is_dir():
        return recorded
    local = PROJECT_DIR / "preprocessing" / recorded.name
    if local.is_dir():
        return local
    raise SystemExit(f"Cannot locate data root; pass --data-root. Tried {recorded} and {local}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--homology-tsv", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    data_root = resolve_data_root(workspace, args.data_root)
    homology_tsv = (args.homology_tsv or DEFAULT_HOMOLOGY).expanduser().resolve()
    obo_path = data_root / "go-basic.obo"
    if not obo_path.is_file():
        raise SystemExit(f"go-basic.obo not found at {obo_path}")

    print(f"[stratified] workspace={workspace}")
    print(f"[stratified] parsing {obo_path}")
    parents, _aliases = parse_obo(obo_path)

    homology_rows, homology_aupr_rows, ic_rows, depth_rows, pr_rows = [], [], [], [], []
    manifest: dict = {
        "workspace": str(workspace),
        "homology_tsv": str(homology_tsv),
        "obo": str(obo_path),
        "min_bin_size": MIN_BIN_SIZE,
        "prediction_ancestor_propagation": True,
        "bins": {"homology": HOMOLOGY_BINS, "ic": IC_BINS, "depth": DEPTH_BINS},
        "ontologies": {},
    }

    for short, ontology in ONTOLOGIES.items():
        protein_ids, go_terms, y_true = load_label_npz(workspace, short, "test")
        _, train_terms, train_labels = load_label_npz(workspace, short, "train")
        if train_terms != go_terms:
            raise SystemExit(f"{ontology}: train/test GO vocabularies differ")
        y_true = y_true.astype(np.uint8)

        # ---- protein-level homology bins -------------------------------
        identity = max_identity_by_protein(homology_tsv, protein_ids)
        protein_bin = np.asarray(
            [homology_bin_of(identity[str(pid)]) for pid in protein_ids]
        )

        # ---- term-level IC and depth bins ------------------------------
        ic = information_content(train_labels)
        term_ic_bin = np.asarray([ic_bin_of(value) for value in ic], dtype=object)
        depth_map = term_depths(go_terms, parents, ROOT_TERMS[short])
        term_depth_bin = np.asarray(
            [
                depth_bin_of(depth_map[str(term)]) if str(term) in depth_map else None
                for term in go_terms
            ],
            dtype=object,
        )

        manifest["ontologies"][ontology] = {
            "test_proteins": int(len(protein_ids)),
            "test_terms": int(len(go_terms)),
            "terms_with_depth": int(sum(1 for v in term_depth_bin if v is not None)),
            "homology_bin_counts": {
                name: int((protein_bin == name).sum()) for name in HOMOLOGY_BINS
            },
            "ic_bin_term_counts": {
                name: int(sum(1 for v in term_ic_bin if v == name)) for name in IC_BINS
            },
            "depth_bin_term_counts": {
                name: int(sum(1 for v in term_depth_bin if v == name)) for name in DEPTH_BINS
            },
        }
        print(f"[stratified] {ontology}: {len(protein_ids)} proteins, {len(go_terms)} terms")

        seeds = [
            propagate_scores_to_ancestors(matrix, go_terms, parents)
            for matrix in seed_score_matrices(
                workspace, short, protein_ids, go_terms
            )
        ]

        for method in METHODS:
            scores = load_scores(workspace, method, short, protein_ids, go_terms)
            if scores is None:
                continue
            scores = propagate_scores_to_ancestors(scores, go_terms, parents)
            per_seed = seeds if method == "deepgreengo" else []

            # --- panel a: Fmax by homology bin (protein subsets) --------
            for name in HOMOLOGY_BINS:
                mask = protein_bin == name
                n = int(mask.sum())
                if n == 0:
                    continue
                value = protein_centric_fmax(y_true[mask], scores[mask])
                replicates = [
                    protein_centric_fmax(y_true[mask], matrix[mask]) for matrix in per_seed
                ]
                homology_rows.append(_row(
                    ontology, method, name, n, value, replicates, "proteins"
                ))
                aupr_value = micro_auprc(y_true[mask], scores[mask])
                aupr_replicates = [
                    micro_auprc(y_true[mask], matrix[mask]) for matrix in per_seed
                ]
                homology_aupr_rows.append(_row(
                    ontology, method, name, n, aupr_value, aupr_replicates, "proteins"
                ))


            # --- panel c: AUPRC by term IC bin (term subsets) -----------
            for name in IC_BINS:
                columns = np.asarray([i for i, v in enumerate(term_ic_bin) if v == name])
                if columns.size == 0:
                    continue
                value = term_centric_auprc(y_true[:, columns], scores[:, columns])
                replicates = [
                    term_centric_auprc(y_true[:, columns], matrix[:, columns])
                    for matrix in per_seed
                ]
                ic_rows.append(_row(
                    ontology, method, name, int(columns.size), value, replicates, "terms"
                ))

            # --- panel f: AUPRC by GO depth bin (term subsets) ----------
            for name in DEPTH_BINS:
                columns = np.asarray([i for i, v in enumerate(term_depth_bin) if v == name])
                if columns.size == 0:
                    continue
                value = term_centric_auprc(y_true[:, columns], scores[:, columns])
                replicates = [
                    term_centric_auprc(y_true[:, columns], matrix[:, columns])
                    for matrix in per_seed
                ]
                depth_rows.append(_row(
                    ontology, method, name, int(columns.size), value, replicates, "terms"
                ))

            # --- panels b/d/e: IC-weighted PR curve ---------------------
            precision, recall = ic_weighted_pr_curve(y_true, scores, ic)
            thresholds = np.arange(0.01, 1.00, 0.01)
            for threshold, p, r in zip(thresholds, precision, recall):
                pr_rows.append({
                    "ontology": ontology,
                    "method": method,
                    "threshold": round(float(threshold), 4),
                    "ic_weighted_precision": _clean(p),
                    "ic_weighted_recall": _clean(r),
                })

        print(f"[stratified] {ontology}: done")

    results = workspace / "results"
    results.mkdir(parents=True, exist_ok=True)
    _write(results / "stratified_homology.csv", homology_rows)
    _write(results / "stratified_homology_aupr.csv", homology_aupr_rows)
    _write(results / "stratified_ic.csv", ic_rows)
    _write(results / "stratified_depth.csv", depth_rows)
    _write(results / "ic_weighted_pr.csv", pr_rows)
    (results / "stratified_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[stratified] wrote 5 tables + manifest to {results}")


def _clean(value) -> float | str:
    return "" if value is None or not np.isfinite(value) else round(float(value), 6)


def _row(ontology, method, bin_name, n, value, replicates, unit) -> dict:
    finite = [r for r in replicates if np.isfinite(r)]
    return {
        "ontology": ontology,
        "method": method,
        "bin": bin_name,
        "bin_unit": unit,
        "bin_n": n,
        "below_min_bin_size": bool(n < MIN_BIN_SIZE),
        "value": _clean(value),
        "seed_mean": _clean(np.mean(finite)) if finite else "",
        "seed_sd": _clean(np.std(finite, ddof=1)) if len(finite) > 1 else "",
        "seed_n": len(finite),
    }


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[stratified] WARNING: no rows for {path.name}")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stratified]   {path.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
