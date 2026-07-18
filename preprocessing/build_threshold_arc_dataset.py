#!/usr/bin/env python3
"""Build ARC train/valid/test PKLs from an existing threshold split manifest."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

from preprocessing.pdb_clusters.common import ONTOLOGIES, read_fasta, read_id_file
from src.arc_dataset import ArcGraphDataset
from preprocessing.build_arc_tuning_dataset import build_graph_cache, validate_pkls


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--tuning-root", type=Path, required=True)
    p.add_argument("--graph-root", type=Path, required=True)
    p.add_argument("--threshold", type=int, required=True, choices=[40, 50, 70, 90, 95])
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--smoke-batches", type=int, default=1)
    args = p.parse_args()

    data_root = args.data_root.resolve()
    split_root = data_root / "pdb_splits" / f"threshold_{args.threshold}"
    split_log_path = split_root / "split_log.json"
    if not split_log_path.is_file():
        raise SystemExit(f"Missing split metadata: {split_log_path}")
    split_log = json.loads(split_log_path.read_text())
    if int(split_log.get("threshold", -1)) != args.threshold:
        raise SystemExit(f"Threshold metadata mismatch in {split_log_path}")

    sequences = read_fasta(data_root / "all_sequences.fasta")
    with (data_root / "protein_records.pkl").open("rb") as handle:
        records = pickle.load(handle)
    if set(records) != set(sequences):
        raise SystemExit("protein_records.pkl identifiers do not match all_sequences.fasta")

    expected = {name: int(split_log["split_sizes"][name]) for name in ("train", "valid", "test")}
    splits = {}
    for name, count in expected.items():
        path = split_root / f"_{name}.txt"
        ids = read_id_file(path)
        if len(ids) != count:
            raise SystemExit(f"{path}: expected {count} IDs, found {len(ids)}")
        splits[name] = ids
    sets = {k: set(v) for k, v in splits.items()}
    if any(sets[a] & sets[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise SystemExit(f"Threshold {args.threshold}: split identifiers overlap")
    if set().union(*sets.values()) != set(sequences):
        raise SystemExit(f"Threshold {args.threshold}: split IDs do not cover all sequences")

    graph_dir = build_graph_cache(data_root, args.graph_root.resolve(), sorted(sequences))
    out = args.tuning_root.resolve() / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    split_label = f"identity_{args.threshold}_coverage_{int(args.coverage * 100)}"
    vocabularies = {o: sorted({t for r in records.values() for t in r["annotations"][o]}) for o in ONTOLOGIES}
    for ontology, terms in vocabularies.items():
        index = {term: i for i, term in enumerate(terms)}
        for split, ids in splits.items():
            labels = np.zeros((len(ids), len(terms)), dtype=np.float32)
            for row, protein_id in enumerate(ids):
                for term in records[protein_id]["annotations"][ontology]:
                    labels[row, index[term]] = 1.0
            path = out / f"{ontology}_{split}.pkl"
            tmp = path.with_suffix(".pkl.tmp")
            with tmp.open("wb") as handle:
                pickle.dump(ArcGraphDataset(graph_dir, ids, labels, terms, ontology, split, split_label), handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)

    summary = validate_pkls(out, graph_dir, splits, args.smoke_batches)
    manifest = {
        "schema_version": 1,
        "description": f"{args.threshold}% sequence-identity split with {args.coverage:.2f} coverage",
        "threshold_percent": args.threshold,
        "coverage": args.coverage,
        "split_counts": expected,
        "source_split_dir": str(split_root),
        "dataset_dir": str(out),
        "graph_dir": str(graph_dir),
        "ontologies": summary,
        "tuning_loads": ["train", "valid"],
        "test_set_policy": "not loaded by tuning; reserved for final evaluation only",
    }
    (args.tuning_root.resolve() / "tuning_dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
