#!/usr/bin/env python3
"""Build and validate PKLs for the nominal 30% identity / 80% coverage split."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / (chr(115) + chr(114) + chr(99))))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "preprocessing" / "pdb_clusters"))

from preprocessing.create_batch_dataset import compute_residue_features, seq2protbert
from preprocessing.pdb_clusters.common import ONTOLOGIES, read_fasta, read_id_file, sha256
from src.arc_dataset import ArcGraphDataset, make_dataloader

DEFAULT_ARC_ROOT = Path("/home/ganeshiny.sridharan/dgg/deep-green-GO")
EXPECTED_COUNTS = {"train": 6026, "valid": 754, "test": 754}
SPLIT_NAME = "nominal_30_identity_80_coverage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_ARC_ROOT / "preprocessing" / "data_arc_rebuild_2026_07_14",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=1)
    return parser.parse_args()


def load_inputs(data_root: Path):
    split_root = data_root / "pdb_splits" / "threshold_30"
    required = [
        data_root / "all_sequences.fasta",
        data_root / "protein_records.pkl",
        data_root / "strict_mmseqs" / "manifest_30.json",
        data_root / "strict_mmseqs" / "clusters-by-entity-30.txt",
        data_root / "split_verification.json",
        data_root / "blast_leakage.json",
    ] + [split_root / f"_{name}.txt" for name in EXPECTED_COUNTS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing required nominal-split inputs:\n" + "\n".join(missing))

    strict = json.loads(required[2].read_text())
    if strict.get("threshold") != 30 or float(strict.get("coverage", -1)) != 0.8:
        raise SystemExit(f"Unexpected strict MMseqs manifest: {strict}")
    verification = json.loads((data_root / "split_verification.json").read_text())["30"]
    cluster_file = Path(verification["cluster_file"])
    if cluster_file.name != "clusters-by-entity-30.txt" or "strict_mmseqs" not in cluster_file.parts:
        raise SystemExit("30% verification did not use strict_mmseqs/clusters-by-entity-30.txt")
    if verification["components_crossing_splits"] or verification["exact_sequence_duplicates_train_test"]:
        raise SystemExit(f"Split verification failed: {verification}")

    sequences = read_fasta(data_root / "all_sequences.fasta")
    splits = {name: read_id_file(split_root / f"_{name}.txt") for name in EXPECTED_COUNTS}
    for name, expected in EXPECTED_COUNTS.items():
        if len(splits[name]) != expected:
            raise SystemExit(f"{name}: expected {expected}, found {len(splits[name])}")
    split_sets = {name: set(ids) for name, ids in splits.items()}
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    ):
        raise SystemExit("Split identifiers overlap")
    if set().union(*split_sets.values()) != set(sequences):
        raise SystemExit("Split identifiers do not match all_sequences.fasta")

    with (data_root / "protein_records.pkl").open("rb") as handle:
        records = pickle.load(handle)
    if set(records) != set(sequences):
        raise SystemExit("protein_records.pkl identifiers do not match FASTA identifiers")
    for protein_id, sequence in sequences.items():
        if not sequence or records[protein_id].get("sequence") != sequence:
            raise SystemExit(f"Sequence unavailable or inconsistent for {protein_id}")
        annotations = records[protein_id].get("annotations", {})
        if set(annotations) != set(ONTOLOGIES):
            raise SystemExit(f"Ontology keys are inconsistent for {protein_id}")

    audit = json.loads((data_root / "blast_leakage.json").read_text())["30"]
    denominator = int(audit["n_test_sequences"])
    fraction = float(audit["fraction_at_or_above_threshold_and_80pct_coverage"])
    residual_count = round(denominator * fraction)
    if denominator != EXPECTED_COUNTS["test"]:
        raise SystemExit(f"Residual audit denominator is {denominator}, expected 754")
    print(
        "Independent residual-similarity audit: "
        f"{residual_count}/{denominator} test sequences ({100 * residual_count / denominator:.2f}%)."
    )
    return sequences, records, splits, strict, verification, residual_count


def build_graph(protein_id: str, cmap_path: Path, graph_path: Path) -> None:
    with np.load(cmap_path) as cmap:
        raw_sequence = cmap["seqres"]
        sequence = str(raw_sequence.item()) if raw_sequence.ndim == 0 else str(raw_sequence)
        protbert = torch.tensor(seq2protbert(sequence), dtype=torch.float32).squeeze(0)
        hydrophobicity, polarity, charge = compute_residue_features(sequence)
        extra = torch.tensor(
            np.stack([hydrophobicity, polarity, charge], axis=1), dtype=torch.float32
        )
        residue_count = min(len(sequence), protbert.shape[0], cmap["C_alpha"].shape[0], 1022)
        if residue_count <= 0:
            raise ValueError(f"No aligned residues for {protein_id}")
        x = torch.cat([protbert[:residue_count], extra[:residue_count]], dim=1)
        distances = cmap["C_alpha"][:residue_count, :residue_count]
        with np.errstate(invalid="ignore"):
            adjacency = distances <= 10.0
        np.fill_diagonal(adjacency, False)
        rows, cols = np.nonzero(adjacency)
        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = graph_path.with_suffix(".pt.tmp")
    torch.save(
        Data(
            x=x,
            edge_index=edge_index,
            u=protein_id,
            length=torch.tensor(residue_count, dtype=torch.long),
        ),
        temporary,
    )
    temporary.replace(graph_path)


def build_graph_cache(data_root: Path, protein_ids: list[str]) -> Path:
    cmap_dir = data_root / "structure_files" / "tmp_cmap_files"
    graph_dir = data_root / "arc_tuning" / "graphs_protbert"
    missing_cmaps = [protein_id for protein_id in protein_ids if not (cmap_dir / f"{protein_id}.npz").is_file()]
    if missing_cmaps:
        raise SystemExit(
            f"Missing {len(missing_cmaps)} contact maps; first IDs: {missing_cmaps[:5]}. "
            "Tuning never regenerates structures or contact maps."
        )
    pending = [protein_id for protein_id in protein_ids if not (graph_dir / f"{protein_id}.pt").is_file()]
    if pending:
        if torch.cuda.is_available():
            raise SystemExit("Dataset preprocessing must run without a GPU-visible PyTorch device")
        print(f"Building {len(pending):,} shared ProtBERT graph files on CPU (cached files are reused).")
        for index, protein_id in enumerate(pending, 1):
            build_graph(protein_id, cmap_dir / f"{protein_id}.npz", graph_dir / f"{protein_id}.pt")
            if index % 100 == 0 or index == len(pending):
                print(f"  built {index:,}/{len(pending):,}")
    return graph_dir


def build_pkls(data_root: Path, graph_dir: Path, records: dict, splits: dict) -> Path:
    output_dir = data_root / "arc_tuning" / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabularies = {
        ontology: sorted(
            {term for record in records.values() for term in record["annotations"][ontology]}
        )
        for ontology in ONTOLOGIES
    }
    for ontology, terms in vocabularies.items():
        term_index = {term: index for index, term in enumerate(terms)}
        for split, protein_ids in splits.items():
            labels = np.zeros((len(protein_ids), len(terms)), dtype=np.float32)
            for row, protein_id in enumerate(protein_ids):
                for term in records[protein_id]["annotations"][ontology]:
                    labels[row, term_index[term]] = 1.0
            dataset = ArcGraphDataset(
                graph_dir, protein_ids, labels, terms, ontology, split, SPLIT_NAME
            )
            path = output_dir / f"{ontology}_{split}.pkl"
            temporary = path.with_suffix(".pkl.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
    return output_dir


def validate_pkls(output_dir: Path, graph_dir: Path, splits: dict, smoke_batches: int) -> dict:
    ontology_summary = {}
    for ontology in ONTOLOGIES:
        datasets = {}
        for split, protein_ids in splits.items():
            path = output_dir / f"{ontology}_{split}.pkl"
            with path.open("rb") as handle:
                dataset = pickle.load(handle)
            if not isinstance(dataset, ArcGraphDataset) or dataset.split != split:
                raise SystemExit(f"Unexpected dataset schema in {path}")
            if dataset.protein_ids != protein_ids:
                raise SystemExit(f"Dataset identifiers do not match split manifest: {path}")
            if dataset.labels.shape != (len(protein_ids), dataset.num_classes):
                raise SystemExit(f"Invalid label dimensions: {path}")
            datasets[split] = dataset
        terms = [dataset.terms for dataset in datasets.values()]
        if not all(term_list == terms[0] for term_list in terms[1:]):
            raise SystemExit(f"Ontology vocabulary differs across {ontology} splits")
        for dataset in datasets.values():
            missing = [protein_id for protein_id in dataset.protein_ids if not (graph_dir / f"{protein_id}.pt").is_file()]
            if missing:
                raise SystemExit(f"Missing {len(missing)} graphs for {ontology}/{dataset.split}")
        if smoke_batches:
            loader = make_dataloader(datasets["train"], batch_size=2, shuffle=False)
            for batch_index, batch in enumerate(loader, 1):
                if batch.x.ndim != 2 or batch.y.shape[1] != datasets["train"].num_classes:
                    raise SystemExit(f"Dataloader schema smoke test failed for {ontology}")
                if batch_index >= smoke_batches:
                    break
        ontology_summary[ontology] = {
            "label_dimension": datasets["train"].num_classes,
            "train_positive_terms": int((datasets["train"].labels.sum(axis=0) > 0).sum()),
        }
    return ontology_summary


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    sequences, records, splits, strict, verification, residual_count = load_inputs(data_root)
    graph_dir = data_root / "arc_tuning" / "graphs_protbert"
    output_dir = data_root / "arc_tuning" / "datasets"
    if not args.validate_only:
        graph_dir = build_graph_cache(data_root, sorted(sequences))
        output_dir = build_pkls(data_root, graph_dir, records, splits)
    summary = validate_pkls(output_dir, graph_dir, splits, args.smoke_batches)
    manifest = {
        "schema_version": 1,
        "description": "nominal 30% identity / 80% coverage split",
        "threshold_percent": 30,
        "coverage": 0.8,
        "split_counts": EXPECTED_COUNTS,
        "strict_cluster_manifest": str(data_root / "strict_mmseqs" / "manifest_30.json"),
        "strict_cluster_file": verification["cluster_file"],
        "residual_test_to_train_similarity": {
            "count": residual_count,
            "denominator_all_test_sequences": EXPECTED_COUNTS["test"],
            "percentage": 100 * residual_count / EXPECTED_COUNTS["test"],
        },
        "ontologies": summary,
        "dataset_dir": str(output_dir),
        "graph_dir": str(graph_dir),
        "source_sha256": {
            "all_sequences.fasta": sha256(data_root / "all_sequences.fasta"),
            "strict_manifest": sha256(data_root / "strict_mmseqs" / "manifest_30.json"),
        },
        "tuning_loads": ["train", "valid"],
        "test_set_policy": "not loaded by tuning; reserved for final evaluation only",
    }
    manifest_path = data_root / "arc_tuning" / "tuning_dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
