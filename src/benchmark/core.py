#!/usr/bin/env python3
"""Prepare, run, normalize, evaluate, and plot the ARC baseline benchmark.

All methods are converted to the same NPZ schema:
``protein_ids`` (N), ``go_terms`` (C), and ``scores`` (N, C).  Evaluation
always reindexes those files against the locked threshold-30 ARC datasets.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

ONTOLOGIES = {
    "mf": "molecular_function",
    "bp": "biological_process",
    "cc": "cellular_component",
}
ROOT_TERMS = {"mf": "GO:0003674", "bp": "GO:0008150", "cc": "GO:0005575"}
SEEDS = (1103, 2207, 3301, 4409, 5501)


def log(message: str) -> None:
    print(message, flush=True)


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    records[current] = "".join(chunks)
                current = line[1:].split()[0]
                chunks = []
            elif current is None:
                raise ValueError(f"{path}: sequence before first FASTA header")
            else:
                chunks.append(line)
    if current is not None:
        records[current] = "".join(chunks)
    return records


def write_fasta(path: Path, ids: Iterable[str], sequences: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for protein_id in ids:
            sequence = sequences[protein_id]
            handle.write(f">{protein_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")


def load_dataset(path: Path):
    """Load either the committed record list or a model-ready graph dataset."""
    with path.open("rb") as handle:
        return pickle.load(handle)


def dataset_path(data_root: Path, ontology: str, split: str) -> Path:
    return data_root / "datasets" / "threshold_30" / f"{ontology}_{split}.pkl"


def _records_to_dataset(raw_by_split: dict[str, list[dict]], ontology: str):
    """Recreate the exact sorted label vocabulary used by ARC model preprocessing."""
    from src.arc_dataset import ArcGraphDataset

    terms = sorted({
        term
        for records in raw_by_split.values()
        for record in records
        for term in record.get("labels", ())
    })
    term_index = {term: index for index, term in enumerate(terms)}
    converted = {}
    for split, records in raw_by_split.items():
        protein_ids = [str(record["id"]) for record in records]
        labels = np.zeros((len(records), len(terms)), dtype=np.float32)
        for row, record in enumerate(records):
            if record.get("ontology") not in (None, ontology):
                raise ValueError(
                    f"{ontology}/{split}: record {record['id']} has ontology "
                    f"{record.get('ontology')!r}"
                )
            if record.get("split") not in (None, split):
                raise ValueError(
                    f"{ontology}/{split}: record {record['id']} has split "
                    f"{record.get('split')!r}"
                )
            for term in record.get("labels", ()):
                labels[row, term_index[term]] = 1.0
        converted[split] = ArcGraphDataset(
            ".", protein_ids, labels, terms, ontology, split
        )
    return converted


def load_locked_datasets(data_root: Path) -> dict[str, dict[str, object]]:
    from src.arc_dataset import ArcGraphDataset

    datasets: dict[str, dict[str, object]] = {}
    for short, ontology in ONTOLOGIES.items():
        raw_by_split = {}
        for split in ("train", "valid", "test"):
            path = dataset_path(data_root, ontology, split)
            if not path.is_file():
                raise FileNotFoundError(f"Missing locked dataset: {path}")
            raw_by_split[split] = load_dataset(path)

        if all(isinstance(raw_by_split[split], list) for split in raw_by_split):
            datasets[short] = _records_to_dataset(raw_by_split, ontology)
        elif all(isinstance(raw_by_split[split], ArcGraphDataset) for split in raw_by_split):
            datasets[short] = raw_by_split
        else:
            schemas = {split: type(value).__name__ for split, value in raw_by_split.items()}
            raise TypeError(f"{ontology}: mixed or unsupported dataset schemas: {schemas}")

        for split, dataset in datasets[short].items():
            path = dataset_path(data_root, ontology, split)
            if dataset.split != split:
                raise ValueError(f"{path}: embedded split is {dataset.split!r}, expected {split!r}")
            if dataset.ontology != ontology:
                raise ValueError(f"{path}: embedded ontology is {dataset.ontology!r}")

        reference_terms = list(datasets[short]["train"].terms)
        for split in ("valid", "test"):
            if list(datasets[short][split].terms) != reference_terms:
                raise ValueError(f"{ontology}: GO vocabulary differs across splits")

    for split in ("train", "valid", "test"):
        reference = list(datasets["mf"][split].protein_ids)
        for short in ("bp", "cc"):
            if list(datasets[short][split].protein_ids) != reference:
                raise ValueError(f"Protein order differs across ontologies for {split}")
    return datasets


def save_prediction(
    workspace: Path,
    method: str,
    ontology: str,
    protein_ids: Iterable[str],
    go_terms: Iterable[str],
    scores: np.ndarray,
    *,
    score_type: str = "continuous",
    metadata: dict | None = None,
) -> Path:
    protein_ids = np.asarray(list(protein_ids), dtype=str)
    go_terms = np.asarray(list(go_terms), dtype=str)
    scores = np.asarray(scores, dtype=np.float32)
    expected = (len(protein_ids), len(go_terms))
    if scores.shape != expected:
        raise ValueError(f"{method}/{ontology}: score shape {scores.shape}, expected {expected}")
    if not np.isfinite(scores).all():
        raise ValueError(f"{method}/{ontology}: predictions contain NaN or infinity")
    scores = np.clip(scores, 0.0, 1.0)
    out_dir = workspace / "predictions" / method
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ontology}.npz"
    np.savez_compressed(path, protein_ids=protein_ids, go_terms=go_terms, scores=scores)
    payload = {
        "method": method,
        "ontology": ontology,
        "score_type": score_type,
        "proteins": int(len(protein_ids)),
        "terms": int(len(go_terms)),
        "nonzero_scores": int(np.count_nonzero(scores)),
        **(metadata or {}),
    }
    (out_dir / f"{ontology}.metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_prediction(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        ids = [str(value) for value in data["protein_ids"]]
        terms = [str(value) for value in data["go_terms"]]
        scores = np.asarray(data["scores"], dtype=np.float32)
    if scores.shape != (len(ids), len(terms)):
        raise ValueError(f"{path}: invalid standardized prediction shape")
    return ids, terms, scores


def parse_obo(path: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    parents: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, str] = {}
    current: str | None = None
    obsolete = False
    alt_ids: list[str] = []

    def finish() -> None:
        nonlocal current, obsolete, alt_ids
        if current and not obsolete:
            for alt in alt_ids:
                aliases[alt] = current
        current, obsolete, alt_ids = None, False, []

    with path.open() as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                finish()
            elif line.startswith("["):
                finish()
            elif current is None and line.startswith("id: GO:"):
                current = line.split("id: ", 1)[1]
            elif current is not None:
                if line.startswith("id: GO:"):
                    current = line.split("id: ", 1)[1]
                elif line.startswith("alt_id: GO:"):
                    alt_ids.append(line.split("alt_id: ", 1)[1])
                elif line.startswith("is_a: GO:"):
                    parents[current].add(line.split()[1])
                elif line.startswith("relationship: part_of GO:"):
                    parents[current].add(line.split()[2])
                elif line == "is_obsolete: true":
                    obsolete = True
    finish()
    return dict(parents), aliases


def ancestors(term: str, parents: dict[str, set[str]], cache: dict[str, set[str]]) -> set[str]:
    if term in cache:
        return cache[term]
    found: set[str] = {term}
    queue: deque[str] = deque([term])
    while queue:
        child = queue.popleft()
        for parent in parents.get(child, ()):
            if parent not in found:
                found.add(parent)
                queue.append(parent)
    cache[term] = found
    return found


def normalize_term(term: str, aliases: dict[str, str]) -> str | None:
    match = re.search(r"GO:\d{7}", str(term))
    if not match:
        return None
    value = match.group(0)
    return aliases.get(value, value)


def prepare(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    datasets = load_locked_datasets(data_root)
    test_ids = list(datasets["mf"]["test"].protein_ids)
    if len(test_ids) != args.expected_test_size:
        raise SystemExit(
            f"Locked test-set check failed: found {len(test_ids)}, expected {args.expected_test_size}"
        )

    split_root = data_root / "pdb_splits" / "threshold_30"
    sequence_sets: dict[str, dict[str, str]] = {}
    split_ids: dict[str, list[str]] = {}
    for split in ("train", "valid", "test"):
        ids = list(datasets["mf"][split].protein_ids)
        fasta = split_root / f"_{split}_sequences.fasta"
        sequences = read_fasta(fasta)
        missing = [protein_id for protein_id in ids if protein_id not in sequences]
        extras = sorted(set(sequences) - set(ids))
        if missing or extras:
            raise ValueError(
                f"{fasta}: ID mismatch; missing={len(missing)}, extra={len(extras)}"
            )
        split_ids[split] = ids
        sequence_sets[split] = sequences
        write_fasta(workspace / "inputs" / f"{split}.fasta", ids, sequences)

    query_ids = split_ids["valid"] + split_ids["test"]
    query_sequences = {**sequence_sets["valid"], **sequence_sets["test"]}
    write_fasta(workspace / "inputs" / "valid_test.fasta", query_ids, query_sequences)

    label_dir = workspace / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for short in ONTOLOGIES:
        for split in ("train", "valid", "test"):
            dataset = datasets[short][split]
            np.savez_compressed(
                label_dir / f"{short}_{split}.npz",
                protein_ids=np.asarray(dataset.protein_ids, dtype=str),
                go_terms=np.asarray(dataset.terms, dtype=str),
                labels=np.asarray(dataset.labels, dtype=np.uint8),
            )
            with gzip.open(label_dir / f"{short}_{split}.tsv.gz", "wt") as handle:
                handle.write("protein_id\tgo_term\n")
                rows, cols = np.where(np.asarray(dataset.labels) > 0)
                for row, col in zip(rows, cols):
                    handle.write(f"{dataset.protein_ids[row]}\t{dataset.terms[col]}\n")

    manifest = {
        "schema_version": 1,
        "split": "nominal_30_identity_80_coverage",
        "data_root": str(data_root),
        "test_proteins": len(split_ids["test"]),
        "valid_proteins": len(split_ids["valid"]),
        "train_proteins": len(split_ids["train"]),
        "experimental_structures": True,
        "alphafold_structures": False,
        "plddt_filtering": False,
        "ontology_terms": {
            short: len(datasets[short]["test"].terms) for short in ONTOLOGIES
        },
    }
    (workspace / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(json.dumps(manifest, indent=2))
    if args.extract_structures:
        extract_structures(data_root, workspace, split_ids, sequence_sets)


class _SingleChainSelect:
    """Biopython PDBIO select adapter for one model and one chain."""

    def __init__(self, model, chain):
        self.model = model
        self.chain = chain

    def accept_model(self, model):
        return int(model.id == self.model.id)

    def accept_chain(self, chain):
        return int(chain is self.chain)

    def accept_residue(self, residue):
        return 1

    def accept_atom(self, atom):
        return 1


def extract_structures(
    data_root: Path,
    workspace: Path,
    split_ids: dict[str, list[str]],
    sequence_sets: dict[str, dict[str, str]],
) -> None:
    from Bio.PDB import MMCIFParser, PDBIO
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Structure import Structure

    source_dir = data_root / "structure_files"
    rows: list[dict[str, object]] = []
    parser = MMCIFParser(QUIET=True, auth_chains=True)
    for split in ("train", "valid", "test"):
        out_dir = workspace / "inputs" / "structures" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, protein_id in enumerate(split_ids[split], start=1):
            pdb_id, chain_id = protein_id.split("_", 1)
            source = source_dir / f"{pdb_id.upper()}.cif.gz"
            plain_out = out_dir / f"{protein_id}.pdb"
            gzip_out = out_dir / f"{protein_id}.pdb.gz"
            status, error = "ok", ""
            ca_residues = 0
            try:
                if not source.is_file():
                    raise FileNotFoundError(str(source))
                with gzip.open(source, "rt") as handle:
                    structure = parser.get_structure(pdb_id, handle)
                model = next(structure.get_models())
                chain_lookup = {str(chain.id): chain for chain in model.get_chains()}
                chain = chain_lookup.get(chain_id)
                if chain is None:
                    chain = chain_lookup.get(chain_id.upper()) or chain_lookup.get(chain_id.lower())
                if chain is None:
                    raise KeyError(f"chain {chain_id!r}; available={sorted(chain_lookup)[:20]}")
                selected_residues = [
                    residue for residue in chain.get_residues()
                    if residue.id[0] == " " and residue.has_id("CA")
                ]
                ca_residues = len(selected_residues)
                output_structure = Structure(pdb_id)
                output_model = Model(0)
                output_chain = Chain("A")
                output_structure.add(output_model)
                output_model.add(output_chain)
                for residue in selected_residues:
                    output_chain.add(residue.copy())
                io = PDBIO()
                io.set_structure(output_structure)
                io.save(str(plain_out))
                with plain_out.open("rb") as source_handle, gzip.open(gzip_out, "wb") as target:
                    shutil.copyfileobj(source_handle, target)
            except Exception as exc:
                status, error = "failed", f"{type(exc).__name__}: {exc}"
            sequence_length = len(sequence_sets[split][protein_id])
            rows.append({
                "protein_id": protein_id,
                "split": split,
                "source": str(source),
                "pdb_file": str(plain_out),
                "status": status,
                "ca_residues": ca_residues,
                "sequence_length": sequence_length,
                "structure_sequence_coverage": ca_residues / max(sequence_length, 1),
                "error": error,
            })
            if index % 500 == 0:
                log(f"Extracted {index}/{len(split_ids[split])} {split} structures")
    report = workspace / "inputs" / "structure_manifest.tsv"
    with report.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if row["status"] != "ok"]
    test_failures = [row for row in failures if row["split"] == "test"]
    log(f"Structure extraction: {len(rows) - len(failures)}/{len(rows)} succeeded; test failures={len(test_failures)}")
    if test_failures:
        raise SystemExit(f"Experimental structure extraction failed for {len(test_failures)} test proteins; see {report}")


def load_label_npz(workspace: Path, ontology: str, split: str):
    path = workspace / "labels" / f"{ontology}_{split}.npz"
    with np.load(path, allow_pickle=False) as data:
        return (
            [str(value) for value in data["protein_ids"]],
            [str(value) for value in data["go_terms"]],
            np.asarray(data["labels"], dtype=np.uint8),
        )


def best_micro_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true.ravel(), scores.ravel())
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    index = int(np.nanargmax(f1))
    threshold = float(thresholds[index]) if index < len(thresholds) else 1.0
    return float(f1[index]), threshold


def export_hybrid(args: argparse.Namespace) -> None:
    import torch

    from src.arc_dataset import make_dataloader
    from src.checkpoint_loader import model_from_checkpoint

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; the ARC benchmark job must request and use a GPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Exporting DeepGreenGO predictions on {device}")
    workspace = args.workspace.resolve()
    data_root = args.data_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    graph_root = args.graph_root.resolve()
    datasets = load_locked_datasets(data_root)
    for short, ontology in ONTOLOGIES.items():
        split_predictions: dict[str, list[np.ndarray]] = {"valid": [], "test": []}
        model_variants: set[str] = set()
        for seed in SEEDS:
            checkpoint_path = checkpoint_root / ontology / f"seed_{seed}" / "best_checkpoint.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing five-seed checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_config = checkpoint.get("config", {})
            model_variant = str(
                checkpoint_config.get("model", checkpoint_config.get("model_type", "Hybrid"))
            )
            model_variants.add(model_variant)
            model = model_from_checkpoint(checkpoint, device=device)
            for split in ("valid", "test"):
                dataset = datasets[short][split]
                dataset.graph_dir = str(graph_root)
                loader = make_dataloader(dataset, args.batch_size, False, args.workers)
                chunks: list[np.ndarray] = []
                with torch.inference_mode():
                    for batch in loader:
                        batch = batch.to(device)
                        chunks.append(
                            torch.sigmoid(model(batch.x, batch.edge_index, batch.batch)).cpu().numpy()
                        )
                scores = np.vstack(chunks).astype(np.float32)
                split_predictions[split].append(scores)
                save_prediction(
                    workspace,
                    f"deepgreengo_seed_{seed}_{split}",
                    short,
                    dataset.protein_ids,
                    dataset.terms,
                    scores,
                    metadata={
                        "seed": seed,
                        "split": split,
                        "checkpoint": str(checkpoint_path),
                        "model_variant": model_variant,
                    },
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(model_variants) != 1:
            raise ValueError(
                f"{ontology}: ensemble checkpoints disagree on model variant: "
                f"{sorted(model_variants)}"
            )
        model_variant = next(iter(model_variants))
        valid_scores = np.mean(split_predictions["valid"], axis=0)
        test_scores = np.mean(split_predictions["test"], axis=0)
        valid_dataset = datasets[short]["valid"]
        test_dataset = datasets[short]["test"]
        valid_f1, validation_threshold = best_micro_threshold(valid_dataset.labels, valid_scores)
        save_prediction(
            workspace, "deepgreengo_valid", short,
            valid_dataset.protein_ids, valid_dataset.terms, valid_scores,
            metadata={"ensemble_seeds": list(SEEDS), "split": "valid", "model_variant": model_variant},
        )
        save_prediction(
            workspace, "deepgreengo", short,
            test_dataset.protein_ids, test_dataset.terms, test_scores,
            metadata={
                "ensemble_seeds": list(SEEDS),
                "split": "test",
                "validation_micro_f1": valid_f1,
                "validation_threshold": validation_threshold,
                "model_variant": model_variant,
            },
        )

