"""Shared paths and file helpers for the reproducible PDB-cluster pipeline."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(
    os.environ.get("DGG_DATA_ROOT", PROJECT_DIR / "preprocessing" / "data")
).expanduser().resolve()
PDB_CLUSTER_DIR = DATA_DIR / "pdb_clusters"
STRUCTURE_DIR = DATA_DIR / "structure_files"
SPLIT_ROOT = DATA_DIR / "pdb_splits"
DATASET_ROOT = DATA_DIR / "datasets"

THRESHOLDS = (30, 40, 50, 70, 90, 95)
ONTOLOGIES = (
    "molecular_function",
    "biological_process",
    "cellular_component",
)
ONTOLOGY_ROOTS = {
    "molecular_function": "GO:0003674",
    "biological_process": "GO:0008150",
    "cellular_component": "GO:0005575",
}


def ensure_directories() -> None:
    for path in (DATA_DIR, PDB_CLUSTER_DIR, STRUCTURE_DIR, SPLIT_ROOT, DATASET_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current: str | None = None
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                if current in sequences:
                    raise ValueError(f"Duplicate FASTA identifier: {current}")
                sequences[current] = ""
            elif current is None:
                raise ValueError(f"Sequence before first FASTA header in {path}")
            else:
                sequences[current] += line
    return sequences


def write_fasta(records: dict[str, str], path: Path, width: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for protein_id in sorted(records):
            sequence = records[protein_id]
            handle.write(f">{protein_id}\n")
            for start in range(0, len(sequence), width):
                handle.write(sequence[start:start + width] + "\n")


def read_id_file(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate identifiers in {path}")
    return ids


def write_id_file(ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{protein_id}\n" for protein_id in ids))
