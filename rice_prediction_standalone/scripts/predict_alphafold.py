#!/usr/bin/env python3
"""Run the trained DeepGreenGO ensemble on AlphaFold structures."""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1

HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
POLARITY = {
    "A": 0, "R": 1, "N": 1, "D": 1, "C": 0,
    "Q": 1, "E": 1, "G": 0, "H": 1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 1, "T": 1, "W": 0, "Y": 1, "V": 0,
}
CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}
ONTOLOGIES = ("molecular_function", "biological_process", "cellular_component")
STRUCTURE_SUFFIXES = (".cif.gz", ".mmcif.gz", ".pdb.gz", ".cif", ".mmcif", ".pdb")
PROTBERT = "Rostlab/prot_bert_bfd"
PROTBERT_RESIDUES = 1022
AFDB_ID_PATTERN = re.compile(r"^AF-(?P<accession>.+?)-F\d+-model_v\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    aliases: frozenset[str]
    sequence: str


@dataclass(frozen=True)
class GraphInputs:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_node_features(self) -> int:
        return int(self.x.shape[1])


def read_fasta(path: Path) -> list[FastaRecord]:
    raw_records: list[tuple[str, str]] = []
    header_token: str | None = None
    sequence = ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header_token is not None:
                raw_records.append((header_token, sequence))
            header_token = line[1:].split()[0]
            sequence = ""
            if not header_token:
                raise ValueError(f"empty FASTA identifier in {path}")
        elif header_token is None:
            raise ValueError(f"sequence appears before the first FASTA header in {path}")
        else:
            sequence += re.sub(r"[^A-Z]", "", line.upper())
    if header_token is not None:
        raw_records.append((header_token, sequence))
    if not raw_records:
        raise ValueError(f"no sequences found in {path}")

    records: list[FastaRecord] = []
    seen_identifiers: set[str] = set()
    for token, record_sequence in raw_records:
        token_parts = token.split("|")
        identifier = (
            token_parts[1]
            if len(token_parts) >= 3 and token_parts[0].lower() in {"sp", "tr"}
            else token
        )
        if identifier in seen_identifiers:
            raise ValueError(f"duplicate FASTA identifier: {identifier}")
        seen_identifiers.add(identifier)
        aliases = {token, identifier}
        aliases.update(part for part in token_parts if part)
        afdb_match = AFDB_ID_PATTERN.fullmatch(token)
        if afdb_match:
            aliases.add(afdb_match.group("accession"))
        records.append(FastaRecord(identifier, frozenset(aliases), record_sequence))
    return records


def protein_id(path: Path) -> str:
    lower_name = path.name.lower()
    for suffix in STRUCTURE_SUFFIXES:
        if lower_name.endswith(suffix):
            return path.name[: -len(suffix)]
    raise ValueError(f"unsupported structure filename: {path.name}")


def canonical_structure_id(path: Path) -> str:
    """Return the UniProt accession from a standard AFDB filename when possible."""
    raw_identifier = protein_id(path)
    match = AFDB_ID_PATTERN.fullmatch(raw_identifier)
    return match.group("accession") if match else raw_identifier


def resolve_identifier(
    path: Path,
    structure_sequence: str,
    fasta_records: list[FastaRecord],
) -> str:
    """Resolve a structure to a FASTA ID by aliases or an exact sequence match."""
    raw_identifier = protein_id(path)
    canonical_id = canonical_structure_id(path)
    candidates = {raw_identifier, canonical_id}
    alias_matches = [record for record in fasta_records if candidates & record.aliases]
    if len(alias_matches) == 1:
        return alias_matches[0].identifier
    if len(alias_matches) > 1:
        names = ", ".join(record.identifier for record in alias_matches)
        raise ValueError(f"multiple FASTA records match {path.name}: {names}")

    sequence_matches = [record for record in fasta_records if record.sequence == structure_sequence]
    if len(sequence_matches) == 1:
        return sequence_matches[0].identifier
    if len(sequence_matches) > 1:
        print(
            f"Warning: {path.name} matches multiple identical FASTA sequences; "
            f"using AlphaFold/structure ID {canonical_id!r}.",
            file=sys.stderr,
        )
    elif fasta_records:
        print(
            f"Warning: no FASTA identifier or exact sequence match for {path.name}; "
            f"using AlphaFold/structure ID {canonical_id!r}.",
            file=sys.stderr,
        )
    return canonical_id


def structure_files(input_dir: Path) -> list[Path]:
    files = [
        path for path in input_dir.iterdir()
        if path.is_file() and any(path.name.lower().endswith(suffix) for suffix in STRUCTURE_SUFFIXES)
    ]
    return sorted(files)


def _parse_structure(path: Path):
    name = path.name.lower()
    parser = (
        MMCIFParser(QUIET=True)
        if name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
        else PDBParser(QUIET=True)
    )
    handle: TextIO | str
    if name.endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return parser.get_structure(protein_id(path), handle)
    return parser.get_structure(protein_id(path), str(path))


def structure_residues(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    structure = _parse_structure(path)
    chains = list(structure.get_chains())
    if not chains:
        raise ValueError("structure contains no chains")
    residues = [residue for residue in chains[0] if "CA" in residue]
    if not residues:
        raise ValueError(f"first chain {chains[0].id!r} contains no C-alpha atoms")
    atoms = [residue["CA"] for residue in residues]
    coordinates = np.asarray([atom.coord for atom in atoms], dtype=np.float32)
    plddt = np.asarray([float(atom.get_bfactor()) for atom in atoms], dtype=np.float32)
    sequence = "".join(seq1(residue.resname, undef_code="X") for residue in residues)
    return coordinates, plddt, sequence


def load_protbert_classes() -> tuple[Any, Any]:
    """Import text-only Transformers classes with an actionable ABI error."""
    try:
        from transformers import BertModel, BertTokenizer
    except Exception as exc:
        detail = str(exc)
        if "torchvision::nms" in detail or "Could not import module 'BertModel'" in detail:
            raise SystemExit(
                "Transformers could not import BertModel because the active environment has "
                "an incompatible torchvision/PyTorch installation. Create and activate the "
                "fresh .venv described in README_STANDALONE.md. If you intentionally reuse this "
                "environment, remove the unused broken torchvision package and reinstall any "
                "PyG extension wheels for the active PyTorch/CUDA version."
            ) from exc
        raise
    return BertModel, BertTokenizer


def embed_sequence(
    sequence: str,
    tokenizer: Any,
    bert: Any,
    device: torch.device,
) -> torch.Tensor:
    """Embed at most 1,022 residues, matching final training preprocessing."""
    sequence = sequence[:PROTBERT_RESIDUES]
    tokens = tokenizer(
        " ".join(sequence),
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=PROTBERT_RESIDUES + 2,
    ).to(device)
    with torch.no_grad():
        encoded = bert(**tokens).last_hidden_state[:, 1 : len(sequence) + 1].squeeze(0)
    if encoded.shape[0] != len(sequence):
        raise RuntimeError(
            f"ProtBERT returned {encoded.shape[0]} residue embeddings for {len(sequence)} residues"
        )
    return encoded


def build_graph(
    sequence: str,
    coordinates: np.ndarray,
    plddt: np.ndarray,
    embeddings: torch.Tensor,
    *,
    plddt_threshold: float,
    device: torch.device,
) -> GraphInputs:
    n = min(len(sequence), len(coordinates), embeddings.shape[0])
    if n == 0:
        raise ValueError("sequence and structure have no aligned residues")
    sequence = sequence[:n]
    coordinates = coordinates[:n]
    plddt = plddt[:n]
    embeddings = embeddings[:n]

    extra = torch.tensor(
        [[
            HYDROPHOBICITY.get(residue, 0.0),
            POLARITY.get(residue, 0),
            CHARGE.get(residue, 0),
        ] for residue in sequence],
        dtype=embeddings.dtype,
        device=device,
    )
    features = torch.cat([embeddings, extra], dim=1)

    squared_distance = ((coordinates[:, None] - coordinates[None, :]) ** 2).sum(axis=2)
    confident = plddt >= plddt_threshold
    confident_pairs = np.logical_and(confident[:, None], confident[None, :])
    adjacency = (squared_distance <= 100.0) & (squared_distance > 0.0) & confident_pairs
    edges = np.argwhere(adjacency).T
    edge_index = torch.as_tensor(edges.copy(), dtype=torch.long, device=device)
    batch = torch.zeros(features.shape[0], dtype=torch.long, device=device)
    return GraphInputs(x=features, edge_index=edge_index, batch=batch)


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--fasta",
        type=Path,
        help="Optional FASTA used only to map AFDB/UniProt IDs to preferred output IDs.",
    )
    parser.add_argument("--bundle", type=Path, default=script_root)
    parser.add_argument("--ontology", choices=ONTOLOGIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plddt-threshold", type=float, default=70.0)
    parser.add_argument("--call-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(runtime_root))

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if args.fasta is not None and not args.fasta.is_file():
        raise SystemExit(f"FASTA file does not exist: {args.fasta}")
    files = structure_files(args.input_dir)
    if not files:
        raise SystemExit(f"No supported structure files found in {args.input_dir}")

    fasta_records = read_fasta(args.fasta) if args.fasta is not None else []
    terms = json.loads((args.bundle / "labels" / f"{args.ontology}_terms.json").read_text())
    checkpoints = sorted((args.bundle / "weights" / args.ontology).glob("seed_*/best_checkpoint.pt"))
    if len(checkpoints) != 5:
        raise SystemExit(f"Expected 5 checkpoints for {args.ontology}, found {len(checkpoints)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading ProtBERT and five-model ensemble on {device} ...")
    BertModel, BertTokenizer = load_protbert_classes()
    tokenizer = BertTokenizer.from_pretrained(PROTBERT)
    bert = BertModel.from_pretrained(PROTBERT).to(device).eval()

    from src.checkpoint_loader import load_models

    models = load_models(checkpoints, device=device)
    if any(model.output_layer.out_features != len(terms) for model in models):
        raise SystemExit("Checkpoint output dimension does not match the ontology label file")

    rows: list[str] = []
    for structure_path in files:
        structure_identifier = protein_id(structure_path)
        coordinates, plddt, sequence = structure_residues(structure_path)
        identifier = resolve_identifier(structure_path, sequence, fasta_records)
        embeddings = embed_sequence(sequence, tokenizer, bert, device)
        graph = build_graph(
            sequence,
            coordinates,
            plddt,
            embeddings,
            plddt_threshold=args.plddt_threshold,
            device=device,
        )
        with torch.no_grad():
            scores = torch.stack([
                torch.sigmoid(model(graph.x, graph.edge_index, graph.batch)).squeeze(0)
                for model in models
            ]).mean(0)
        for go_term, score in zip(terms, scores.cpu().tolist()):
            rows.append(
                f"{identifier}\t{structure_identifier}\t{go_term}\t{score:.8g}\t"
                f"{int(score >= args.call_threshold)}\t{args.ontology}\n"
            )
        print(
            f"Processed {structure_identifier} as {identifier}: "
            f"{graph.num_nodes} residues"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "protein_id\tstructure_id\tgo_id\tscore\tcalled\tontology\n" + "".join(rows)
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
