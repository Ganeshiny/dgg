#!/usr/bin/env python3
"""Build the canonical eligible Viridiplantae chain dataset from fresh inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import obonet
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from common import DATA_DIR, ONTOLOGIES, ONTOLOGY_ROOTS, STRUCTURE_DIR, sha256, write_fasta


EXPERIMENTAL_EVIDENCE = {
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
    "HTP", "HDA", "HMP", "HGI", "HEP",
}


def as_list(value) -> list[str]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def canonical_sequence(raw: str) -> str:
    sequence = re.sub(r"\s+", "", raw).upper()
    sequence = re.sub(r"[^A-Z]", "", sequence)
    return sequence


def parse_cif_sequences(
    path: Path,
    allowed_entities: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({PDB_auth-chain: sequence}, {auth-chain: entity-number})."""
    with gzip.open(path, "rt", errors="replace") as handle:
        cif = MMCIF2Dict(handle)

    pdb_id = path.name[:-7].upper()  # remove .cif.gz
    entity_numbers = as_list(cif.get("_entity_poly.entity_id"))
    entity_sequences = as_list(cif.get("_entity_poly.pdbx_seq_one_letter_code_can"))
    sequence_by_entity = {
        str(entity): canonical_sequence(str(sequence))
        for entity, sequence in zip(entity_numbers, entity_sequences)
    }

    scheme_entities = as_list(cif.get("_pdbx_poly_seq_scheme.entity_id"))
    scheme_auth_chains = as_list(cif.get("_pdbx_poly_seq_scheme.pdb_strand_id"))
    if len(scheme_entities) != len(scheme_auth_chains):
        raise ValueError("pdbx_poly_seq_scheme columns have different lengths")

    chain_to_entity: dict[str, str] = {}
    for entity_number, raw_chains in zip(scheme_entities, scheme_auth_chains):
        for auth_chain in str(raw_chains).split(","):
            auth_chain = auth_chain.strip()
            if not auth_chain or auth_chain in {".", "?"}:
                continue
            previous = chain_to_entity.setdefault(auth_chain, str(entity_number))
            if previous != str(entity_number):
                raise ValueError(
                    f"author chain {auth_chain!r} maps to entities {previous} and {entity_number}"
                )

    sequences: dict[str, str] = {}
    retained_map: dict[str, str] = {}
    for auth_chain, entity_number in chain_to_entity.items():
        entity_id = f"{pdb_id}_{entity_number}"
        if entity_id not in allowed_entities:
            continue
        sequence = sequence_by_entity.get(entity_number, "")
        if not sequence:
            raise ValueError(f"missing canonical sequence for plant entity {entity_id}")
        protein_id = f"{pdb_id}_{auth_chain}"
        sequences[protein_id] = sequence
        retained_map[auth_chain] = entity_number
    return sequences, retained_map


def read_sifts(
    path: Path,
    eligible_ids: set[str],
    evidence_codes: set[str],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    direct: dict[str, set[str]] = defaultdict(set)
    counts = defaultdict(int)
    with path.open(newline="") as handle:
        rows = (line for line in handle if not line.startswith("#"))
        reader = csv.DictReader(rows, delimiter="\t")
        required = {"PDB", "CHAIN", "EVIDENCE", "GO_ID"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected SIFTS columns: {reader.fieldnames}")
        for row in reader:
            counts["sifts_rows"] += 1
            evidence = row["EVIDENCE"].strip().upper()
            if evidence not in evidence_codes:
                continue
            counts["experimental_rows"] += 1
            protein_id = f"{row['PDB'].strip().upper()}_{row['CHAIN'].strip()}"
            if protein_id not in eligible_ids:
                continue
            go_id = row["GO_ID"].strip()
            if go_id:
                direct[protein_id].add(go_id)
                counts["eligible_rows"] += 1
    return dict(direct), dict(counts)


def propagate_annotations(
    direct: dict[str, set[str]],
    graph: nx.MultiDiGraph,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, str]]:
    annotations: dict[str, dict[str, list[str]]] = {}
    term_names: dict[str, str] = {}
    roots = set(ONTOLOGY_ROOTS.values())

    for protein_id, terms in direct.items():
        by_ontology: dict[str, set[str]] = {ontology: set() for ontology in ONTOLOGIES}
        for term in terms:
            if term not in graph:
                continue
            propagated = nx.descendants(graph, term) | {term}
            for propagated_term in propagated - roots:
                namespace = graph.nodes[propagated_term].get("namespace")
                if namespace in by_ontology:
                    by_ontology[namespace].add(propagated_term)
                    term_names[propagated_term] = graph.nodes[propagated_term].get(
                        "name", propagated_term
                    )
        if any(by_ontology.values()):
            annotations[protein_id] = {
                ontology: sorted(by_ontology[ontology]) for ontology in ONTOLOGIES
            }
    return annotations, term_names


def write_legacy_annotation_tsv(
    path: Path,
    annotations: dict[str, dict[str, list[str]]],
    term_names: dict[str, str],
) -> None:
    vocabularies = {
        ontology: sorted({term for values in annotations.values() for term in values[ontology]})
        for ontology in ONTOLOGIES
    }
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for ontology in ONTOLOGIES:
            terms = vocabularies[ontology]
            writer.writerow([f"### GO-terms ({ontology})"])
            writer.writerow(terms)
            writer.writerow([f"### GO-names ({ontology})"])
            writer.writerow([term_names.get(term, term) for term in terms])
        writer.writerow([
            "### PDB-chain",
            "GO-terms (molecular_function)",
            "GO-terms (biological_process)",
            "GO-terms (cellular_component)",
        ])
        for protein_id in sorted(annotations):
            writer.writerow([
                protein_id,
                ",".join(annotations[protein_id]["molecular_function"]),
                ",".join(annotations[protein_id]["biological_process"]),
                ",".join(annotations[protein_id]["cellular_component"]),
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--min-length", type=int, default=60)
    parser.add_argument(
        "--evidence",
        default=",".join(sorted(EXPERIMENTAL_EVIDENCE)),
        help="Comma-separated GO evidence codes retained from SIFTS.",
    )
    parser.add_argument("--allow-parse-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    structure_dir = data_dir / "structure_files"
    entities_path = data_dir / "viridiplantae_entities.txt"
    sifts_path = data_dir / "pdb_chain_go.tsv"
    go_path = data_dir / "go-basic.obo"
    for required in (entities_path, sifts_path, go_path):
        if not required.exists():
            raise SystemExit(f"Missing required input: {required}")

    allowed_entities = {
        line.strip().upper() for line in entities_path.read_text().splitlines() if line.strip()
    }
    cif_files = sorted(structure_dir.glob("*.cif.gz"))
    if not cif_files:
        raise SystemExit(f"No CIF files found in {structure_dir}")

    all_sequences: dict[str, str] = {}
    entity_map: dict[str, dict[str, str]] = {}
    parse_failures: dict[str, str] = {}
    for index, cif_path in enumerate(cif_files, start=1):
        try:
            sequences, chain_map = parse_cif_sequences(cif_path, allowed_entities)
            overlap = set(all_sequences) & set(sequences)
            if overlap:
                raise ValueError(f"duplicate protein IDs: {sorted(overlap)[:3]}")
            all_sequences.update(sequences)
            if chain_map:
                entity_map[cif_path.name[:-7].upper()] = chain_map
        except Exception as exc:
            parse_failures[cif_path.name] = str(exc)
        if index % 250 == 0 or index == len(cif_files):
            print(
                f"Parsed {index:,}/{len(cif_files):,} CIFs; "
                f"plant chains={len(all_sequences):,}; failures={len(parse_failures):,}"
            )

    length_eligible = {
        protein_id: sequence
        for protein_id, sequence in all_sequences.items()
        if len(sequence) > args.min_length
    }
    evidence_codes = {code.strip().upper() for code in args.evidence.split(",") if code.strip()}
    direct, sifts_counts = read_sifts(sifts_path, set(length_eligible), evidence_codes)
    print(f"Loading GO graph from {go_path}")
    graph = obonet.read_obo(go_path)
    annotations, term_names = propagate_annotations(direct, graph)

    final_sequences = {
        protein_id: length_eligible[protein_id] for protein_id in annotations
    }
    records = {
        protein_id: {
            "sequence": final_sequences[protein_id],
            "length": len(final_sequences[protein_id]),
            "annotations": annotations[protein_id],
            "entity_id": (
                f"{protein_id.split('_', 1)[0]}_"
                f"{entity_map[protein_id.split('_', 1)[0]][protein_id.split('_', 1)[1]]}"
            ),
        }
        for protein_id in sorted(final_sequences)
    }

    write_fasta(final_sequences, data_dir / "all_sequences.fasta")
    with (data_dir / "protein_records.pkl").open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (data_dir / "entity_map.json").write_text(json.dumps(entity_map, sort_keys=True) + "\n")
    (data_dir / "go_term_names.json").write_text(
        json.dumps(term_names, indent=2, sort_keys=True) + "\n"
    )
    write_legacy_annotation_tsv(data_dir / "pdb2go.tsv", annotations, term_names)

    summary = {
        "schema_version": 1,
        "minimum_sequence_length_exclusive": args.min_length,
        "retained_evidence_codes": sorted(evidence_codes),
        "counts": {
            "rcsb_plant_entities": len(allowed_entities),
            "cif_files": len(cif_files),
            "cif_parse_failures": len(parse_failures),
            "plant_author_chains": len(all_sequences),
            "chains_longer_than_minimum": len(length_eligible),
            "chains_with_retained_go_annotations": len(records),
            **sifts_counts,
        },
        "ontology_protein_counts": {
            ontology: sum(bool(record["annotations"][ontology]) for record in records.values())
            for ontology in ONTOLOGIES
        },
        "parse_failures": parse_failures,
        "sha256": {
            "all_sequences": sha256(data_dir / "all_sequences.fasta"),
            "protein_records": sha256(data_dir / "protein_records.pkl"),
            "pdb2go": sha256(data_dir / "pdb2go.tsv"),
        },
    }
    (data_dir / "preprocessing_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["counts"], indent=2))

    if parse_failures and not args.allow_parse_failures:
        raise SystemExit(
            f"ERROR: {len(parse_failures)} CIF files could not be parsed; "
            "see preprocessing_manifest.json. Re-run only with --allow-parse-failures "
            "after documenting the exclusions."
        )


if __name__ == "__main__":
    main()
