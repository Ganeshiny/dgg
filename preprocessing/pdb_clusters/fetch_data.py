#!/usr/bin/env python3
"""Retrieve all external inputs needed by the PDB-cluster pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from common import DATA_DIR, PDB_CLUSTER_DIR, STRUCTURE_DIR, THRESHOLDS, ensure_directories, sha256


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GO_URL = "https://purl.obolibrary.org/obo/go/go-basic.obo"
SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_go.tsv.gz"
CLUSTER_URL = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{threshold}.txt"
CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


def atomic_download(session: requests.Session, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with session.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded empty file from {url}")
    temporary.replace(destination)


def query_viridiplantae_entities(session: requests.Session) -> tuple[list[str], dict]:
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.taxonomy_lineage.id",
                "operator": "exact_match",
                "value": "33090",
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "return_all_hits": True,
            "results_content_type": ["experimental"],
        },
    }
    response = session.post(RCSB_SEARCH_URL, json=query, timeout=(30, 300))
    response.raise_for_status()
    payload = response.json()
    entities = sorted({row["identifier"].upper() for row in payload.get("result_set", [])})
    if not entities:
        raise RuntimeError("RCSB query returned no Viridiplantae polymer entities")
    return entities, payload


def download_structure(pdb_id: str, structure_dir: Path) -> tuple[str, str | None]:
    destination = structure_dir / f"{pdb_id}.cif.gz"
    if destination.exists() and destination.stat().st_size > 0:
        return pdb_id, None
    session = requests.Session()
    session.headers["User-Agent"] = "DeepGreenGO-reproducible-preprocessing/1.0"
    try:
        atomic_download(session, CIF_URL.format(pdb_id=pdb_id), destination)
        return pdb_id, None
    except Exception as exc:  # keep the complete failure list for the validation gate
        destination.unlink(missing_ok=True)
        return pdb_id, str(exc)
    finally:
        session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--reuse-metadata", action="store_true",
        help="Reuse existing GO, SIFTS, RCSB query, and cluster files instead of refreshing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    cluster_dir = data_dir / "pdb_clusters"
    structure_dir = data_dir / "structure_files"
    for directory in (data_dir, cluster_dir, structure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "DeepGreenGO-reproducible-preprocessing/1.0"
    retrieved_at = datetime.now(timezone.utc).isoformat()

    go_path = data_dir / "go-basic.obo"
    sifts_gz = data_dir / "pdb_chain_go.tsv.gz"
    sifts_tsv = data_dir / "pdb_chain_go.tsv"
    query_json = data_dir / "viridiplantae_rcsb_query.json"
    entities_path = data_dir / "viridiplantae_entities.txt"
    pdb_ids_path = data_dir / "viridiplantae_pdb_ids.csv"

    if not args.reuse_metadata or not go_path.exists():
        print(f"Downloading GO ontology: {GO_URL}")
        atomic_download(session, GO_URL, go_path)

    if not args.reuse_metadata or not sifts_gz.exists():
        print(f"Downloading SIFTS annotations: {SIFTS_URL}")
        atomic_download(session, SIFTS_URL, sifts_gz)
    with gzip.open(sifts_gz, "rb") as source, sifts_tsv.with_suffix(".tsv.part").open("wb") as target:
        shutil.copyfileobj(source, target)
    sifts_tsv.with_suffix(".tsv.part").replace(sifts_tsv)

    if args.reuse_metadata and entities_path.exists() and query_json.exists():
        entities = [line.strip() for line in entities_path.read_text().splitlines() if line.strip()]
    else:
        print("Querying RCSB for Viridiplantae polymer entities (NCBI taxonomy lineage 33090)")
        entities, query_payload = query_viridiplantae_entities(session)
        query_json.write_text(json.dumps(query_payload, indent=2))
        entities_path.write_text("".join(f"{entity}\n" for entity in entities))

    pdb_ids = sorted({entity.rsplit("_", 1)[0] for entity in entities})
    pdb_ids_path.write_text(",".join(pdb_ids) + "\n")

    cluster_paths: dict[int, Path] = {}
    for threshold in THRESHOLDS:
        path = cluster_dir / f"clusters-by-entity-{threshold}.txt"
        cluster_paths[threshold] = path
        if not args.reuse_metadata or not path.exists():
            url = CLUSTER_URL.format(threshold=threshold)
            print(f"Downloading {threshold}% PDB entity clusters: {url}")
            atomic_download(session, url, path)

    print(f"Downloading/checking {len(pdb_ids):,} PDB structure files with {args.workers} workers")
    failures: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(download_structure, pdb_id, structure_dir) for pdb_id in pdb_ids]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            pdb_id, error = future.result()
            if error:
                failures[pdb_id] = error
            if index % 250 == 0 or index == len(futures):
                print(f"  checked {index:,}/{len(futures):,}; failures={len(failures):,}")

    manifest = {
        "schema_version": 1,
        "retrieved_at_utc": retrieved_at,
        "taxonomy_lineage_id": 33090,
        "sources": {
            "rcsb_search": RCSB_SEARCH_URL,
            "go_basic": GO_URL,
            "sifts_pdb_chain_go": SIFTS_URL,
            "pdb_entity_clusters": CLUSTER_URL,
            "pdb_cif": CIF_URL,
        },
        "counts": {
            "polymer_entities": len(entities),
            "pdb_entries": len(pdb_ids),
            "structure_download_failures": len(failures),
        },
        "failed_structures": failures,
        "sha256": {
            "go_basic": sha256(go_path),
            "sifts_gz": sha256(sifts_gz),
            "rcsb_query": sha256(query_json),
            **{f"clusters_{threshold}": sha256(path) for threshold, path in cluster_paths.items()},
        },
    }
    (data_dir / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    session.close()

    if failures:
        print(f"ERROR: {len(failures)} structure downloads failed; see retrieval_manifest.json", file=sys.stderr)
        raise SystemExit(1)
    print("Data retrieval completed and checksummed.")


if __name__ == "__main__":
    main()
