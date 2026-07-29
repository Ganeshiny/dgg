#!/usr/bin/env python3
"""Export the union of ARC test proteins as SPROF-GO-compatible two-line FASTA."""
from __future__ import annotations
import argparse, json
from pathlib import Path

from pickle_compat import load_pickle_compat

ONTOLOGIES = ("molecular_function", "biological_process", "cellular_component")

def read_fasta(path):
    seqs, key = {}, None
    for raw in Path(path).read_text().splitlines():
        if raw.startswith(">"):
            key = raw[1:].split()[0]; seqs[key] = ""
        elif key:
            seqs[key] += raw.strip()
    return seqs

def dataset_ids(obj):
    if isinstance(obj, (list, tuple)):
        ids = []
        for index, record in enumerate(obj):
            if isinstance(record, dict):
                value = next(
                    (
                        record.get(name)
                        for name in ("protein_id", "id", "protein", "accession")
                        if record.get(name) is not None
                    ),
                    None,
                )
            else:
                value = record[0] if isinstance(record, (list, tuple)) and record else None
            if value is None:
                raise TypeError(f"Cannot find protein ID in record {index}: {record!r}")
            ids.append(str(value))
        return ids
    for name in ("protein_ids", "ids", "proteins", "accessions"):
        value = getattr(obj, name, None)
        if value is not None and not isinstance(value, dict): return [str(x) for x in value]
        if isinstance(obj, dict) and name in obj: return [str(x) for x in obj[name]]
    if hasattr(obj, "columns"):
        for name in ("protein_id", "protein", "id", "accession"):
            if name in obj.columns: return [str(x) for x in obj[name].tolist()]
    for name in ("records", "samples", "data"):
        value = getattr(obj, name, None)
        if value is not None:
            try:
                return [str(x.get("protein_id", x.get("id"))) if isinstance(x, dict) else str(x[0]) for x in value]
            except Exception: pass
    raise TypeError(f"Cannot find protein IDs in {type(obj).__name__}; keys={getattr(obj, '__dict__', {}).keys()}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tuning-root", type=Path, required=True)
    p.add_argument("--all-fasta", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    a = p.parse_args()
    ids = set()
    counts = {}
    for ont in ONTOLOGIES:
        path = a.tuning_root / "datasets" / f"{ont}_test.pkl"
        with path.open("rb") as handle: obj = load_pickle_compat(handle)
        found = dataset_ids(obj); counts[ont] = len(found); ids.update(found)
    seqs = read_fasta(a.all_fasta)
    missing = sorted(ids - seqs.keys())
    if missing: raise SystemExit(f"{len(missing)} test IDs absent from FASTA; first: {missing[:10]}")
    # SPROF-GO's parser requires one sequence line per header and modifies pipe IDs.
    if any("|" in x or " " in x for x in ids):
        raise SystemExit("SPROF-GO-unsafe test IDs contain spaces or pipes")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w") as out:
        for pid in sorted(ids): out.write(f">{pid}\n{seqs[pid]}\n")
    manifest = a.manifest or a.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"test_counts": counts, "union_count": len(ids), "fasta": str(a.output)}, indent=2)+"\n")
    print(f"Wrote {len(ids)} unique test proteins to {a.output}")

if __name__ == "__main__": main()
