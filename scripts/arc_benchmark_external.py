#!/usr/bin/env python3
"""Headless helpers for external annotation systems used by the ARC benchmark."""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from arc_benchmark import ONTOLOGIES, load_label_npz
from arc_benchmark_methods import run_checked


def make_dpfunc_manifest(args) -> None:
    workspace = args.workspace.resolve()
    interpro = defaultdict(set)
    path = args.interpro_tsv.resolve()
    with path.open() as handle:
        for raw in handle:
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if not fields:
                continue
            for accession in re.findall(r"IPR\d+", raw):
                interpro[fields[0]].add(accession)
    rows = []
    for split in ("valid", "test"):
        protein_ids, _, _ = load_label_npz(workspace, "mf", split)
        for protein in protein_ids:
            pdb_file = workspace / "inputs" / "structures" / split / f"{protein}.pdb"
            if not pdb_file.is_file():
                raise FileNotFoundError(f"DPFunc input structure is missing: {pdb_file}")
            rows.append({
                "protein_id": protein,
                "split": "predict",
                "pdb_file": str(pdb_file.resolve()),
                "chain_id": "A",
                "interpro_terms": ";".join(sorted(interpro.get(protein, ()))),
                "go_terms": "",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote DPFunc manifest with {len(rows)} proteins: {args.output}")


def clone_dpfunc_workspace(args) -> None:
    """Reuse ontology-independent DPFunc ESM and graph features via hard links."""
    source = args.source.resolve()
    target = args.target.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source DPFunc workspace is missing: {source}")
    target.mkdir(parents=True, exist_ok=True)

    def link_or_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            return
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    for src in source.rglob("*"):
        relative = src.relative_to(source)
        dst = target / relative
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.is_symlink():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() and not dst.is_symlink():
                dst.symlink_to(os.readlink(src))
        else:
            link_or_copy(src, dst)

    source_token = args.source_ontology + "_"
    for src in list(target.rglob(f"{args.source_ontology}_*")):
        if src.is_file() and src.name.startswith(source_token):
            dst = src.with_name(args.target_ontology + "_" + src.name[len(source_token):])
            link_or_copy(src, dst)

    source_config = source / "configure" / f"{args.source_ontology}.yaml"
    target_config = target / "configure" / f"{args.target_ontology}.yaml"
    target_config.write_text(
        source_config.read_text().replace(args.source_ontology, args.target_ontology)
    )
    mlb_target = target / "mlb" / f"{args.target_ontology}_go.mlb"
    shutil.copy2(args.mlb_source.resolve(), mlb_target)
    print(
        f"Cloned DPFunc {args.source_ontology} workspace to "
        f"{args.target_ontology} using shared hard-linked features: {target}"
    )


def run_hayai(args) -> None:
    """Run the non-interactive functional-annotation portion of Hayai v3.2.

    This mirrors the upstream Shiny application's DIAMOND call and, when
    available, its Viridiplantae ODB-mapper fallback.  It does not launch the
    GUI or produce Hayai's visualization-only files.
    """
    repo = args.hayai_root.resolve()
    workspace = args.workspace.resolve()
    raw_dir = workspace / "raw" / "hayai"
    raw_dir.mkdir(parents=True, exist_ok=True)
    hits_path = raw_dir / "diamond_hits.tsv"
    if not hits_path.is_file():
        run_checked([
            args.diamond, "blastp", "-q", workspace / "inputs" / "valid_test.fasta",
            "-d", repo / "db" / "zen.dmnd", "-o", hits_path,
            "--threads", args.threads, "--" + args.sensitivity, "--top", "1",
            "--evalue", "1e-6", "--quiet", "-f", "6", "qseqid", "pident",
            "length", "qstart", "qend", "sstart", "send", "evalue", "bitscore", "stitle",
        ])
    diamond_rows = {}
    with hits_path.open() as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t", 9)
            if len(fields) < 10:
                continue
            query = fields[0]
            title = fields[9].split("|")
            title += [""] * (10 - len(title))
            def clean(index, prefix):
                return title[index].strip().removeprefix(prefix).strip()
            diamond_rows[query] = {
                "Query": query,
                "Accession": clean(0, "Accession:"),
                "Product_Name": clean(1, "Product_Name:"),
                "OrthoDB": clean(2, "Zen_OrthoDB:"),
                "Evidence_existence": clean(4, "PE:"),
                "InterPro": clean(5, "InterPro:"),
                "Pfam": clean(6, "Pfam:"),
                "GO_BP": clean(7, "GO_BP:"),
                "GO_MF": clean(8, "GO_MF:"),
                "GO_CC": clean(9, "GO_CC:"),
                "Identity": fields[1],
                "Length": fields[2],
                "Evalue": fields[7],
                "Score": fields[8],
            }

    if args.orthologer:
        project_name = "dggbenchmark"
        log_path = raw_dir / "orthologer.log"
        command = [args.odb_mapper, "MAP", project_name,
                   workspace / "inputs" / "valid_test.fasta", "33090"]
        print("RUN " + " ".join(map(str, command)), flush=True)
        with log_path.open("w") as log_handle:
            subprocess.run([str(value) for value in command], cwd=repo,
                           stdout=log_handle, stderr=subprocess.STDOUT, check=True)
        annotation = repo / "odbmapper" / "v12" / "pipeline" / "Results" / f"{project_name}.og.annotations"
        if not annotation.is_file():
            raise FileNotFoundError(f"Hayai ODB-mapper output missing: {annotation}")
        with annotation.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                query = row.get("Query") or row.get("query") or next(iter(row.values()))
                target = diamond_rows.setdefault(query, {"Query": query})
                if not target.get("GO_MF"):
                    target["GO_MF"] = row.get("ODB_GO_MF", row.get("GO_MF", ""))
                if not target.get("GO_BP"):
                    target["GO_BP"] = row.get("ODB_GO_BP", row.get("GO_BP", ""))
                if row.get("ODB_OG"):
                    target["OrthoDB"] = row["ODB_OG"]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Query", "Accession", "Product_Name", "OrthoDB", "GO_BP", "GO_MF", "GO_CC",
              "Evidence_existence", "InterPro", "Pfam", "Identity", "Length", "Evalue", "Score"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for protein in sorted(diamond_rows):
            writer.writerow(diamond_rows[protein])
    print(f"Wrote headless Hayai annotations for {len(diamond_rows)} proteins: {output}")

