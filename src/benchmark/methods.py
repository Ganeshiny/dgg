#!/usr/bin/env python3
"""Runnable baselines and external-output adapters for the ARC benchmark."""

from __future__ import annotations

import csv
import gzip
import json
import re
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import (
    ONTOLOGIES,
    ROOT_TERMS,
    ancestors,
    load_label_npz,
    normalize_term,
    parse_obo,
    read_fasta,
    save_prediction,
)


def run_checked(command: list[str | Path], *, cwd: Path | None = None) -> None:
    print("RUN " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def run_search_once(output: Path, build_command, *, force: bool = False) -> None:
    """Run a similarity search unless a *complete* result already exists.

    A bare ``output.is_file()`` resume check is unsafe here. blastp, DIAMOND
    and Foldseek all create their ``-out``/``--out`` file as soon as they
    start, so a run killed by the Slurm wall clock leaves a zero-byte or
    truncated file behind. On the next resume that file looks exactly like a
    finished search: the tool is skipped, the label transfer reads almost
    nothing, all-zero predictions are written, and the stage `.done` marker
    then locks the empty result in permanently.

    That is precisely how BLAST came to report CAFA Fmax 0.000 with 0%
    coverage while its own hit file held 7,108 alignments that re-transfer to
    2,619 nonzero scores (docs/figure_data_integrity.md).

    The search therefore writes to a temporary path and is renamed into place
    only after the process exits successfully, so the final filename can never
    name a partial result. os.replace is atomic within a filesystem.
    """
    if force and output.exists():
        print(f"[benchmark] forcing search refresh for {output}", flush=True)
        output.unlink()
    if output.is_file() and output.stat().st_size > 0:
        print(f"[benchmark] reusing complete search output {output}", flush=True)
        return
    if output.is_file():
        print(f"[benchmark] {output} exists but is empty — discarding and re-running", flush=True)
        output.unlink()
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        partial.unlink()
    run_checked(build_command(partial))
    if not partial.is_file():
        raise RuntimeError(f"search did not produce {partial}")
    if partial.stat().st_size == 0:
        # Exit code 0 with no rows is possible in principle, but against a
        # 6,026-sequence database it means something is wrong upstream.
        raise RuntimeError(
            f"search wrote an empty {partial}; refusing to record a zero-coverage "
            "result as if it were a real one")
    os.replace(partial, output)


def canonicalize_similarity_id(raw_id: str) -> str:
    """Map search-tool sequence identifiers back to locked ARC protein IDs.

    ``makeblastdb -parse_seqids`` recognizes identifiers such as ``7FHK_A`` as
    PDB accessions and emits them as ``pdb|7FHK|A``. DIAMOND and the locked
    datasets retain ``7FHK_A``. Normalize both forms before label transfer so
    a BLAST hit is not silently discarded due to formatting alone.
    """
    value = raw_id.strip()
    pdb_match = re.fullmatch(r"pdb\|([^|]+)\|(.+)", value, flags=re.IGNORECASE)
    if pdb_match:
        return f"{pdb_match.group(1)}_{pdb_match.group(2)}"

    # Foldseek commonly emits a structure filename rather than a FASTA ID.
    name = Path(value).name
    if name.endswith(".pdb.gz"):
        return name[:-7]
    if name.endswith(".pdb"):
        return name[:-4]
    return Path(name).stem


def parse_similarity_hits(path: Path):
    """Return query -> (target, bits, identity, qcov, tcov) hits."""
    hits = defaultdict(list)
    with path.open() as handle:
        for raw in handle:
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            query, target = fields[0], fields[1]
            identity = float(fields[2])
            aligned = float(fields[3])
            qlen, tlen = max(float(fields[4]), 1.0), max(float(fields[5]), 1.0)
            evalue, bits = float(fields[6]), float(fields[7])
            qcov = float(fields[8]) if len(fields) >= 9 else aligned / qlen
            tcov = float(fields[9]) if len(fields) >= 10 else aligned / tlen
            if qcov > 1:
                qcov /= 100.0
            if tcov > 1:
                tcov /= 100.0
            if evalue <= 1e-3:
                hits[canonicalize_similarity_id(query)].append(
                    (canonicalize_similarity_id(target), bits, identity, qcov, tcov)
                )
    return hits


def transfer_scores(query_ids, train_ids, train_labels, hits, top_k, min_qcov, min_tcov):
    train_index = {protein_id: index for index, protein_id in enumerate(train_ids)}
    scores = np.zeros((len(query_ids), train_labels.shape[1]), dtype=np.float32)
    for query_index, query in enumerate(query_ids):
        selected = [
            hit for hit in hits.get(query, [])
            if hit[0] in train_index and hit[3] >= min_qcov and hit[4] >= min_tcov
        ]
        selected.sort(key=lambda value: value[1], reverse=True)
        selected = selected[:top_k]
        if not selected:
            continue
        weights = np.asarray([max(value[1], 1e-8) for value in selected], dtype=np.float64)
        rows = np.asarray([train_labels[train_index[value[0]]] for value in selected])
        scores[query_index] = np.average(rows, axis=0, weights=weights)
    return scores


def best_identity_transfer(
    query_ids, train_ids, train_labels, hits, top_k, min_qcov, min_tcov
):
    """Transfer annotations from exactly one eligible highest-identity hit.

    The previous implementation took the union of annotations over every hit
    and assigned each term the maximum identity among its supporting hits. That
    was not a single-best-hit baseline and could cover more terms than top-k.
    """
    train_index = {protein_id: index for index, protein_id in enumerate(train_ids)}
    scores = np.zeros((len(query_ids), train_labels.shape[1]), dtype=np.float32)
    for query_index, query in enumerate(query_ids):
        eligible = [
            hit for hit in hits.get(query, ())
            if hit[0] in train_index and hit[3] >= min_qcov and hit[4] >= min_tcov
        ]
        eligible.sort(key=lambda hit: hit[1], reverse=True)
        selected = eligible[:top_k]
        if not selected:
            continue
        target, _, identity, _, _ = max(
            selected, key=lambda hit: (hit[2], hit[1])
        )
        similarity = identity / 100.0 if identity > 1.0 else identity
        annotated = train_labels[train_index[target]] > 0
        scores[query_index, annotated] = np.clip(similarity, 0.0, 1.0)
    return scores


def sequence_baselines(args) -> None:
    workspace = args.workspace.resolve()
    input_dir, db_dir, raw_dir = workspace / "inputs", workspace / "databases", workspace / "raw"
    db_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    query_ids = list(read_fasta(input_dir / "valid_test.fasta"))
    methods = {value.strip() for value in args.methods.split(",") if value.strip()}
    hit_files = {}
    if "blast" in methods:
        db = db_dir / "blast_train"
        if not Path(str(db) + ".pin").is_file():
            run_checked([args.makeblastdb, "-in", input_dir / "train.fasta", "-dbtype", "prot",
                         "-parse_seqids", "-out", db])
        hits = raw_dir / "blast_hits.tsv"
        run_search_once(hits, lambda out: [
            args.blastp, "-query", input_dir / "valid_test.fasta", "-db", db,
            "-out", out, "-evalue", "1e-3", "-max_target_seqs", args.max_hits,
            "-num_threads", args.threads, "-outfmt",
            "6 qseqid sseqid pident length qlen slen evalue bitscore qcovs"])
        hit_files["blast"] = hits
    if "diamond" in methods:
        db = db_dir / "diamond_train"
        if not Path(str(db) + ".dmnd").is_file():
            run_checked([args.diamond, "makedb", "--in", input_dir / "train.fasta", "--db", db])
        hits = raw_dir / "diamond_hits.tsv"
        run_search_once(hits, lambda out: [
            args.diamond, "blastp", "--query", input_dir / "valid_test.fasta",
            "--db", db, "--out", out, "--evalue", "1e-3",
            "--max-target-seqs", args.max_hits, "--threads", args.threads,
            "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "qlen",
            "slen", "evalue", "bitscore", "qcovhsp"])
        hit_files["diamond"] = hits

    for short in ONTOLOGIES:
        train_ids, terms, train_labels = load_label_npz(workspace, short, "train")
        valid_ids, valid_terms, _ = load_label_npz(workspace, short, "valid")
        test_ids, test_terms, _ = load_label_npz(workspace, short, "test")
        if terms != valid_terms or terms != test_terms:
            raise ValueError(f"{short}: GO vocabularies differ between splits")
        all_ids = valid_ids + test_ids
        if set(all_ids) != set(query_ids):
            raise ValueError("Locked validation/test IDs disagree with the query FASTA")
        prevalence = train_labels.mean(axis=0, dtype=np.float64)
        for split_name, ids in (("valid", valid_ids), ("test", test_ids)):
            method = "naive_valid" if split_name == "valid" else "naive"
            scores = np.repeat(prevalence[None, :], len(ids), axis=0)
            save_prediction(workspace, method, short, ids, terms, scores,
                            metadata={"source": "locked training-label prevalence", "split": split_name})
        for method, hit_file in hit_files.items():
            parsed_hits = parse_similarity_hits(hit_file)
            scores = transfer_scores(all_ids, train_ids, train_labels, parsed_hits,
                                     args.top_k, args.min_qcov, args.min_tcov)
            meta = {
                "transfer": "bitscore-weighted top-k training-label transfer",
                "top_k": args.top_k,
                "minimum_query_coverage": args.min_qcov,
                "minimum_target_coverage": args.min_tcov,
                "database": "locked threshold-30 training proteins only",
                "raw_hits": str(hit_file),
            }
            save_prediction(workspace, method + "_valid", short, valid_ids, terms,
                            scores[:len(valid_ids)], metadata={**meta, "split": "valid"})
            save_prediction(workspace, method, short, test_ids, terms,
                            scores[len(valid_ids):], metadata={**meta, "split": "test"})

            max_scores = best_identity_transfer(
                all_ids, train_ids, train_labels, parsed_hits,
                args.top_k, args.min_qcov, args.min_tcov
            )
            max_meta = {
                **meta,
                "transfer": "annotations from one highest-identity hit within the eligible top-k pool",
            }
            save_prediction(workspace, method + "_max_valid", short, valid_ids, terms,
                            max_scores[:len(valid_ids)], metadata={**max_meta, "split": "valid"})
            save_prediction(workspace, method + "_max", short, test_ids, terms,
                            max_scores[len(valid_ids):], metadata={**max_meta, "split": "test"})


def foldseek_baseline(args) -> None:
    workspace = args.workspace.resolve()
    raw_dir = workspace / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    hits_path = raw_dir / "foldseek_hits.tsv"
    structure_root = workspace / "inputs" / "structures" / "foldseek"
    query_dir = structure_root / "valid_test"
    target_dir = structure_root / "train"
    tmp_dir = workspace / "tmp" / "foldseek"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_search_once(hits_path, lambda out: [
        args.foldseek, "easy-search", query_dir, target_dir, out, tmp_dir,
        "--threads", args.threads, "--max-seqs", args.max_hits,
        "--format-output", "query,target,fident,alnlen,qlen,tlen,evalue,bits,qcov,tcov"],
        force=args.force_search)
    parsed = parse_similarity_hits(hits_path)
    for short in ONTOLOGIES:
        train_ids, terms, train_labels = load_label_npz(workspace, short, "train")
        valid_ids, _, _ = load_label_npz(workspace, short, "valid")
        test_ids, _, _ = load_label_npz(workspace, short, "test")
        all_ids = valid_ids + test_ids
        scores = transfer_scores(all_ids, train_ids, train_labels, parsed,
                                 args.top_k, args.min_qcov, args.min_tcov)
        meta = {"transfer": "bitscore-weighted top-k structural-neighbour transfer",
                "database": "locked threshold-30 training experimental structures only",
                "raw_hits": str(hits_path)}
        save_prediction(workspace, "foldseek_valid", short, valid_ids, terms,
                        scores[:len(valid_ids)], metadata={**meta, "split": "valid"})
        save_prediction(workspace, "foldseek", short, test_ids, terms,
                        scores[len(valid_ids):], metadata={**meta, "split": "test"})
        max_scores = best_identity_transfer(
            all_ids, train_ids, train_labels, parsed,
            args.top_k, args.min_qcov, args.min_tcov
        )
        max_meta = {
            **meta,
            "transfer": "annotations from one highest-identity structural hit within the eligible top-k pool",
        }
        save_prediction(workspace, "foldseek_max_valid", short, valid_ids, terms,
                        max_scores[:len(valid_ids)], metadata={**max_meta, "split": "valid"})
        save_prediction(workspace, "foldseek_max", short, test_ids, terms,
                        max_scores[len(valid_ids):], metadata={**max_meta, "split": "test"})


def empty_query_matrix(workspace: Path, ontology: str):
    valid_ids, terms, _ = load_label_npz(workspace, ontology, "valid")
    test_ids, test_terms, _ = load_label_npz(workspace, ontology, "test")
    if terms != test_terms:
        raise ValueError(f"{ontology}: validation/test GO vocabularies differ")
    return valid_ids + test_ids, terms, np.zeros((len(valid_ids) + len(test_ids), len(terms)), np.float32), len(valid_ids)


def benchmark_obo(workspace: Path) -> Path:
    manifest = json.loads((workspace / "benchmark_manifest.json").read_text())
    return Path(manifest["data_root"]) / "go-basic.obo"


def normalize_scored_rows(workspace: Path, method: str, files: dict[str, Path],
                          delimiter: str | None = None, score_type: str = "continuous",
                          score_column: int = 2) -> None:
    parents, aliases = parse_obo(benchmark_obo(workspace))
    cache = {}
    for short in ONTOLOGIES:
        ids, terms, scores, valid_count = empty_query_matrix(workspace, short)
        id_index, term_index = {v: i for i, v in enumerate(ids)}, {v: i for i, v in enumerate(terms)}
        source_rows = mapped = 0
        path = files[short]
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            for raw in handle:
                fields = raw.strip().split(delimiter)
                if len(fields) <= score_column:
                    continue
                protein = Path(fields[0]).stem.replace(".pdb", "")
                if protein not in id_index:
                    continue
                term = normalize_term(fields[1], aliases)
                try:
                    score = float(fields[score_column])
                except ValueError:
                    continue
                if term is None:
                    continue
                source_rows += 1
                for propagated in ancestors(term, parents, cache):
                    if propagated in term_index and propagated != ROOT_TERMS[short]:
                        row, col = id_index[protein], term_index[propagated]
                        scores[row, col] = max(scores[row, col], score)
                        mapped += 1
        valid_ids, _, _ = load_label_npz(workspace, short, "valid")
        test_ids, _, _ = load_label_npz(workspace, short, "test")
        meta = {"source_file": str(path), "source_rows": source_rows,
                "mapped_assignments": mapped, "ancestor_propagation": True}
        save_prediction(workspace, method + "_valid", short, valid_ids, terms, scores[:valid_count],
                        score_type=score_type, metadata={**meta, "split": "valid"})
        save_prediction(workspace, method, short, test_ids, terms, scores[valid_count:],
                        score_type=score_type, metadata={**meta, "split": "test"})


def normalize_deepgoplus(workspace: Path, path: Path) -> None:
    """Normalize DeepGOPlus's protein<TAB>GO|score wide-row output.

    DeepGOPlus 1.0.2 does not emit one three-column record per prediction.
    Each line starts with the protein ID and is followed by any number of
    GO:nnnnnnn|score fields. Parsing it as a generic scored table silently
    produces zero predictions.
    """
    terms_by_ontology = {
        short: set(load_label_npz(workspace, short, "test")[1])
        for short in ONTOLOGIES
    }
    rows = {short: [] for short in ONTOLOGIES}
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.rstrip("\n").split("\t")
            if not fields or not fields[0]:
                continue
            protein = fields[0]
            for item in fields[1:]:
                try:
                    term, raw_score = item.rsplit("|", 1)
                    score = float(raw_score)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid DeepGOPlus field {item!r}"
                    ) from exc
                for short in ONTOLOGIES:
                    if term in terms_by_ontology[short]:
                        rows[short].append((protein, term, score))
                        break
    normalize_scored_rows(
        workspace,
        "deepgoplus",
        write_rows(workspace, "deepgoplus", rows),
        "\t",
    )


def normalize_deepfri(workspace: Path, method: str, files: dict[str, Path]) -> None:
    for short, path in files.items():
        payload = json.loads(path.read_text())
        chains = payload.get("pdb_chains", [])
        source_terms = [normalize_term(term, {}) or str(term) for term in payload.get("goterms", [])]
        values = np.asarray(payload.get("Y_hat", []), np.float32)
        ids, terms, scores, valid_count = empty_query_matrix(workspace, short)
        id_index, term_index = {v: i for i, v in enumerate(ids)}, {v: i for i, v in enumerate(terms)}
        for source_row, raw_id in enumerate(chains):
            protein = Path(str(raw_id)).stem.replace(".pdb", "")
            if protein not in id_index or source_row >= len(values):
                continue
            for source_col, term in enumerate(source_terms):
                if term in term_index and source_col < values.shape[1]:
                    scores[id_index[protein], term_index[term]] = values[source_row, source_col]
        valid_ids, _, _ = load_label_npz(workspace, short, "valid")
        test_ids, _, _ = load_label_npz(workspace, short, "test")
        save_prediction(workspace, method + "_valid", short, valid_ids, terms, scores[:valid_count],
                        metadata={"source_file": str(path), "split": "valid"})
        save_prediction(workspace, method, short, test_ids, terms, scores[valid_count:],
                        metadata={"source_file": str(path), "split": "test"})


def normalize_interpro(workspace: Path, path: Path) -> None:
    terms_by_ontology = {short: set(load_label_npz(workspace, short, "test")[1]) for short in ONTOLOGIES}
    rows = {short: [] for short in ONTOLOGIES}
    with path.open() as handle:
        for raw in handle:
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 14:
                continue
            for term in re.findall(r"GO:\d{7}", fields[13]):
                for short in ONTOLOGIES:
                    if term in terms_by_ontology[short]:
                        rows[short].append((fields[0], term, 1.0))
    files = write_rows(workspace, "interproscan", rows)
    normalize_scored_rows(workspace, "interproscan", files, "\t", "binary")


def write_rows(workspace: Path, prefix: str, rows):
    directory = workspace / "raw" / "normalized_sources"
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for short in ONTOLOGIES:
        path = directory / f"{prefix}_{short}.tsv"
        with path.open("w") as handle:
            for row in rows[short]:
                handle.write("\t".join(map(str, row)) + "\n")
        files[short] = path
    return files


def normalize_eggnog(workspace: Path, path: Path) -> None:
    header = None
    rows = {short: [] for short in ONTOLOGIES}
    terms_by_ontology = {
        short: set(load_label_npz(workspace, short, "test")[1])
        for short in ONTOLOGIES
    }
    with path.open() as handle:
        for raw in handle:
            if raw.startswith("#query"):
                header = raw[1:].rstrip("\n").split("\t")
                continue
            if raw.startswith("#") or not raw.strip():
                continue
            if header is None:
                raise ValueError(f"{path}: eggNOG annotations header '#query' was not found")
            fields = raw.rstrip("\n").split("\t")
            record = dict(zip(header, fields))
            protein = record.get("query", "")
            for term in re.findall(r"GO:\d{7}", record.get("GOs", "") or ""):
                for short in ONTOLOGIES:
                    if term in terms_by_ontology[short]:
                        rows[short].append((protein, term, 1.0))
    normalize_scored_rows(
        workspace, "eggnog_mapper", write_rows(workspace, "eggnog_mapper", rows),
        "\t", "binary"
    )


def normalize_hayai(workspace: Path, path: Path) -> None:
    rows = {short: [] for short in ONTOLOGIES}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            for short, column in (("mf", "GO_MF"), ("bp", "GO_BP"), ("cc", "GO_CC")):
                rows[short].extend((row.get("Query", ""), term, 1.0)
                                   for term in re.findall(r"GO:\d{7}", row.get(column, "") or ""))
    normalize_scored_rows(workspace, "hayai", write_rows(workspace, "hayai", rows), "\t", "binary")


def normalize_gomap(workspace: Path, path: Path) -> None:
    term_to_ontology = {}
    for short in ONTOLOGIES:
        term_to_ontology.update({term: short for term in load_label_npz(workspace, short, "test")[1]})
    rows = {short: [] for short in ONTOLOGIES}
    with path.open() as handle:
        for raw in handle:
            if not raw.strip() or raw.startswith("!"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) >= 5 and fields[4] in term_to_ontology:
                rows[term_to_ontology[fields[4]]].append((fields[1], fields[4], 1.0))
    normalize_scored_rows(workspace, "gomap", write_rows(workspace, "gomap", rows), "\t", "binary")


def normalize_dpfunc(workspace: Path, method: str, files: dict[str, Path]) -> None:
    import pandas as pd

    rows = {short: [] for short in ONTOLOGIES}
    for short, path in files.items():
        frame = pd.read_pickle(path)
        required = {"protein_id", "predictions"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: DPFunc output is missing columns {sorted(required - set(frame.columns))}")
        for _, row in frame.iterrows():
            protein = str(row["protein_id"])
            predictions = row["predictions"]
            if not isinstance(predictions, dict):
                raise TypeError(f"{path}: predictions for {protein} are not a GO-score dictionary")
            rows[short].extend(
                (protein, term, float(score))
                for term, score in predictions.items()
                if float(score) > 0
            )
    normalize_scored_rows(workspace, method, write_rows(workspace, method, rows), "\t")
