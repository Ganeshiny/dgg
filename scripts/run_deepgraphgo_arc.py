#!/usr/bin/env python3
"""Run the released three-model DeepGraphGO ensemble on ARC sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np


ONTOLOGIES = ("mf", "bp", "cc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepgraphgo-root", type=Path, required=True)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-proteins", type=int, default=1508)
    parser.add_argument("--runtime-smoke-test", action="store_true")
    return parser.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    records.append((current, "".join(chunks)))
                current, chunks = line[1:].split()[0], []
            elif current is None:
                raise ValueError(f"{path}: sequence before first header")
            else:
                chunks.append(line)
    if current is not None:
        records.append((current, "".join(chunks)))
    ids = [protein_id for protein_id, _ in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Input FASTA identifiers are not unique")
    return records


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def yaml_quote(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def main() -> None:
    args = parse_args()
    root = args.deepgraphgo_root.resolve()
    if args.runtime_smoke_test:
        import dgl
        import dgl.data.utils  # noqa: F401  (dgl.data is not guaranteed importable as an attribute of a bare `import dgl`; DeepGraphGO's own main.py imports it explicitly for the same reason)
        import scipy.sparse as ssp
        import torch

        sys.path.insert(0, str(root))
        from deepgraphgo.networks import GcnNet

        graph = dgl.data.utils.load_graphs(str(root / "data/ppi_dgl_top_100"))[0][0]
        features = ssp.load_npz(root / "data/ppi_interpro.npz")
        if graph.number_of_nodes() != features.shape[0]:
            raise ValueError("DeepGraphGO graph and InterPro feature row counts differ")
        if not hasattr(dgl, "contrib") or not hasattr(dgl.contrib, "sampling"):
            raise RuntimeError("DeepGraphGO requires the released DGL NodeFlow API")
        classes = joblib.load(root / "data/mf_go.mlb").classes_
        model = GcnNet(
            labels_num=len(classes), input_size=features.shape[1],
            hidden_size=512, num_gcn=2,
        )
        state = torch.load(
            root / "models/DeepGraphGO-Model-0-mf", map_location="cpu"
        )
        model.load_state_dict(state, strict=True)
        print(
            f"DeepGraphGO runtime smoke test passed: nodes={graph.number_of_nodes()}, "
            f"features={features.shape[1]}, labels={len(classes)}"
        )
        return
    if args.fasta is None or args.workspace is None or args.output_dir is None:
        raise ValueError("--fasta, --workspace, and --output-dir are required for ARC inference")
    output = args.output_dir.resolve()
    records = read_fasta(args.fasta.resolve())
    if len(records) != args.expected_proteins:
        raise ValueError(
            f"DeepGraphGO requires {args.expected_proteins} validation+test proteins; found {len(records)}"
        )
    output.mkdir(parents=True, exist_ok=True)
    # Synthetic query names prevent accidental direct lookup in the released PPI
    # network. Every ARC sequence must use DeepGraphGO's documented BLAST mapping.
    synthetic_ids = [f"DGGQ{index:06d}" for index in range(len(records))]
    query_text = "".join(
        f">{synthetic}\n{sequence}\n"
        for synthetic, (_, sequence) in zip(synthetic_ids, records)
    )
    query_sha256 = hashlib.sha256(query_text.encode()).hexdigest()
    cache = output / "cache" / query_sha256[:16]
    cache.mkdir(parents=True, exist_ok=True)
    query_fasta = cache / "arc_queries_synthetic_ids.fasta"
    query_fasta.write_text(query_text)

    for relative in (
        "main.py", "configure/dgg.yaml", "data/ppi_pid_list.txt",
        "data/ppi_interpro.npz", "data/ppi_dgl_top_100", "data/ppi_blastdb.pin",
    ):
        require(root / relative)
    summaries = {}
    model_config = cache / "model.yaml"
    model_config.write_text(
        "name: DeepGraphGO\nmodel:\n  hidden_size: 512\n  num_gcn: 2\n"
        "test:\n  batch_size: 40\n"
    )
    target_terms = {}
    shared_blast_xml = next(cache.glob("*/*-arc_valid_test-ppi-blast-out.xml"), None)
    for ontology in ONTOLOGIES:
        with np.load(args.workspace / "labels" / f"{ontology}_test.npz", allow_pickle=False) as data:
            target_terms[ontology] = [str(value) for value in data["go_terms"]]
        mlb_path = root / "data" / f"{ontology}_go.mlb"
        require(mlb_path)
        for model_id in range(3):
            require(root / "models" / f"DeepGraphGO-Model-{model_id}-{ontology}")
        ontology_output = cache / ontology
        ontology_output.mkdir(exist_ok=True)
        blast_xml = ontology_output / f"{ontology}-arc_valid_test-ppi-blast-out.xml"
        if shared_blast_xml is not None and not blast_xml.exists():
            shutil.copy2(shared_blast_xml, blast_xml)
        data_config = cache / f"{ontology}.yaml"
        data_config.write_text(
            f"name: {ontology}\nmodel_path: {yaml_quote(root / 'models')}\n"
            f"mlb: {yaml_quote(mlb_path)}\nresults: {yaml_quote(ontology_output)}\n"
            "network:\n"
            f"  pid_list: {yaml_quote(root / 'data/ppi_pid_list.txt')}\n"
            f"  weight_mat: {yaml_quote(root / 'data/ppi_mat.npz')}\n"
            f"  blastdb: {yaml_quote(root / 'data/ppi_blastdb')}\n"
            f"  dgl: {yaml_quote(root / 'data/ppi_dgl_top_100')}\n"
            f"  feature: {yaml_quote(root / 'data/ppi_interpro.npz')}\n"
            "test:\n  name: arc_valid_test\n"
            f"  fasta_file: {yaml_quote(query_fasta)}\n"
        )
        arrays = []
        for model_id in range(3):
            result = ontology_output / (
                f"DeepGraphGO-Model-{model_id}-{ontology}-arc_valid_test.npy"
            )
            if result.exists():
                print(f"Reusing complete DeepGraphGO output {result}", flush=True)
            else:
                subprocess.run([
                    "python", str(root / "main.py"), "-m", str(model_config),
                    "-d", str(data_config), "--mode", "eval", "--model-id", str(model_id),
                ], cwd=root, check=True)
            require(result)
            if shared_blast_xml is None and blast_xml.exists():
                shared_blast_xml = blast_xml
            array = np.load(result)
            if array.shape[0] != len(records):
                raise ValueError(f"{result}: {array.shape[0]} rows, expected {len(records)}")
            if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                raise ValueError(f"{result}: predictions are non-finite or outside [0, 1]")
            arrays.append(array)
        scores = np.mean(np.stack(arrays), axis=0)
        classes = [str(term) for term in joblib.load(mlb_path).classes_]
        if scores.shape[1] != len(classes):
            raise ValueError(f"{ontology}: score columns do not match released GO vocabulary")
        source_index = {term: index for index, term in enumerate(classes)}
        common = [term for term in target_terms[ontology] if term in source_index]
        score_file = output / f"deepgraphgo_{ontology}.tsv"
        with score_file.open("w") as handle:
            for row, (protein_id, _) in enumerate(records):
                for term in common:
                    handle.write(
                        f"{protein_id}\t{term}\t{float(scores[row, source_index[term]]):.9g}\n"
                    )
        summaries[ontology] = {
            "released_terms": len(classes),
            "target_terms": len(target_terms[ontology]),
            "common_terms": len(common),
            "queries_without_blast_mapping": int(np.count_nonzero(np.all(scores == 0, axis=1))),
        }
    (output / "manifest.json").write_text(json.dumps({
        "method": "DeepGraphGO",
        "upstream_revision": (root / ".dgg_upstream_revision").read_text().strip(),
        "proteins": len(records),
        "query_sha256": query_sha256,
        "ensemble_models_per_ontology": 3,
        "query_identifiers_synthetic": True,
        "mapping": "one-iteration PSI-BLAST to released PPI network (official inference path)",
        "device": "cpu",
        "ontologies": summaries,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
