#!/usr/bin/env python3
"""Run the official GAT-GO checkpoint on a fully covered ARC query set.

GAT-GO's release accepts only precomputed per-chain feature dictionaries.  This
adapter deliberately refuses partial coverage and never reads the bundled
``label`` field, which is not an inference feature.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ONTOLOGIES = ("mf", "bp", "cc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gat-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--go-map", type=Path, required=True)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-proteins", type=int, default=1508)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--runtime-smoke-test", action="store_true")
    return parser.parse_args()


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    if current in records:
                        raise ValueError(f"Duplicate FASTA identifier: {current}")
                    records[current] = "".join(chunks)
                current, chunks = line[1:].split()[0], []
            elif current is None:
                raise ValueError(f"{path}: sequence before first header")
            else:
                chunks.append(line)
    if current is not None:
        if current in records:
            raise ValueError(f"Duplicate FASTA identifier: {current}")
        records[current] = "".join(chunks)
    return records


def resolve_feature(root: Path, protein_id: str) -> Path:
    candidates = (root / f"{protein_id}.pt", root / f"{protein_id.replace('_', '-')}.pt")
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"{protein_id}: expected exactly one GAT-GO feature file; checked "
            + ", ".join(map(str, candidates))
        )
    return found[0]


def load_object(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_feature(path: Path, protein_id: str, sequence: str) -> dict:
    obj = load_object(path)
    required = {"x", "pssm", "seq", "edge_index", "seq_embed"}
    missing = required - set(obj)
    if missing:
        raise ValueError(f"{path}: missing inference fields {sorted(missing)}")
    length = len(sequence)
    x, pssm, seq = obj["x"], obj["pssm"], obj["seq"]
    edge_index, seq_embed = obj["edge_index"], obj["seq_embed"]
    if tuple(x.shape) != (length, 1280):
        raise ValueError(f"{protein_id}: x shape {tuple(x.shape)}, expected {(length, 1280)}")
    if tuple(pssm.shape) != (20, length):
        raise ValueError(f"{protein_id}: pssm shape {tuple(pssm.shape)}, expected {(20, length)}")
    if tuple(seq.shape) != (25, length):
        raise ValueError(f"{protein_id}: seq shape {tuple(seq.shape)}, expected {(25, length)}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{protein_id}: edge_index must have shape (2, E)")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= length):
        raise ValueError(f"{protein_id}: contact edge index is outside sequence length {length}")
    if seq_embed.numel() != 1280:
        raise ValueError(f"{protein_id}: seq_embed has {seq_embed.numel()} values, expected 1280")
    # Never access obj['label']; labels serialized by upstream are forbidden at inference.
    return {key: obj[key] for key in required}


def target_terms(workspace: Path) -> dict[str, list[str]]:
    result = {}
    for ontology in ONTOLOGIES:
        path = workspace / "labels" / f"{ontology}_test.npz"
        with np.load(path, allow_pickle=False) as data:
            result[ontology] = [str(value) for value in data["go_terms"]]
    return result


def main() -> None:
    args = parse_args()
    if args.runtime_smoke_test:
        sys.path.insert(0, str(args.gat_root.resolve()))
        from src.GnnPF import GnnPF

        checkpoint = load_object(args.model.resolve())
        state = checkpoint.get("state_dict", checkpoint)
        model = GnnPF().cpu()
        model.load_state_dict(state, strict=True)
        model.eval()
        go2index = load_object(args.go_map.resolve())
        if set(go2index.values()) != set(range(2752)):
            raise ValueError("GAT-GO go2index.pt must map exactly onto output indices 0..2751")
        feature_path = next(args.feature_root.resolve().glob("*.pt"), None)
        if feature_path is None:
            raise FileNotFoundError(f"No .pt feature files under {args.feature_root}")
        raw = load_object(feature_path)
        length = int(raw["x"].shape[0])
        obj = validate_feature(feature_path, feature_path.stem, "X" * length)
        with torch.inference_mode():
            scores = torch.sigmoid(model(
                esm_rep=obj["x"].T.unsqueeze(0).float(),
                seq=obj["seq"].unsqueeze(0).float(),
                pssm=obj["pssm"].unsqueeze(0).float(),
                seq_embed=obj["seq_embed"].reshape(1, 1280).float(),
                A=obj["edge_index"].long(),
                batch=torch.zeros(length, dtype=torch.long),
            )).squeeze(0).numpy()
        if scores.shape != (2752,) or not np.isfinite(scores).all():
            raise ValueError("GAT-GO runtime smoke test produced invalid output")
        print(f"GAT-GO runtime smoke test passed with {feature_path.name}")
        return
    if args.fasta is None or args.workspace is None or args.output_dir is None:
        raise ValueError("--fasta, --workspace, and --output-dir are required for ARC inference")
    records = read_fasta(args.fasta.resolve())
    if len(records) != args.expected_proteins:
        raise ValueError(
            f"GAT-GO requires {args.expected_proteins} validation+test proteins; found {len(records)}"
        )
    resolved: dict[str, Path] = {}
    errors: list[str] = []
    for protein_id, sequence in records.items():
        try:
            path = resolve_feature(args.feature_root.resolve(), protein_id)
            validate_feature(path, protein_id, sequence)
            resolved[protein_id] = path
        except (FileNotFoundError, ValueError, KeyError) as exc:
            errors.append(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "method": "GAT-GO",
        "proteins_expected": args.expected_proteins,
        "proteins_with_valid_features": len(resolved),
        "proteins_missing_or_invalid": len(errors),
        "labels_used_for_inference": False,
        "feature_release_limitation": (
            "Upstream provides precomputed features but no feature-generation pipeline; "
            "partial ARC coverage is rejected."
        ),
        "errors": errors[:100],
    }
    (args.output_dir / "preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
    if errors:
        raise RuntimeError(
            f"GAT-GO feature audit failed for {len(errors)}/{len(records)} proteins; "
            f"see {args.output_dir / 'preflight.json'}"
        )
    if args.preflight_only:
        print(json.dumps(audit, indent=2))
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GAT-GO requested CUDA but torch.cuda.is_available() is false")
    sys.path.insert(0, str(args.gat_root.resolve()))
    from src.GnnPF import GnnPF

    checkpoint = load_object(args.model.resolve())
    state = checkpoint.get("state_dict", checkpoint)
    model = GnnPF().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    go2index = load_object(args.go_map.resolve())
    if set(go2index.values()) != set(range(2752)):
        raise ValueError("GAT-GO go2index.pt must map exactly onto output indices 0..2751")
    index_by_term = {str(term): int(index) for term, index in go2index.items()}
    terms_by_ontology = target_terms(args.workspace.resolve())
    handles = {
        ontology: (args.output_dir / f"gat_go_{ontology}.tsv").open("w")
        for ontology in ONTOLOGIES
    }
    try:
        with torch.inference_mode():
            for number, (protein_id, sequence) in enumerate(records.items(), start=1):
                obj = validate_feature(resolved[protein_id], protein_id, sequence)
                length = len(sequence)
                esm_rep = obj["x"].T.unsqueeze(0).float().to(device)
                seq = obj["seq"].unsqueeze(0).float().to(device)
                pssm = obj["pssm"].unsqueeze(0).float().to(device)
                edge_index = obj["edge_index"].long().to(device)
                seq_embed = obj["seq_embed"].reshape(1, 1280).float().to(device)
                batch = torch.zeros(length, dtype=torch.long, device=device)
                scores = torch.sigmoid(model(
                    esm_rep=esm_rep, seq=seq, pssm=pssm, seq_embed=seq_embed,
                    A=edge_index, batch=batch,
                )).squeeze(0).detach().cpu().numpy()
                if scores.shape != (2752,) or not np.isfinite(scores).all():
                    raise ValueError(f"{protein_id}: invalid GAT-GO output shape or values")
                if np.any((scores < 0) | (scores > 1)):
                    raise ValueError(f"{protein_id}: GAT-GO scores fall outside [0, 1]")
                for ontology, terms in terms_by_ontology.items():
                    handle = handles[ontology]
                    for term in terms:
                        index = index_by_term.get(term)
                        if index is not None:
                            handle.write(f"{protein_id}\t{term}\t{float(scores[index]):.9g}\n")
                if number % 25 == 0 or number == len(records):
                    print(f"GAT-GO predicted {number}/{len(records)} proteins", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    (args.output_dir / "manifest.json").write_text(json.dumps({
        **audit,
        "upstream_revision": (
            args.gat_root.resolve() / ".dgg_upstream_revision"
        ).read_text().strip(),
        "checkpoint": str(args.model.resolve()),
        "go_map": str(args.go_map.resolve()),
        "target_vocabulary_only": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
