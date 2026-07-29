#!/usr/bin/env python3
"""Run released HEAL checkpoints on ARC PDB chains with resumable score caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


RESTYPE_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
TASKS = ("mf", "bp", "cc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heal-root", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--esm1b-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-variant",
        choices=("pdb-only", "combined"),
        default="pdb-only",
        help="pdb-only uses upstream *CL.pt; combined uses PDBch+AFch *CLaf.pt",
    )
    parser.add_argument(
        "--max-residues",
        type=int,
        default=1022,
        help="ESM-1b residue limit; longer proteins are reported and left at zero",
    )
    parser.add_argument("--expected-structures", type=int, default=1508)
    return parser.parse_args()


def extract_chain(path: Path):
    """Replicate HEAL's first-chain residue-span representation."""
    import numpy as np
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    model = next(structure.get_models())
    chain = next(model.get_chains())
    standard = [residue for residue in chain if residue.id[0] == " "]
    if not standard:
        raise ValueError(f"{path}: first chain has no standard residues")
    first = min(int(residue.id[1]) for residue in standard)
    last = max(int(residue.id[1]) for residue in standard)
    sequence = []
    coordinates = []
    for residue_number in range(first, last + 1):
        try:
            residue = chain[(" ", residue_number, " ")]
        except KeyError:
            residue = None
        sequence.append(
            RESTYPE_3TO1.get(residue.get_resname(), "X")
            if residue is not None
            else "X"
        )
        if residue is not None and "CA" in residue:
            coordinates.append(residue["CA"].get_coord())
        else:
            coordinates.append([np.nan, np.nan, np.nan])
    return "".join(sequence), np.asarray(coordinates, dtype=np.float32)


def edge_index_from_coordinates(coordinates):
    """Return HEAL's 10-Angstrom contact graph without Python O(n^2) loops."""
    import numpy as np

    gram = coordinates @ coordinates.T
    diagonal = np.diag(gram)
    with np.errstate(invalid="ignore"):
        squared = diagonal[:, None] + diagonal[None, :] - 2.0 * gram
        distance = np.sqrt(np.maximum(squared, 0.0))
    return np.asarray(np.where(distance <= 10.0), dtype=np.int64)


def load_state_dict(torch, path: Path, device: str):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a state-dict checkpoint")
    if payload and all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    return payload


def load_cached_scores(path: Path, expected_terms: int):
    import numpy as np

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            scores = np.asarray(payload["scores"], dtype=np.float32)
    except (OSError, KeyError, ValueError):
        return None
    return scores if scores.shape == (expected_terms,) else None


def save_cached_scores(path: Path, scores) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, scores=np.asarray(scores, dtype=np.float32))
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    heal_root = args.heal_root.resolve()
    required = [
        heal_root / "network.py",
        heal_root / "utils.py",
        heal_root / "data" / "nrPDB-GO_2019.06.18_annot.tsv",
        args.esm1b_model,
    ]
    suffix = "CL" if args.model_variant == "pdb-only" else "CLaf"
    required.extend(heal_root / "model" / f"model_{task}{suffix}.pt" for task in TASKS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing HEAL components: " + ", ".join(missing))

    structures = sorted(args.structure_dir.resolve().glob("*.pdb"))
    if len(structures) != args.expected_structures:
        raise SystemExit(
            f"Expected {args.expected_structures} ARC PDB files; found {len(structures)}"
        )

    sys.path.insert(0, str(heal_root))
    import esm
    import numpy as np
    import torch
    from torch_geometric.data import Batch
    from network import CL_protNET
    from utils import load_GO_annot, protein_graph

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"HEAL requested {args.device}, but CUDA is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, goterms, _, _ = load_GO_annot(
        str(heal_root / "data" / "nrPDB-GO_2019.06.18_annot.tsv")
    )
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(
        str(args.esm1b_model.resolve())
    )
    esm_model = esm_model.to(args.device)
    esm_model.eval()
    batch_converter = alphabet.get_batch_converter()

    report = {
        "method": "HEAL",
        "upstream_commit": "def3a3d478e6e809dacba9c24ff1a2ee87468b61",
        "model_variant": args.model_variant,
        "max_residues": args.max_residues,
        "expected_structures": args.expected_structures,
        "ontologies": {},
    }
    for task in TASKS:
        terms = list(goterms[task])
        model = CL_protNET(len(terms)).to(args.device)
        checkpoint = heal_root / "model" / f"model_{task}{suffix}.pt"
        model.load_state_dict(load_state_dict(torch, checkpoint, args.device))
        model.eval()

        cache_dir = args.output_dir / "cache" / args.model_variant / task
        skipped = {}
        processed = []
        for index, structure_path in enumerate(structures, start=1):
            protein_id = structure_path.stem
            cache_path = cache_dir / f"{protein_id}.npz"
            scores = load_cached_scores(cache_path, len(terms))
            if scores is None:
                sequence, coordinates = extract_chain(structure_path)
                if len(sequence) > args.max_residues:
                    skipped[protein_id] = {
                        "reason": "esm1b_length_limit",
                        "residues": len(sequence),
                    }
                    continue
                edge_index = edge_index_from_coordinates(coordinates)
                _, _, tokens = batch_converter([(protein_id, sequence)])
                tokens = tokens.to(args.device)
                with torch.no_grad():
                    representation = esm_model(
                        tokens,
                        repr_layers=[33],
                        return_contacts=False,
                    )["representations"][33][0, 1 : len(sequence) + 1]
                    graph = protein_graph(
                        sequence,
                        edge_index,
                        representation.cpu().numpy().astype(np.float16),
                    )
                    batch = Batch.from_data_list([graph]).to(args.device)
                    scores = model(batch)[0].detach().cpu().numpy().astype(np.float32)
                save_cached_scores(cache_path, scores)
                del tokens, representation, graph, batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            processed.append(protein_id)
            if index % 25 == 0:
                print(
                    f"[HEAL] {task}: visited {index}/{len(structures)} structures",
                    flush=True,
                )

        output = args.output_dir / f"heal_{task}.tsv"
        with output.open("w", encoding="utf-8") as handle:
            for protein_id in processed:
                scores = load_cached_scores(cache_dir / f"{protein_id}.npz", len(terms))
                if scores is None:
                    raise RuntimeError(f"Missing HEAL cache for {task}/{protein_id}")
                for term, score in zip(terms, scores):
                    handle.write(f"{protein_id}\t{term}\t{float(score):.9g}\n")
        report["ontologies"][task] = {
            "terms": len(terms),
            "predicted_proteins": len(processed),
            "skipped_proteins": skipped,
            "output": str(output),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
