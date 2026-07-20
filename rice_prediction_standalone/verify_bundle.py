#!/usr/bin/env python3
"""Validate runtime imports, labels, and trained checkpoint architectures."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = ROOT if (ROOT / "src").is_dir() else ROOT.parent
sys.path.insert(0, str(RUNTIME_ROOT))

SEEDS = (1103, 2207, 3301, 4409, 5501)
ONTOLOGIES = ("molecular_function", "biological_process", "cellular_component")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    errors: list[str] = []
    for relative_path in ("scripts/predict_alphafold.py", "src/model.py", "src/checkpoint_loader.py"):
        path = ROOT / relative_path
        if not path.is_file() and ROOT == RUNTIME_ROOT:
            errors.append(f"missing runtime file: {path}")

    if not args.assets_only:
        try:
            from transformers import BertModel, BertTokenizer
            del BertModel, BertTokenizer
            print(f"transformers: OK ({package_version('transformers')})")
        except Exception as exc:
            errors.append(
                "runtime environment cannot import Transformers BertModel. "
                f"torch={torch.__version__}, torchvision={package_version('torchvision')}, "
                f"transformers={package_version('transformers')}. The usual cause is an old "
                "torchvision or compiled PyG extension left behind after PyTorch was upgraded. "
                "Create and activate the fresh .venv from README_STANDALONE.md instead of reusing "
                f"the training environment. Underlying error: {exc}"
            )

            print("\nFAIL")
            print(errors[-1])
            return 1
    try:
        with warnings.catch_warnings():
            if args.assets_only:
                warnings.filterwarnings("ignore", message="An issue occurred while importing")
            from src.checkpoint_loader import checkpoint_state, model_from_checkpoint
    except Exception as exc:
        errors.append(f"runtime environment cannot import the graph model: {exc}")
        checkpoint_state = None
        model_from_checkpoint = None

    for ontology in ONTOLOGIES:
        label_path = ROOT / "labels" / f"{ontology}_terms.json"
        try:
            terms = json.loads(label_path.read_text())
            if not isinstance(terms, list) or not terms or not all(isinstance(term, str) for term in terms):
                raise ValueError("must be a non-empty JSON list of strings")
            if len(terms) != len(set(terms)):
                raise ValueError("contains duplicate GO terms")
        except Exception as exc:
            errors.append(f"{label_path}: {exc}")
            continue

        print(f"{ontology}: {len(terms)} labels")
        for seed in SEEDS:
            path = ROOT / "weights" / ontology / f"seed_{seed}" / "best_checkpoint.pt"
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing checkpoint: {path}")
                continue
            if checkpoint_state is None or model_from_checkpoint is None:
                continue
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                state = checkpoint_state(checkpoint)
                output_dimension = int(state["output_layer.weight"].shape[0])
                if output_dimension != len(terms):
                    raise ValueError(
                        f"output dimension {output_dimension} does not match {len(terms)} labels"
                    )
                model = model_from_checkpoint(checkpoint)
                dummy_features = torch.zeros(3, model.input_linear.in_features)
                dummy_edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
                dummy_batch = torch.zeros(3, dtype=torch.long)
                with torch.no_grad():
                    prediction = model(dummy_features, dummy_edges, dummy_batch)
                if tuple(prediction.shape) != (1, len(terms)):
                    raise ValueError(
                        f"forward output shape {tuple(prediction.shape)} "
                        f"does not match (1, {len(terms)})"
                    )
                del model, checkpoint, state
                print(f"  seed {seed}: OK ({path.stat().st_size / 1e6:.1f} MB)")
            except Exception as exc:
                errors.append(f"invalid checkpoint {path}: {exc}")

    if errors:
        print("\nFAIL")
        print("\n".join(errors))
        return 1
    if args.assets_only:
        print("\nPASS: 15 checkpoints and 3 label files are compatible.")
    else:
        print("\nPASS: runtime imports, 15 checkpoints, and 3 label files are compatible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
