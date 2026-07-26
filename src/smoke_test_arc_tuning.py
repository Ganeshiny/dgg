#!/usr/bin/env python3
"""Small CPU smoke test for ARC PKL and PyG dataloader compatibility."""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.arc_dataset import ArcGraphDataset, make_dataloader


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        ids = ["TOY_A", "TOY_B", "TOY_C"]
        for index, protein_id in enumerate(ids):
            torch.save(
                Data(
                    x=torch.randn(4 + index, 7),
                    edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                    u=protein_id,
                ),
                root / f"{protein_id}.pt",
            )
        dataset = ArcGraphDataset(
            root, ids, np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
            ["GO:1", "GO:2"], "biological_process", "train",
        )
        path = root / "toy.pkl"
        with path.open("wb") as handle:
            pickle.dump(dataset, handle)
        with path.open("rb") as handle:
            restored = pickle.load(handle)
        batch = next(iter(make_dataloader(restored, batch_size=2, shuffle=False)))
        assert batch.x.shape[1] == 7
        assert tuple(batch.y.shape) == (2, 2)
        assert restored.num_classes == 2
    print("CPU PKL/dataloader smoke test passed")


if __name__ == "__main__":
    main()
