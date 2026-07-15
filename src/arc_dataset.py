"""Lightweight, pickle-safe datasets for the ARC homology-controlled workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class ArcGraphDataset(Dataset):
    """Attach split-specific labels to shared, precomputed PyG graph files."""

    schema_version = 1

    def __init__(
        self,
        graph_dir: str | Path,
        protein_ids: Sequence[str],
        labels: np.ndarray,
        terms: Sequence[str],
        ontology: str,
        split: str,
        split_name: str = "nominal_30_identity_80_coverage",
    ) -> None:
        self.graph_dir = str(Path(graph_dir).expanduser().resolve())
        self.protein_ids = list(protein_ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.terms = list(terms)
        self.ontology = ontology
        self.split = split
        self.split_name = split_name
        if self.labels.shape != (len(self.protein_ids), len(self.terms)):
            raise ValueError(
                f"label shape {self.labels.shape} does not match "
                f"({len(self.protein_ids)}, {len(self.terms)})"
            )

    @property
    def num_classes(self) -> int:
        return len(self.terms)

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, index: int):
        protein_id = self.protein_ids[index]
        path = Path(self.graph_dir) / f"{protein_id}.pt"
        try:
            graph = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.0
            graph = torch.load(path, map_location="cpu")
        graph = graph.clone()
        graph.y = torch.from_numpy(self.labels[index].copy()).unsqueeze(0)
        graph.u = protein_id
        return graph


def make_dataloader(dataset: ArcGraphDataset, batch_size: int, shuffle: bool, workers: int = 0):
    """Construct the PyG loader used by both smoke tests and GPU tuning."""
    from torch_geometric.loader import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
