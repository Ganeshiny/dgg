"""Utilities for reconstructing DeepGreenGO models from deployment checkpoints."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from .model import HybridGNN, HybridGNN_JK


def checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Return the model state dictionary from supported checkpoint layouts."""
    for key in ("model_state_dict", "model_state", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, Mapping):
            return state
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("checkpoint does not contain a model state dictionary")


def model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> torch.nn.Module:
    """Reconstruct a Hybrid model using dimensions encoded in its tensors."""
    state = checkpoint_state(checkpoint)
    required = (
        "input_linear.weight",
        "gcn_conv.lin.weight",
        "gat_conv.att",
        "gat_conv.bias",
        "output_layer.weight",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"checkpoint is missing architecture tensors: {', '.join(missing)}")

    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    model_name = str(config.get("model", config.get("model_type", "Hybrid")))
    model_class = HybridGNN_JK if "jk" in model_name.lower() else HybridGNN

    input_size = int(state["input_linear.weight"].shape[1])
    hidden_sizes = [
        int(state["input_linear.weight"].shape[0]),
        int(state["gcn_conv.lin.weight"].shape[0]),
        int(state["gat_conv.bias"].shape[0]),
    ]
    output_size = int(state["output_layer.weight"].shape[0])
    attention_heads = int(state["gat_conv.att"].shape[1])

    model = model_class(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        num_attention_heads=attention_heads,
        dropout=float(config.get("dropout", 0.3)),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_models(
    checkpoint_paths: Iterable[Path],
    *,
    device: torch.device | str = "cpu",
) -> list[torch.nn.Module]:
    """Load and reconstruct an ensemble from trusted local checkpoints."""
    models: list[torch.nn.Module] = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{path}: expected a checkpoint mapping")
        models.append(model_from_checkpoint(checkpoint, device=device))
    return models
