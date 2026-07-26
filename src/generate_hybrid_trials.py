#!/usr/bin/env python3
"""Generate a reproducible random search for Hybrid (not a Cartesian grid)."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 30 <= args.trials <= 40:
        raise SystemExit("Random search must contain 30-40 trials")
    rng = random.Random(args.seed)
    rows = []
    seen = set()
    while len(rows) < args.trials:
        loss = rng.choice(["BCE", "Focal"])
        row = {
            "trial_id": len(rows),
            "learning_rate": math.exp(rng.uniform(math.log(1e-5), math.log(3e-3))),
            "weight_decay": math.exp(rng.uniform(math.log(1e-7), math.log(1e-2))),
            "dropout": rng.choice([0.0, 0.1, 0.2, 0.4]),
            "hidden_dim": rng.choice([128, 256, 512]),
            "batch_size": rng.choice([16, 32, 64]),
            "gradient_clip": rng.choice([0.5, 1.0, 5.0]),
            "patience": rng.choice([8, 9, 10]),
            "loss": loss,
            "focal_gamma": rng.choice([1, 2, 3]) if loss == "Focal" else None,
            "search_seed": args.seed,
        }
        signature = tuple((key, row[key]) for key in row if key not in {"trial_id"})
        if signature in seen:
            continue
        seen.add(signature)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"Wrote {len(rows)} random-search trials to {args.output}")


if __name__ == "__main__":
    main()
