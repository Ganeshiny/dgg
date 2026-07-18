#!/usr/bin/env python3
"""Select Hybrid parameters using validation micro-Fmax only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ONTOLOGIES = ("molecular_function", "biological_process", "cellular_component")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-trials", type=int, default=40)
    parser.add_argument("--metric", choices=("validation_micro_fmax", "validation_macro_fmax", "validation_micro_aupr", "validation_macro_aupr", "validation_micro_auroc", "validation_macro_auroc", "validation_smin"), default="validation_micro_fmax")
    args = parser.parse_args()
    selected = {}
    for ontology in ONTOLOGIES:
        candidates = []
        for trial_dir in sorted(args.runs_dir.glob("trial_*")):
            config_path = trial_dir / ontology / "config.json"
            metrics_path = trial_dir / ontology / "validation_metrics.json"
            if not config_path.is_file() or not metrics_path.is_file():
                continue
            config = json.loads(config_path.read_text())
            metrics = json.loads(metrics_path.read_text())
            candidates.append((float(metrics[args.metric]), trial_dir.name, config, metrics))
        if len(candidates) != args.expected_trials:
            raise SystemExit(
                f"{ontology}: expected {args.expected_trials} completed validation results, found {len(candidates)}"
            )
        if args.metric == "validation_smin":
            score, trial_name, config, metrics = min(candidates, key=lambda item: (item[0], item[1]))
        else:
            score, trial_name, config, metrics = max(candidates, key=lambda item: (item[0], item[1]))
        search_keys = (
            "learning_rate", "weight_decay", "dropout", "hidden_dim", "batch_size",
            "gradient_clip", "patience", "loss", "focal_gamma",
        )
        selected[ontology] = {
            "trial": trial_name,
            "selection_metric": args.metric,
            args.metric: score,
            "config": {key: config[key] for key in search_keys},
            "validation_metrics": metrics,
        }
    payload = {
        "selection_data": "validation only",
        "selection_metric": args.metric,
        "split": "nominal 30% identity / 80% coverage split",
        "ontologies": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
