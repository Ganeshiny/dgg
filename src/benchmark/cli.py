#!/usr/bin/env python3
"""Command-line entry point used by the single ARC Slurm benchmark job."""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import export_hybrid, prepare
from .evaluate import evaluate, plot
from .external import clone_dpfunc_workspace, make_dpfunc_manifest, run_hayai
from .methods import (
    foldseek_baseline,
    normalize_deepfri,
    normalize_deepgoplus,
    normalize_dpfunc,
    normalize_eggnog,
    normalize_gomap,
    normalize_hayai,
    normalize_interpro,
    normalize_scored_rows,
    sequence_baselines,
)


def add_workspace(parser):
    parser.add_argument("--workspace", type=Path, required=True)


def files_by_ontology(args):
    return {"mf": args.mf, "bp": args.bp, "cc": args.cc}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("prepare")
    command.add_argument("--data-root", type=Path, required=True)
    add_workspace(command)
    command.add_argument("--expected-test-size", type=int, default=754)
    command.add_argument("--extract-structures", action="store_true")
    command.set_defaults(func=prepare)

    command = sub.add_parser("export-hybrid")
    command.add_argument("--data-root", type=Path, required=True)
    command.add_argument("--checkpoint-root", type=Path, required=True)
    command.add_argument("--graph-root", type=Path, required=True)
    add_workspace(command)
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--require-cuda", action="store_true")
    command.set_defaults(func=export_hybrid)

    command = sub.add_parser("sequence-baselines")
    add_workspace(command)
    command.add_argument("--methods", default="blast,diamond")
    command.add_argument("--makeblastdb", default="makeblastdb")
    command.add_argument("--blastp", default="blastp")
    command.add_argument("--diamond", default="diamond")
    command.add_argument("--threads", type=int, default=8)
    command.add_argument("--max-hits", type=int, default=50)
    command.add_argument("--top-k", type=int, default=10)
    command.add_argument("--min-qcov", type=float, default=0.5)
    command.add_argument("--min-tcov", type=float, default=0.5)
    command.set_defaults(func=sequence_baselines)

    command = sub.add_parser("foldseek")
    add_workspace(command)
    command.add_argument("--foldseek", default="foldseek")
    command.add_argument("--threads", type=int, default=8)
    command.add_argument("--max-hits", type=int, default=50)
    command.add_argument("--top-k", type=int, default=10)
    command.add_argument("--min-qcov", type=float, default=0.5)
    command.add_argument("--min-tcov", type=float, default=0.5)
    command.add_argument("--force-search", action="store_true")
    command.set_defaults(func=foldseek_baseline)

    command = sub.add_parser("normalize-scored")
    add_workspace(command)
    command.add_argument("--method", required=True)
    command.add_argument("--mf", type=Path, required=True)
    command.add_argument("--bp", type=Path, required=True)
    command.add_argument("--cc", type=Path, required=True)
    command.add_argument("--delimiter", choices=("whitespace", "tab"), default="whitespace")
    command.add_argument("--score-type", choices=("continuous", "binary"), default="continuous")
    command.add_argument("--score-column", type=int, default=2)
    command.set_defaults(func=lambda args: normalize_scored_rows(
        args.workspace.resolve(), args.method, files_by_ontology(args),
        None if args.delimiter == "whitespace" else "\t",
        args.score_type, args.score_column))

    command = sub.add_parser("normalize-deepfri")
    add_workspace(command)
    command.add_argument("--method", required=True)
    command.add_argument("--mf", type=Path, required=True)
    command.add_argument("--bp", type=Path, required=True)
    command.add_argument("--cc", type=Path, required=True)
    command.set_defaults(func=lambda args: normalize_deepfri(
        args.workspace.resolve(), args.method, files_by_ontology(args)))

    command = sub.add_parser("normalize-deepgoplus")
    add_workspace(command)
    command.add_argument("--input", type=Path, required=True)
    command.set_defaults(func=lambda args: normalize_deepgoplus(
        args.workspace.resolve(), args.input.resolve()))

    command = sub.add_parser("normalize-dpfunc")
    add_workspace(command)
    command.add_argument("--method", default="dpfunc")
    command.add_argument("--mf", type=Path, required=True)
    command.add_argument("--bp", type=Path, required=True)
    command.add_argument("--cc", type=Path, required=True)
    command.set_defaults(func=lambda args: normalize_dpfunc(
        args.workspace.resolve(), args.method, files_by_ontology(args)))

    command = sub.add_parser("normalize-interpro")
    add_workspace(command)
    command.add_argument("--input", type=Path, required=True)
    command.set_defaults(func=lambda args: normalize_interpro(args.workspace.resolve(), args.input.resolve()))

    command = sub.add_parser("normalize-eggnog")
    add_workspace(command)
    command.add_argument("--input", type=Path, required=True)
    command.set_defaults(func=lambda args: normalize_eggnog(
        args.workspace.resolve(), args.input.resolve()))

    command = sub.add_parser("normalize-hayai")
    add_workspace(command)
    command.add_argument("--input", type=Path, required=True)
    command.set_defaults(func=lambda args: normalize_hayai(args.workspace.resolve(), args.input.resolve()))

    command = sub.add_parser("normalize-gomap")
    add_workspace(command)
    command.add_argument("--input", type=Path, required=True)
    command.set_defaults(func=lambda args: normalize_gomap(args.workspace.resolve(), args.input.resolve()))

    command = sub.add_parser("make-dpfunc-manifest")
    add_workspace(command)
    command.add_argument("--interpro-tsv", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=make_dpfunc_manifest)

    command = sub.add_parser("clone-dpfunc-workspace")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--target", type=Path, required=True)
    command.add_argument("--source-ontology", choices=("mf", "bp", "cc"), required=True)
    command.add_argument("--target-ontology", choices=("mf", "bp", "cc"), required=True)
    command.add_argument("--mlb-source", type=Path, required=True)
    command.set_defaults(func=clone_dpfunc_workspace)

    command = sub.add_parser("run-hayai")
    add_workspace(command)
    command.add_argument("--hayai-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--diamond", default="diamond")
    command.add_argument("--odb-mapper", default="ODB-mapper")
    command.add_argument("--threads", type=int, default=8)
    command.add_argument("--sensitivity", default="very-sensitive")
    command.add_argument("--orthologer", action="store_true")
    command.set_defaults(func=run_hayai)

    command = sub.add_parser("evaluate")
    add_workspace(command)
    command.add_argument("--data-root", type=Path, default=None,
                         help="Override the manifest's recorded data root (needed when "
                              "re-evaluating a workspace produced on another host).")
    command.add_argument("--bootstraps", type=int, default=1000)
    command.add_argument("--bootstrap-seed", type=int, default=20260720)
    command.add_argument(
        "--aupr-workers",
        type=int,
        default=8,
        help="Threads used for paired AUPR bootstrap calculations.",
    )
    command.add_argument("--require-methods", default="")
    command.set_defaults(func=evaluate)

    command = sub.add_parser("plot")
    add_workspace(command)
    command.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

