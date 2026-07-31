#!/usr/bin/env bash
# CPU-partition entry point for the same benchmark driver as
# run_full_benchmark.slurm. Use it for selections that never touch CUDA -
# DeepGraphGO, the alignment baselines, and the final scoring/figure pass - so
# those hours do not hold an idle L40.
#
# Submit from the project root, e.g.:
#   DGG_BENCHMARK_METHODS=hybrid,deepgraphgo \
#   DGG_BENCHMARK_SKIP_EVALUATION=1 \
#   sbatch 'arc slurms/run_benchmark_cpu.slurm'

#SBATCH --job-name=dgg_bench_cpu
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/benchmark_cpu_%j.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/benchmark_cpu_%j.err

set -euo pipefail

PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"

# No GPU is allocated here, so the core preflight must not demand one. An
# explicit caller-supplied value still wins.
export DGG_BENCHMARK_REQUIRE_GPU="${DGG_BENCHMARK_REQUIRE_GPU:-0}"

echo "========================================================"
echo "  Job ID   : ${SLURM_JOB_ID:-interactive}"
echo "  Node     : ${SLURM_NODELIST:-local}"
echo "  CPUs     : ${SLURM_CPUS_PER_TASK:-unset}"
echo "  Methods  : ${DGG_BENCHMARK_METHODS:-<script default>}"
echo "  Skip eval: ${DGG_BENCHMARK_SKIP_EVALUATION:-0}"
echo "========================================================"

# The driver's own #SBATCH lines are inert comments when it is run as a script,
# so this wrapper's CPU resource request is the one Slurm honours.
exec bash "${PROJECT_DIR}/arc slurms/run_full_benchmark.slurm"
