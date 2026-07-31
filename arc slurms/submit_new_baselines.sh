#!/usr/bin/env bash
# Submit the HEAL benchmark chain in one command.
#
#   cd /home/ganeshiny.sridharan/dgg/deep-green-GO
#   bash 'arc slurms/submit_new_baselines.sh'
#
# GAT-GO and DeepGraphGO are deliberately not part of this chain, both for
# release-coverage reasons rather than environment bugs. All of their setup,
# runner, and benchmark wiring (including a verified state-dict remap for
# GAT-GO's checkpoint) is left in place; only this submission omits them.
#
#   - DeepGraphGO: the released repository ships no pretrained checkpoints
#     (verified against the pinned commit
#     efdb1cb9425f4f48e4613c0a89e603f5542bcb19 - no models/ directory, no
#     GitHub Release). Its main.py trains three seeds per ontology from
#     scratch; nothing in this pipeline runs that training.
#   - GAT-GO: the official precomputed per-chain features cover only 68 of
#     the 1508 locked ARC query proteins (4.5%, confirmed via
#     scripts/run_gat_go_arc.py --report-coverage-only during setup). The
#     runner's strict preflight correctly refuses to score that small a
#     subset as if it were full coverage, so the GPU job would fail there
#     even with a working environment.
#
# Layout and why:
#   A  setup   CPU  HEAL only for now.
#   B  HEAL    GPU  ESM-1b inference over 1508 structures.
#   C  scoring CPU  the 1000-bootstrap evaluation and figure build.
#
# B uses DGG_BENCHMARK_SKIP_EVALUATION=1 so the scoring pass in C is computed
# once rather than repeated per method job (relevant again once more methods
# are added back to this chain).

set -euo pipefail

PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
cd "${PROJECT_DIR}"
mkdir -p logs

SETUP_JOB='arc slurms/arc_01_setup_sota.slurm'
GPU_JOB='arc slurms/run_full_benchmark.slurm'
CPU_JOB='arc slurms/run_benchmark_cpu.slurm'

for script in "${SETUP_JOB}" "${GPU_JOB}" "${CPU_JOB}"; do
    [[ -f "${script}" ]] || { echo "[ERROR] Missing ${script}" >&2; exit 1; }
done

# Resource right-sizing. The #SBATCH directives inside the driver scripts are
# sized for a *full* benchmark (every method, from scratch); this chain runs
# only HEAL plus scoring, so the requests below override them on the sbatch
# command line (CLI flags take precedence over in-script directives). Asking
# only for what a job uses is both an ARC expectation and the practical way to
# get scheduled sooner, since a smaller, shorter job backfills far more
# readily than a 24-hour whole-node reservation.
#
# Confirm these against reality after the run and tighten further if needed:
#   seff <jobid>      # CPU/memory efficiency and actual elapsed time
# Every stage is resumable (stage markers plus HEAL's per-protein score
# cache), so an occasional timeout costs a resubmit, not lost work.

# Setup: repo checkouts, Conda solves, and downloads that are already cached
# from the previous successful run - so this is mostly verification now.
setup=$(DGG_SOTA_SETUP_METHODS=heal \
    sbatch --parsable \
    --time=04:00:00 --cpus-per-task=8 --mem=32G \
    "${SETUP_JOB}")

# HEAL: GPU-bound ESM-1b inference over 1508 structures, now computing the
# encoder once per protein rather than once per ontology. The surrounding
# per-protein work (Biopython parsing, graph construction) is single-threaded
# Python, so 16 cores and 96 GB on a shared L40 node were both unused
# reservations that keep other GPU work off the node.
heal=$(DGG_BENCHMARK_METHODS=hybrid,heal \
    DGG_BENCHMARK_SKIP_EVALUATION=1 \
    sbatch --parsable --dependency=afterok:"${setup}" \
    --time=06:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1 \
    "${GPU_JOB}")

# Scoring: the 1000-replicate paired bootstrap genuinely parallelises across
# --aupr-workers, so the 32 cores here are actually used; only the walltime
# and memory were over-reserved.
# An empty DGG_BENCHMARK_REQUIRE_METHODS scores every method already normalized
# into predictions/, so a comparator that failed upstream is simply absent from
# the figures rather than aborting the scoring pass.
score=$(DGG_BENCHMARK_METHODS=hybrid \
    DGG_BENCHMARK_REQUIRE_METHODS= \
    DGG_BENCHMARK_SKIP_EVALUATION=0 \
    sbatch --parsable --dependency=afterany:"${heal}" \
    --time=04:00:00 --cpus-per-task=32 --mem=64G \
    "${CPU_JOB}")

printf '%s\n' \
    "A setup (CPU, heal):            ${setup}" \
    "B HEAL (GPU):                   ${heal}" \
    "C evaluation and figures (CPU): ${score}"

squeue -j "${setup},${heal},${score}" || true

echo
echo "Watch the setup repair first:"
echo "  tail -f logs/setup_sota_${setup}.out"
echo "A successful setup ends with:"
echo "  [SETUP COMPLETE] All requested SOTA dependencies and model files passed verification"
