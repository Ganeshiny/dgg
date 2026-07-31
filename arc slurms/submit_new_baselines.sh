#!/usr/bin/env bash
# Submit the HEAL / GAT-GO benchmark chain in one command.
#
#   cd /home/ganeshiny.sridharan/dgg/deep-green-GO
#   bash 'arc slurms/submit_new_baselines.sh'
#
# DeepGraphGO is deliberately not part of this chain: the released repository
# ships no pretrained checkpoints (verified against the pinned commit
# efdb1cb9425f4f48e4613c0a89e603f5542bcb19 - no models/ directory, no GitHub
# Release). Its main.py trains three seeds per ontology from scratch; nothing
# in this pipeline runs that training, so there is no checkpoint to evaluate.
# All of its setup/runner/benchmark wiring is left in place for that as a
# separately scoped task - only this submission chain omits it.
#
# Layout and why:
#   A  setup   CPU  HEAL and GAT-GO in one pass. The environments are
#                   independent, so two separate setup jobs would only
#                   repeat the same Conda solves.
#   B  HEAL    GPU  ESM-1b inference over 1508 structures.
#   C  GAT-GO  GPU  separate from B so a released-feature coverage failure
#                   cannot discard a completed HEAL run.
#   D  scoring CPU  one 1000-bootstrap evaluation and one figure build over
#                   every method present, instead of one per method job.
#
# B and C use DGG_BENCHMARK_SKIP_EVALUATION=1 and are chained with afterany so
# one comparator failing still lets the other finish and be scored.

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

setup=$(DGG_SOTA_SETUP_METHODS=heal,gat_go \
    sbatch --parsable "${SETUP_JOB}")

heal=$(DGG_BENCHMARK_METHODS=hybrid,heal \
    DGG_BENCHMARK_SKIP_EVALUATION=1 \
    sbatch --parsable --dependency=afterok:"${setup}" "${GPU_JOB}")

gat=$(DGG_BENCHMARK_METHODS=hybrid,gat_go \
    DGG_BENCHMARK_SKIP_EVALUATION=1 \
    sbatch --parsable --dependency=afterany:"${heal}" "${GPU_JOB}")

# An empty DGG_BENCHMARK_REQUIRE_METHODS scores every method already normalized
# into predictions/, so a comparator that failed upstream is simply absent from
# the figures rather than aborting the scoring pass.
score=$(DGG_BENCHMARK_METHODS=hybrid \
    DGG_BENCHMARK_REQUIRE_METHODS= \
    DGG_BENCHMARK_SKIP_EVALUATION=0 \
    sbatch --parsable --dependency=afterany:"${gat}" "${CPU_JOB}")

printf '%s\n' \
    "A setup (CPU, heal+gat_go):     ${setup}" \
    "B HEAL (GPU):                   ${heal}" \
    "C GAT-GO (GPU):                 ${gat}" \
    "D evaluation and figures (CPU): ${score}"

squeue -j "${setup},${heal},${gat},${score}" || true

echo
echo "Watch the setup repair first:"
echo "  tail -f logs/setup_sota_${setup}.out"
echo "A successful setup ends with:"
echo "  [SETUP COMPLETE] All requested SOTA dependencies and model files passed verification"
