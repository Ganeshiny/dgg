#!/usr/bin/env bash
# Submit the HEAL / GAT-GO / DeepGraphGO benchmark chain in one command.
#
#   cd /home/ganeshiny.sridharan/dgg/deep-green-GO
#   bash 'arc slurms/submit_new_baselines.sh'
#
# Layout and why:
#   A  setup        CPU  all three methods in a single pass. The environments are
#                        independent, so three separate setup jobs would only
#                        repeat the same Conda solves.
#   B  HEAL         GPU  ESM-1b inference over 1508 structures.
#   C  GAT-GO       GPU  separate from B so a released-feature coverage failure
#                        cannot discard a completed HEAL run.
#   D  DeepGraphGO  CPU  the pinned PyTorch 1.6 / DGL 0.4 stack is CPU-only;
#                        running it on gpu-l40 would idle an L40 for hours.
#   E  scoring      CPU  one 1000-bootstrap evaluation and one figure build over
#                        every method present, instead of one per method job.
#
# B..D use DGG_BENCHMARK_SKIP_EVALUATION=1 and are chained with afterany so one
# comparator failing still lets the rest finish and be scored.

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

setup=$(DGG_SOTA_SETUP_METHODS=heal,gat_go,deepgraphgo \
    sbatch --parsable "${SETUP_JOB}")

heal=$(DGG_BENCHMARK_METHODS=hybrid,heal \
    DGG_BENCHMARK_SKIP_EVALUATION=1 \
    sbatch --parsable --dependency=afterok:"${setup}" "${GPU_JOB}")

gat=$(DGG_BENCHMARK_METHODS=hybrid,gat_go \
    DGG_BENCHMARK_SKIP_EVALUATION=1 \
    sbatch --parsable --dependency=afterany:"${heal}" "${GPU_JOB}")

graph=$(DGG_BENCHMARK_METHODS=hybrid,deepgraphgo \
    DGG_BENCHMARK_SKIP_EVALUATION=1 \
    sbatch --parsable --dependency=afterany:"${gat}" "${CPU_JOB}")

# An empty DGG_BENCHMARK_REQUIRE_METHODS scores every method already normalized
# into predictions/, so a comparator that failed upstream is simply absent from
# the figures rather than aborting the scoring pass.
score=$(DGG_BENCHMARK_METHODS=hybrid \
    DGG_BENCHMARK_REQUIRE_METHODS= \
    DGG_BENCHMARK_SKIP_EVALUATION=0 \
    sbatch --parsable --dependency=afterany:"${graph}" "${CPU_JOB}")

printf '%s\n' \
    "A setup (CPU, heal+gat_go+deepgraphgo): ${setup}" \
    "B HEAL (GPU):                           ${heal}" \
    "C GAT-GO (GPU):                         ${gat}" \
    "D DeepGraphGO (CPU):                    ${graph}" \
    "E evaluation and figures (CPU):         ${score}"

squeue -j "${setup},${heal},${gat},${graph},${score}" || true

echo
echo "Watch the setup repair first:"
echo "  tail -f logs/setup_sota_${setup}.out"
echo "A successful setup ends with:"
echo "  [SETUP COMPLETE] All requested SOTA dependencies and model files passed verification"
