#!/usr/bin/env bash
# Submit held-out evaluation -> model/input ablations -> homology/IC bins.
# Usage: submit_arc_post_tuning.sh <repeat_job_id> [tuning_root]
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <repeat_job_id> [tuning_root]" >&2
  exit 2
fi

PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
REPEAT_JOB="$1"
TUNING_ROOT="${2:-${DGG_TUNING_ROOT:-${PROJECT_DIR}/arc_tuning_cafa}}"
GRAPH_ROOT="${DGG_GRAPH_ROOT:-${PROJECT_DIR}/arc_tuning/graphs_protbert}"
DEPENDENCY_TYPE="${DGG_DEPENDENCY_TYPE:-afterok}"

cd "${PROJECT_DIR}"

TEST_JOB=$(sbatch --parsable --dependency="${DEPENDENCY_TYPE}:${REPEAT_JOB}" \
  --export="ALL,DGG_TUNING_ROOT=${TUNING_ROOT},DGG_GRAPH_ROOT=${GRAPH_ROOT}" \
  "arc slurms/run_hybrid_test_eval.slurm")

ABLATION_JOB=$(sbatch --parsable --dependency="${DEPENDENCY_TYPE}:${REPEAT_JOB}" \
  --export="ALL,DGG_TUNING_ROOT=${TUNING_ROOT},DGG_GRAPH_ROOT=${GRAPH_ROOT}" \
  "arc slurms/run_arc_ablations.slurm")

BIN_JOB=$(sbatch --parsable --dependency="afterok:${ABLATION_JOB}" \
  --export="ALL,DGG_TUNING_ROOT=${TUNING_ROOT},DGG_GRAPH_ROOT=${GRAPH_ROOT}" \
  "arc slurms/run_arc_bin_eval.slurm")

echo "Held-out test evaluation: ${TEST_JOB} (after ${REPEAT_JOB})"
echo "Model/input ablations:    ${ABLATION_JOB} (after ${REPEAT_JOB})"
echo "Homology/IC bin analysis: ${BIN_JOB} (after ${ABLATION_JOB})"
