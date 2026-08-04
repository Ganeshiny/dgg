#!/usr/bin/env bash
# Evaluate every completed model/input ablation in homology and IC bins.
# Submit after run_arc_ablations.slurm with --dependency=afterok:<jobid>.
#SBATCH --job-name=dgg_arc_bins
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/arc_bin_eval_%j.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/arc_bin_eval_%j.err

set -euo pipefail
PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
TUNING_ROOT="${DGG_TUNING_ROOT:-${PROJECT_DIR}/arc_tuning_cafa}"
GRAPH_ROOT="${DGG_GRAPH_ROOT:-${PROJECT_DIR}/arc_tuning_cafa/graphs_protbert}"
SPLIT_LABEL="${DGG_SPLIT_LABEL:-nominal_30_identity_80_coverage}"
ABLATIONS_ROOT="${DGG_ABLATIONS_ROOT:-${TUNING_ROOT}/ablations/${SPLIT_LABEL}}"
OUTPUT_DIR="${DGG_BIN_OUTPUT:-${PROJECT_DIR}/plots/arc_tuning_cafa/bin_evaluation}"
OVERALL_OUTPUT="${DGG_ABLATION_ARCHIVE:-${PROJECT_DIR}/plots/arc_tuning_cafa/ablations/ablation_test_metrics.csv}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DGG_CONDA_ENV:-deepgreengo}"
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" DGG_GRAPH_ROOT="${GRAPH_ROOT}"

python3 src/evaluate_arc_bins.py \
  --data-root "${DATA_ROOT}" --tuning-root "${TUNING_ROOT}" \
  --ablations-root "${ABLATIONS_ROOT}" --graph-root "${GRAPH_ROOT}" \
  --output-dir "${OUTPUT_DIR}" --overall-output "${OVERALL_OUTPUT}" --splits valid test \
  --threads "${SLURM_CPUS_PER_TASK}" --workers "${SLURM_CPUS_PER_TASK}"
