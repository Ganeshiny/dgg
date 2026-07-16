#!/usr/bin/env bash
#SBATCH --job-name=dgg_hybrid_test
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_test_%j.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_test_%j.err

set -euo pipefail
PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
TUNING_ROOT="${DATA_ROOT}/arc_tuning"
MODEL="${DGG_EVAL_MODEL:-Hybrid}"
CHECKPOINT_ROOT="${TUNING_ROOT}/five_seed_$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="${TUNING_ROOT}/test_evaluation/$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DGG_CONDA_ENV:-deepgreengo}"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python3 scripts/evaluate_arc_test.py \
  --data-root "${DATA_ROOT}" \
  --checkpoints-root "${CHECKPOINT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --batch-size 16 \
  --workers "${SLURM_CPUS_PER_TASK}"
