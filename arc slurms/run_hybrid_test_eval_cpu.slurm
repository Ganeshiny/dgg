#!/usr/bin/env bash
#SBATCH --job-name=dgg_hybrid_test_cpu
#SBATCH --partition=cpu2022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_test_cpu_%j.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_test_cpu_%j.err

set -euo pipefail
PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
TUNING_ROOT="${DGG_TUNING_ROOT:-${PROJECT_DIR}/arc_tuning}"
MODEL="${DGG_EVAL_MODEL:-Hybrid}"
CHECKPOINT_ROOT="${TUNING_ROOT}/five_seed_$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
OUTPUT_DIR="${TUNING_ROOT}/test_evaluation/$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')_cpu"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DGG_CONDA_ENV:-deepgreengo}"
set -u
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python3 scripts/evaluate_arc_test.py --data-root "${DATA_ROOT}" --tuning-root "${TUNING_ROOT}" --checkpoints-root "${CHECKPOINT_ROOT}" --output-dir "${OUTPUT_DIR}" --model "${MODEL}" --batch-size 8 --workers 0