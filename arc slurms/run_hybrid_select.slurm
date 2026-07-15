#!/usr/bin/env bash
#SBATCH --job-name=dgg_hybrid_select
#SBATCH --partition=cpu2022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_select_%j.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_select_%j.err

set -euo pipefail
PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
TUNING_ROOT="${DGG_TUNING_ROOT:-${DATA_ROOT}/arc_tuning}"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
python3 scripts/select_hybrid_trials.py \
  --runs-dir "${TUNING_ROOT}/hybrid_search" \
  --output "${TUNING_ROOT}/selected_hybrid.json" \
  --expected-trials "${DGG_EXPECTED_TRIALS:-40}"
