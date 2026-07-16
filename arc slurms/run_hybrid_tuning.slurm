#!/usr/bin/env bash
#SBATCH --job-name=dgg_hybrid_setup
#SBATCH --partition=cpu2022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_tuning_controller_%j.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_tuning_controller_%j.err

set -euo pipefail

PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
LOG_DIR="${PROJECT_DIR}/logs"
MODE="${DGG_MODE:-tune}"
TRIALS="${DGG_TUNING_TRIALS:-40}"
CONCURRENCY="${DGG_TUNING_CONCURRENCY:-4}"
TUNING_ROOT="${DGG_TUNING_ROOT:-${PROJECT_DIR}/arc_tuning}"
TRIAL_CONFIGS="${TUNING_ROOT}/hybrid_random_trials.jsonl"
SELECTED_CONFIG="${TUNING_ROOT}/selected_hybrid.json"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${TUNING_ROOT}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DGG_CONDA_ENV:-deepgreengo}"
set -u
export PYTHONUNBUFFERED=1
export DGG_HF_LOCAL_ONLY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""

echo "Mode: ${MODE}"
echo "Project: ${PROJECT_DIR}"
echo "Data: ${DATA_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "Split: nominal 30% identity / 80% coverage split"

if [[ "${MODE}" == "tune" ]]; then
  python3 preprocessing/build_arc_tuning_dataset.py --data-root "${DATA_ROOT}" --tuning-root "${TUNING_ROOT}" --smoke-batches 1
  python3 scripts/generate_hybrid_trials.py --trials "${TRIALS}" --output "${TRIAL_CONFIGS}"
  unset CUDA_VISIBLE_DEVICES
  LAST_TASK=$((TRIALS * 3 - 1))
  ARRAY_JOB=$(sbatch --parsable \
    --array="0-${LAST_TASK}%${CONCURRENCY}" \
    --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TRIAL_CONFIGS=${TRIAL_CONFIGS},DGG_TUNING_ROOT=${TUNING_ROOT}" \
    "arc slurms/run_hybrid_trial.slurm")
  SELECT_JOB=$(sbatch --parsable \
    --dependency="afterok:${ARRAY_JOB}" \
    --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_EXPECTED_TRIALS=${TRIALS}" \
    "arc slurms/run_hybrid_select.slurm")
  REPEAT_JOB=$(sbatch --parsable \
    --dependency="afterok:${SELECT_JOB}" \
    --array="0-14%${CONCURRENCY}" \
    --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_REPEAT_MODEL=Hybrid" \
    "arc slurms/run_hybrid_repeat.slurm")
  echo "Submitted Hybrid search array: ${ARRAY_JOB} (40 configurations x 3 ontologies)"
  echo "Submitted validation-only selection: ${SELECT_JOB}"
  echo "Submitted five-seed Hybrid repeats: ${REPEAT_JOB}"
elif [[ "${MODE}" == "confirm_jk" ]]; then
  python3 preprocessing/build_arc_tuning_dataset.py --data-root "${DATA_ROOT}" --tuning-root "${TUNING_ROOT}" --validate-only --smoke-batches 1
  [[ -f "${SELECTED_CONFIG}" ]] || { echo "Missing selected Hybrid parameters: ${SELECTED_CONFIG}" >&2; exit 2; }
  unset CUDA_VISIBLE_DEVICES
  JK_JOB=$(sbatch --parsable \
    --array="0-14%${CONCURRENCY}" \
    --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_REPEAT_MODEL=Hybrid_JK" \
    "arc slurms/run_hybrid_repeat.slurm")
  echo "Submitted Hybrid-JK confirmatory five-seed ablation: ${JK_JOB}"
else
  echo "DGG_MODE must be 'tune' or 'confirm_jk', got '${MODE}'" >&2
  exit 2
fi
