#!/usr/bin/env bash
# Submit one independent retuning/test pipeline for each alternative split.
# The controller array task builds PKLs, then submits trial, selection, repeat,
# and held-out test jobs with dependencies.
# Submit with: sbatch --array=0-4%1 --export=ALL,DGG_THRESHOLD_LIST=40,50,70,90,95 ...
#              "arc slurms/run_threshold_pipeline.slurm"
#SBATCH --job-name=dgg_threshold_ctl
#SBATCH --partition=cpu2022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --array=0-4%1
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/threshold_ctl_%A_%a.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/threshold_ctl_%A_%a.err

set -euo pipefail
PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
GRAPH_ROOT="${DGG_GRAPH_ROOT:-${PROJECT_DIR}/arc_tuning/graphs_protbert}"
THRESHOLDS=(40 50 70 90 95)
TRIALS="${DGG_TUNING_TRIALS:-40}"
CONCURRENCY="${DGG_TUNING_CONCURRENCY:-4}"
SELECTION_METRIC="${DGG_SELECTION_METRIC:-validation_macro_fmax}"

THRESHOLD="${THRESHOLDS[${SLURM_ARRAY_TASK_ID}]:?No threshold for array index ${SLURM_ARRAY_TASK_ID}}"
TUNING_ROOT="${PROJECT_DIR}/arc_tuning_threshold_${THRESHOLD}"
SPLIT_LABEL="identity_${THRESHOLD}_coverage_80"
TRIAL_CONFIGS="${TUNING_ROOT}/hybrid_random_trials.jsonl"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${TUNING_ROOT}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DGG_CONDA_ENV:-deepgreengo}"
set -u
export PYTHONUNBUFFERED=1 DGG_HF_LOCAL_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""

echo "Threshold: ${THRESHOLD}% identity / 80% coverage"
echo "Tuning root: ${TUNING_ROOT}"

python3 preprocessing/build_threshold_arc_dataset.py \
  --data-root "${DATA_ROOT}" --tuning-root "${TUNING_ROOT}" \
  --graph-root "${GRAPH_ROOT}" --threshold "${THRESHOLD}" --coverage 0.8

python3 scripts/generate_hybrid_trials.py --trials "${TRIALS}" --output "${TRIAL_CONFIGS}"
unset CUDA_VISIBLE_DEVICES
LAST_TASK=$((TRIALS * 3 - 1))

ARRAY_JOB=$(sbatch --parsable --array="0-${LAST_TASK}%${CONCURRENCY}" \
  --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TRIAL_CONFIGS=${TRIAL_CONFIGS},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_GRAPH_ROOT=${GRAPH_ROOT},DGG_SELECTION_METRIC=${SELECTION_METRIC}" \
  "arc slurms/run_hybrid_trial.slurm")
SELECT_JOB=$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}" \
  --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_EXPECTED_TRIALS=${TRIALS},DGG_SELECTION_METRIC=${SELECTION_METRIC}" \
  "arc slurms/run_hybrid_select.slurm")
REPEAT_JOB=$(sbatch --parsable --dependency="afterok:${SELECT_JOB}" --array="0-14%${CONCURRENCY}" \
  --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_GRAPH_ROOT=${GRAPH_ROOT},DGG_REPEAT_MODEL=Hybrid" \
  "arc slurms/run_hybrid_repeat.slurm")
TEST_JOB=$(sbatch --parsable --dependency="afterok:${REPEAT_JOB}" \
  --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_GRAPH_ROOT=${GRAPH_ROOT},DGG_EVAL_MODEL=Hybrid" \
  "arc slurms/run_hybrid_test_eval.slurm")

echo "Threshold ${THRESHOLD}: trials=${ARRAY_JOB} select=${SELECT_JOB} repeats=${REPEAT_JOB} test=${TEST_JOB}"
