#!/usr/bin/env bash
# Resume an interrupted threshold-specific Hybrid pipeline without rerunning
# completed random-search trials.
# ARC: bash resume_threshold_pipeline.sh 90

set -euo pipefail

THRESHOLD="${1:-}"
case "${THRESHOLD}" in
  40|50|70|90|95) ;;
  *) echo "Usage: $0 {40|50|70|90|95}" >&2; exit 2 ;;
esac

PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
GRAPH_ROOT="${DGG_GRAPH_ROOT:-${PROJECT_DIR}/arc_tuning/graphs_protbert}"
TUNING_ROOT="${DGG_TUNING_ROOT:-${PROJECT_DIR}/arc_tuning_threshold_${THRESHOLD}}"
TRIAL_CONFIGS="${DGG_TRIAL_CONFIGS:-${TUNING_ROOT}/hybrid_random_trials.jsonl}"
TRIALS="${DGG_TUNING_TRIALS:-40}"
CONCURRENCY="${DGG_TUNING_CONCURRENCY:-4}"
SELECTION_METRIC="${DGG_SELECTION_METRIC:-validation_macro_fmax}"
DRY_RUN="${DGG_RESUME_DRY_RUN:-0}"
ONTOLOGIES=(molecular_function biological_process cellular_component)

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
[[ -s "${TRIAL_CONFIGS}" ]] || { echo "Missing trial configurations: ${TRIAL_CONFIGS}" >&2; exit 1; }
CONFIG_COUNT=$(wc -l < "${TRIAL_CONFIGS}")
(( CONFIG_COUNT >= TRIALS )) || { echo "Expected ${TRIALS} trial configurations; found ${CONFIG_COUNT}" >&2; exit 1; }

for ontology in "${ONTOLOGIES[@]}"; do
  for split in train valid test; do
    dataset="${TUNING_ROOT}/datasets/${ontology}_${split}.pkl"
    [[ -s "${dataset}" ]] || { echo "Missing threshold-${THRESHOLD} dataset: ${dataset}" >&2; exit 1; }
  done
done

# Selection consumes these two artifacts. Retry only tasks lacking either one.
missing_tasks=()
for ((task_id = 0; task_id < TRIALS * 3; task_id++)); do
  trial_id=$((task_id / 3))
  ontology="${ONTOLOGIES[$((task_id % 3))]}"
  printf -v trial_dir 'trial_%03d' "${trial_id}"
  run_dir="${TUNING_ROOT}/hybrid_search/${trial_dir}/${ontology}"
  if [[ ! -s "${run_dir}/config.json" || ! -s "${run_dir}/validation_metrics.json" ]]; then
    missing_tasks+=("${task_id}")
  fi
done

missing_spec=""
if ((${#missing_tasks[@]})); then
  missing_spec=$(IFS=,; echo "${missing_tasks[*]}")
  echo "Threshold ${THRESHOLD}: retrying missing tuning tasks ${missing_spec}"
else
  echo "Threshold ${THRESHOLD}: all $((TRIALS * 3)) tuning tasks are complete"
fi
[[ "${DRY_RUN}" != "1" ]] || { echo "Dry run; no jobs submitted."; exit 0; }

retry_job=""
if [[ -n "${missing_spec}" ]]; then
  retry_raw=$(sbatch --parsable --array="${missing_spec}%${CONCURRENCY}" \
    --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TRIAL_CONFIGS=${TRIAL_CONFIGS},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_GRAPH_ROOT=${GRAPH_ROOT},DGG_SELECTION_METRIC=${SELECTION_METRIC}" \
    "arc slurms/run_hybrid_trial.slurm")
  retry_job="${retry_raw%%;*}"
fi

select_args=(--parsable)
[[ -z "${retry_job}" ]] || select_args+=(--dependency="afterok:${retry_job}")
select_raw=$(sbatch "${select_args[@]}" \
  --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_EXPECTED_TRIALS=${TRIALS},DGG_SELECTION_METRIC=${SELECTION_METRIC}" \
  "arc slurms/run_hybrid_select.slurm")
select_job="${select_raw%%;*}"

repeat_raw=$(sbatch --parsable --dependency="afterok:${select_job}" --array="0-14%${CONCURRENCY}" \
  --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_GRAPH_ROOT=${GRAPH_ROOT},DGG_REPEAT_MODEL=Hybrid" \
  "arc slurms/run_hybrid_repeat.slurm")
repeat_job="${repeat_raw%%;*}"
test_raw=$(sbatch --parsable --dependency="afterok:${repeat_job}" \
  --export="ALL,DGG_PROJECT_ROOT=${PROJECT_DIR},DGG_DATA_ROOT=${DATA_ROOT},DGG_TUNING_ROOT=${TUNING_ROOT},DGG_GRAPH_ROOT=${GRAPH_ROOT},DGG_EVAL_MODEL=Hybrid" \
  "arc slurms/run_hybrid_test_eval.slurm")
test_job="${test_raw%%;*}"

echo "Resume submitted: threshold=${THRESHOLD} retry=${retry_job:-not-needed} select=${select_job} repeats=${repeat_job} test=${test_job}"
echo "Old blocked jobs are not cancelled automatically."
