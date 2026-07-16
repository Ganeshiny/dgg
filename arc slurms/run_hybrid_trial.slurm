#!/usr/bin/env bash
#SBATCH --job-name=dgg_hybrid_trial
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_trial_%A_%a.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/hybrid_trial_%A_%a.err

set -euo pipefail
PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
TUNING_ROOT="${DGG_TUNING_ROOT:-${PROJECT_DIR}/arc_tuning}"
GRAPH_ROOT="${DGG_GRAPH_ROOT:-${PROJECT_DIR}/arc_tuning/graphs_protbert}"
TRIAL_CONFIGS="${DGG_TRIAL_CONFIGS:-${TUNING_ROOT}/hybrid_random_trials.jsonl}"
SELECTION_METRIC="${DGG_SELECTION_METRIC:-validation_micro_fmax}"
ONTOLOGIES=(molecular_function biological_process cellular_component)
TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"
TRIAL_ID=$((TASK_ID / 3))
ONTOLOGY="${ONTOLOGIES[$((TASK_ID % 3))]}"
RUN_DIR="${TUNING_ROOT}/hybrid_search/trial_$(printf '%03d' "${TRIAL_ID}")/${ONTOLOGY}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${RUN_DIR}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DGG_CONDA_ENV:-deepgreengo}"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DGG_GRAPH_ROOT="${GRAPH_ROOT}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export DGG_GRAPH_ROOT="${GRAPH_ROOT}"

python3 - <<'PY'
import subprocess, torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA tensor check:", (torch.ones(4, device="cuda") + 1).sum().item())
subprocess.run(["nvidia-smi"], check=True)
PY

python3 src/tune_hybrid.py \
  --dataset-dir "${TUNING_ROOT}/datasets" \
  --ontology "${ONTOLOGY}" \
  --trial-configs "${TRIAL_CONFIGS}" \
  --trial-id "${TRIAL_ID}" \
  --model Hybrid \
  --seed 42 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --selection-metric "${SELECTION_METRIC}" \
  --output-dir "${RUN_DIR}"
