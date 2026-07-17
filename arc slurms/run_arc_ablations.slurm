#!/usr/bin/env bash
# Train model/input ablations after the selected five-seed checkpoints exist.
# Array layout: 3 ontologies x 5 models x 3 input modes = 45 tasks.
# Set DGG_TUNING_ROOT to arc_tuning_cafa for the CAFA-selected pass.
# Set DGG_DATASET_DIR to an alternate dataset directory for a homology-split pass.
#SBATCH --job-name=dgg_arc_ablate
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-44%6
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/arc_ablation_%A_%a.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/arc_ablation_%A_%a.err

set -euo pipefail
PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
TUNING_ROOT="${DGG_TUNING_ROOT:-${PROJECT_DIR}/arc_tuning}"
GRAPH_ROOT="${DGG_GRAPH_ROOT:-${PROJECT_DIR}/arc_tuning/graphs_protbert}"
DATASET_DIR="${DGG_DATASET_DIR:-${TUNING_ROOT}/datasets}"
CONFIG_JSON="${DGG_CONFIG_JSON:-${TUNING_ROOT}/selected_hybrid.json}"
SPLIT_LABEL="${DGG_SPLIT_LABEL:-nominal_30_identity_80_coverage}"
ONTOLOGIES=(molecular_function biological_process cellular_component)
MODELS=(MLP GCN GAT Hybrid Hybrid_JK)
MODALITIES=(full seq_only struct_only)
TASK_ID="${SLURM_ARRAY_TASK_ID:?array required}"
ONTOLOGY="${ONTOLOGIES[$((TASK_ID / 15))]}"
MODEL="${MODELS[$(((TASK_ID / 3) % 5))]}"
MODALITY="${MODALITIES[$((TASK_ID % 3))]}"
OUT="${TUNING_ROOT}/ablations/${SPLIT_LABEL}/${ONTOLOGY}/${MODEL}/${MODALITY}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${OUT}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DGG_CONDA_ENV:-deepgreengo}"
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" DGG_GRAPH_ROOT="${GRAPH_ROOT}"

python3 src/train_arc_ablation.py \
  --dataset-dir "${DATASET_DIR}" --graph-root "${GRAPH_ROOT}" \
  --ontology "${ONTOLOGY}" --model "${MODEL}" --input-modality "${MODALITY}" \
  --split-label "${SPLIT_LABEL}" --config-json "${CONFIG_JSON}" \
  --output-dir "${OUT}" --workers "${SLURM_CPUS_PER_TASK}"
