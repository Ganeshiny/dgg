#!/bin/bash
# ==============================================================================
# arc_submit_5seeds.slurm
# ARC Cluster — Train Hybrid & Hybrid_JK for 5 seeds × 3 ontologies.
#
# This is a SLURM job ARRAY covering 30 individual jobs:
#   - 2 models  (Hybrid, Hybrid_JK)
#   - 3 ontologies (bp, mf, cc)
#   - 5 seeds   (42, 43, 44, 45, 46)
#
# Submit with:
#   sbatch --array=0-29 arc_submit_5seeds.slurm
#
# Monitor with:
#   squeue -u $USER
#
# Output layout:
#   runs_5seeds/{ont}/{model}/{seed}/
#       training_log.csv
#       best_model.pth
#       test_metrics.json
#       test_y_true.npy
#       test_y_pred.npy
# ==============================================================================

#SBATCH --job-name=dgg_5seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/5seeds_%A_%a.out
#SBATCH --error=logs/5seeds_%A_%a.err

# ── diagnostics ───────────────────────────────────────────────────────────────
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "  Job ID      : ${SLURM_JOB_ID}  (array task ${SLURM_ARRAY_TASK_ID})"
echo "  Node        : ${SLURM_NODELIST}"
echo "  CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "  Start time  : $(date)"
echo "  Project dir : ${PROJECT_DIR}"
echo "========================================================"

# ── environment ──────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepgreengo \
  || { echo "ERROR: 'deepgreengo' conda env not found. Aborting."; exit 1; }

# ── job matrix ────────────────────────────────────────────────────────────────
# We enumerate all (model, ontology, seed) combinations and index by
# SLURM_ARRAY_TASK_ID (0-based, 0..29).
#
# Index layout:
#   models    : [Hybrid, Hybrid_JK]             — 2
#   ontologies: [biological_process, mf, cc]    — 3
#   seeds     : [42, 43, 44, 45, 46]            — 5
#   Total     : 2 × 3 × 5 = 30
#
MODELS=("Hybrid" "Hybrid_JK")
ONTOLOGIES=("biological_process" "molecular_function" "cellular_component")
SEEDS=(42 43 44 45 46)

N_ONTS=${#ONTOLOGIES[@]}      # 3
N_SEEDS=${#SEEDS[@]}          # 5

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Decompose TASK_ID → (model_idx, ont_idx, seed_idx)
model_idx=$(( TASK_ID / (N_ONTS * N_SEEDS) ))
rem=$(( TASK_ID % (N_ONTS * N_SEEDS) ))
ont_idx=$(( rem / N_SEEDS ))
seed_idx=$(( rem % N_SEEDS ))

MODEL="${MODELS[$model_idx]}"
ONTOLOGY="${ONTOLOGIES[$ont_idx]}"
SEED="${SEEDS[$seed_idx]}"

echo "  Model     : ${MODEL}"
echo "  Ontology  : ${ONTOLOGY}"
echo "  Seed      : ${SEED}"
echo "========================================================"

# ── best hyperparameters (from completed tuning runs) ─────────────────────────
# Hardcoded from tuning_results_summary.csv (best Val_Micro_Fmax per ontology).
# Hybrid_JK best params:
#   BP: lr=0.001,   dropout=0.2, bs=16
#   MF: lr=0.0005,  dropout=0.2, bs=32
#   CC: lr=0.0001,  dropout=0.3, bs=32
# Hybrid best params:
#   BP: lr=0.0001,  dropout=0.2, bs=64
#   MF: lr=0.0005,  dropout=0.3, bs=32
#   CC: lr=0.0001,  dropout=0.2, bs=32

if [ "${MODEL}" == "Hybrid_JK" ]; then
    case "${ONTOLOGY}" in
        "biological_process")  LR=0.001;   DROPOUT=0.2; BATCH_SIZE=16 ;;
        "molecular_function")  LR=0.0005;  DROPOUT=0.2; BATCH_SIZE=32 ;;
        "cellular_component")  LR=0.0001;  DROPOUT=0.3; BATCH_SIZE=32 ;;
    esac
elif [ "${MODEL}" == "Hybrid" ]; then
    case "${ONTOLOGY}" in
        "biological_process")  LR=0.0001;  DROPOUT=0.2; BATCH_SIZE=64 ;;
        "molecular_function")  LR=0.0005;  DROPOUT=0.3; BATCH_SIZE=32 ;;
        "cellular_component")  LR=0.0001;  DROPOUT=0.2; BATCH_SIZE=32 ;;
    esac
fi

# ── short ontology name for output directory ──────────────────────────────────
case "${ONTOLOGY}" in
    "biological_process") ONT_SHORT="bp" ;;
    "molecular_function") ONT_SHORT="mf" ;;
    "cellular_component") ONT_SHORT="cc" ;;
esac

OUTPUT_DIR="${PROJECT_DIR}/runs_5seeds/${ONT_SHORT}/${MODEL}/${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "  LR=${LR}  Dropout=${DROPOUT}  BatchSize=${BATCH_SIZE}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "========================================================"

# ── run training ─────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

export CMAP_THREADS=${SLURM_CPUS_PER_TASK}

python3 train.py \
    --model          "${MODEL}"    \
    --loss           "Focal"       \
    --seed           "${SEED}"     \
    --ontology       "${ONTOLOGY}" \
    --epochs         1000          \
    --lr             "${LR}"       \
    --dropout        "${DROPOUT}"  \
    --batch_size     "${BATCH_SIZE}" \
    --output_dir     "${OUTPUT_DIR}" \
    --patience       20

EXIT_CODE=$?
echo "========================================================"
echo "  Training finished with exit code: ${EXIT_CODE}"
echo "  End time  : $(date)"
echo "========================================================"
exit ${EXIT_CODE}
