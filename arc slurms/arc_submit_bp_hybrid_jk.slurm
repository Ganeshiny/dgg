#!/bin/bash
# ==============================================================================
# ARC Cluster Submission Script for DeepGreenGO (BP Hybrid_JK ONLY)
# ==============================================================================

#SBATCH --job-name=deepgreengo_bp_fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --mem=128000
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bp_fast_%j.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepgreengo \
  || { echo "ERROR: 'deepgreengo' conda env not found. Aborting."; exit 1; }

echo "--- Starting BP Hybrid_JK Sweep ---"

ONT="biological_process"
MODELS=("Hybrid_JK")
LOSSES=("BCE" "Focal")
SEEDS=(42 123 456)

BEST_PARAMS=$(python3 get_best_hyperparams.py --ontology "$ONT" --summary_file "tuning_runs_jk/tuning_results_summary.csv")
eval "$BEST_PARAMS"

for model in "${MODELS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running: ${ONT}_${model}_${loss}_s${seed}"
            python3 train.py \
                --model "$model" \
                --loss "$loss" \
                --seed "$seed" \
                --ontology "$ONT" \
                --epochs 1000 \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --dropout "$DROPOUT"
        done
    done
done
