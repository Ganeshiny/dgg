#!/bin/bash
#SBATCH --job-name=deepgreen-loss
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/loss_ablations_%A_%a.out

# Activate conda environment (adjust path if needed)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepgreengo \
  || { echo "ERROR: 'deepgreengo' conda env not found. Aborting."; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting Loss Function Ablation on $(hostname)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Assuming you want to split by ontology for the array
ONTOLOGIES=("biological_process" "molecular_function" "cellular_component")
CURRENT_ONT=${ONTOLOGIES[$SLURM_ARRAY_TASK_ID]}

echo "Running for ontology: $CURRENT_ONT"
bash run_ablation_loss.sh "$CURRENT_ONT"
