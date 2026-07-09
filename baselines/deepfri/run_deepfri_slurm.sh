#!/bin/bash
#SBATCH --job-name=deepfri_eval
#SBATCH --output=deepfri_eval_%j.out
#SBATCH --error=deepfri_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Activate your deepgreengo environment (or whichever env has TensorFlow/PyTorch)
# Adjust this path if your miniconda is elsewhere
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepgreengo

# Move to the root directory of the project
cd ~/dgg/deep-green-GO

echo "Starting DeepFRI baseline evaluation..."
# Run the baseline script (it automatically runs both 'seq' and 'cmap' modes for all 3 ontologies)
python baselines/deepfri/run_deepfri_baseline.py

echo "Finished DeepFRI baseline evaluation!"
