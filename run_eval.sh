#!/bin/bash
#SBATCH --job-name=eval_sota
#SBATCH --output=eval_sota_%j.out
#SBATCH --error=eval_sota_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepgreengo

# Run the evaluation and plotting
python plot_sota_comparison.py
