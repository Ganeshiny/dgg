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
# 1. Standard full evaluation (All Models)
python sota/evaluate_sota_5seeds.py
python plots/plot_sota_comparison.py --mode all

# 2. Fair "Common Subset" evaluation (Baselines Only)
python sota/evaluate_sota_5seeds.py --common_subset
python plots/plot_sota_comparison.py --mode baselines_only --common_subset
