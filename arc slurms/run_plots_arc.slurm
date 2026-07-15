#!/bin/bash
#SBATCH --job-name=sota_plots
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32000
#SBATCH --output=sota_plots_%j.out
#SBATCH --error=sota_plots_%j.err

echo "========================================================"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : ${SLURM_NODELIST}"
echo "  Start time  : $(date)"
echo "========================================================"

# Activate the exact environment used on ARC
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dgg_sota_torch

PROJECT_DIR=$(pwd)
cd ${PROJECT_DIR}

echo "Starting plotting script (Common Subset)..."
# Corrected argument: uses underscore instead of dash!
python plots/plot_sota_comparison.py --common_subset

echo "========================================================"
echo "  Plotting Complete!"
echo "  End time  : $(date)"
echo "========================================================"
exit 0
