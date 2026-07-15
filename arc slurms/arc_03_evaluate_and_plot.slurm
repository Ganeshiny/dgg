#!/bin/bash
# ==============================================================================
# arc_03_evaluate_and_plot.slurm
# ARC Cluster — Run Bootstrapping Evaluation and generate Comparison Plots
#
# Runs `evaluate_sota_5seeds.py` (which bootstraps the DeepFRI, TransFun, and
# DPFunc predictions and aggregates Hybrid/Hybrid_JK seeds).
# Then runs `plot_sota_comparison.py` to generate the publication figures.
#
# Submit with:
#   sbatch arc_03_evaluate_and_plot.slurm
#
# Monitor with:
#   squeue -u $USER
# ==============================================================================

#SBATCH --job-name=eval_sota
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32000
#SBATCH --output=logs/eval_sota_%j.out
#SBATCH --error=logs/eval_sota_%j.err

echo "========================================================"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : ${SLURM_NODELIST}"
echo "  Start time  : $(date)"
echo "========================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dgg_sota_torch

PROJECT_DIR=$(pwd)
cd ${PROJECT_DIR}

echo "1. Generating BLAST maximum sequence identity (for Plot A)"
# Make sure blast+ is available (if ARC has a module system)
# module load blast+  # Uncomment this line if ARC requires it
python preprocessing/calc_blast_identity.py

echo "2. Running 5-seed Evaluation and Bootstrapping (Mode: all)"
python sota/evaluate_sota_5seeds.py --bootstrap_seeds 5 --mode all
python sota/evaluate_sota_5seeds.py --bootstrap_seeds 5 --mode all --common_subset

echo "3. Generating Publication Comparison Plots (Mode: all)"
python plots/plot_sota_comparison.py --mode all
python plots/plot_sota_comparison.py --mode all --common_subset
python plots/plot_sota_comparison.py --mode all --supplementary
python plots/plot_sota_comparison.py --mode all --common_subset --supplementary

echo "4. Running 5-seed Evaluation and Bootstrapping (Mode: baselines_only)"
python sota/evaluate_sota_5seeds.py --bootstrap_seeds 5 --mode baselines_only
python sota/evaluate_sota_5seeds.py --bootstrap_seeds 5 --mode baselines_only --common_subset

echo "5. Generating Publication Comparison Plots (Mode: baselines_only)"
python plots/plot_sota_comparison.py --mode baselines_only
python plots/plot_sota_comparison.py --mode baselines_only --common_subset
python plots/plot_sota_comparison.py --mode baselines_only --supplementary
python plots/plot_sota_comparison.py --mode baselines_only --common_subset --supplementary

echo "========================================================"
echo "  Evaluation and Plotting Complete!"
echo "  End time  : $(date)"
echo "========================================================"
exit 0
