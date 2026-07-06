#!/bin/bash
# ==============================================================================
# arc_01_setup_sota.slurm
# ARC Cluster — Setup Environment for SOTA Baselines
#
# This script creates a unified Conda environment ('dgg_sota') that contains
# all dependencies needed to run DeepFRI, TransFun, and DPFunc natively on ARC.
#
# Submit with:
#   sbatch arc_01_setup_sota.slurm
#
# Monitor with:
#   squeue -u $USER
# ==============================================================================

#SBATCH --job-name=setup_sota
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32000
#SBATCH --output=logs/setup_sota_%j.out
#SBATCH --error=logs/setup_sota_%j.err

echo "========================================================"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : ${SLURM_NODELIST}"
echo "  Start time  : $(date)"
echo "========================================================"

# Source conda
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="dgg_sota"

echo "1. Checking for existing ${ENV_NAME} environment..."
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "  Environment ${ENV_NAME} already exists. Updating/Ensuring packages..."
    conda activate ${ENV_NAME}
else
    echo "  Creating new conda environment: ${ENV_NAME} (Python 3.9)..."
    conda create -y -n ${ENV_NAME} python=3.9
    conda activate ${ENV_NAME}
fi

echo "2. Installing Core Packages via Conda (PyTorch, Diamond)..."
# PyTorch (for DPFunc and TransFun) and Diamond (for TransFun)
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c bioconda diamond
conda install -y -c conda-forge openbabel  # often helpful for chemo/bio dependencies

echo "3. Installing Packages via Pip (TensorFlow, DGL, ESM, Biopython)..."
# TensorFlow is needed for DeepFRI
# DGL and fair-esm are needed for DPFunc
# Networkx, Biopython, pandas, scikit-learn are universally needed
pip install --upgrade pip
pip install tensorflow==2.13.0
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install fair-esm
pip install networkx biopython pandas scikit-learn tqdm h5py

echo "========================================================"
echo "  Setup Complete!"
echo "  End time  : $(date)"
echo "========================================================"
exit 0
