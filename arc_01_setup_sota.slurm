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

ENV_TF="dgg_sota_tf"
ENV_TORCH="dgg_sota_torch"

echo "1. Setting up DeepFRI Environment (TensorFlow) -> ${ENV_TF}"
if conda info --envs | grep -q "^${ENV_TF} "; then
    echo "  Environment ${ENV_TF} exists."
else
    conda create -y -n ${ENV_TF} python=3.9
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_TF}
pip install --upgrade pip
pip install tensorflow==2.13.0 networkx biopython pandas scikit-learn tqdm h5py

echo "2. Setting up DPFunc & TransFun Environment (PyTorch) -> ${ENV_TORCH}"
if conda info --envs | grep -q "^${ENV_TORCH} "; then
    echo "  Environment ${ENV_TORCH} exists."
else
    conda create -y -n ${ENV_TORCH} python=3.9
fi
conda activate ${ENV_TORCH}
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c bioconda diamond
conda install -y -c conda-forge openbabel
pip install --upgrade pip
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install fair-esm networkx biopython pandas scikit-learn tqdm h5py matplotlib seaborn obonet torchdata ruamel.yaml

echo "========================================================"
echo "  Setup Complete!"
echo "  End time  : $(date)"
echo "========================================================"
exit 0
