#!/bin/bash
echo "Creating deepfri environment..."
conda create -n deepfri python=3.9 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepfri

echo "Installing DeepFRI dependencies..."
pip install tensorflow==2.10.1 "numpy<2" scikit-learn biopython tqdm

echo "DeepFRI environment setup complete!"
