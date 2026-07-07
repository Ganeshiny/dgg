#!/bin/bash
# ==============================================================================
# arc_02_predict_sota.slurm
# ARC Cluster — Generate Inference Predictions for SOTA Baselines
#
# Runs inference for DeepFRI, TransFun, and DPFunc sequentially.
#
# Submit with:
#   sbatch arc_02_predict_sota.slurm
#
# Monitor with:
#   squeue -u $USER
# ==============================================================================

#SBATCH --job-name=predict_sota
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/predict_sota_%j.out
#SBATCH --error=logs/predict_sota_%j.err

echo "========================================================"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : ${SLURM_NODELIST}"
echo "  Start time  : $(date)"
echo "========================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"

PROJECT_DIR=$(pwd)
mkdir -p baselines/deepfri_results
mkdir -p SOTA_predictions/DPFunc

echo "========================================================"
echo "  [1/3] DeepFRI Inference"
echo "========================================================"
# The prep script needs PyTorch to read the .pt files, so use the torch env
conda activate dgg_sota_torch
cd baselines/DeepFRI

# Generate fasta for test sequences using the dataset pickle
cat << 'EOF' > prep_deepfri_fasta.py
import os, pickle
PKL = '../../preprocessing/data/split_files/datasets.pkl'
OUT = 'test_seqs.fasta'
class _Up(pickle.Unpickler):
    def find_class(self, m, n):
        if m == 'numpy._core.multiarray': m = 'numpy.core.multiarray'
        return super().find_class(m, n)
import __main__
from preprocessing.create_batch_dataset import PDB_Dataset
__main__.PDB_Dataset = PDB_Dataset
with open(PKL, 'rb') as f: ds = _Up(f).load()
test_prots = ds['biological_process']['test'].pdb_split_list
import torch
recs = []
for p in test_prots:
    pt = f'../../preprocessing/data/processed/{p}.pt'
    if not os.path.exists(pt): continue
    try:
        d = torch.load(pt, weights_only=False, map_location='cpu')
        seq = d.sequence if hasattr(d, 'sequence') else d['atoms'].sequence
        recs.append(f">{p}\n{seq}")
    except: pass
with open(OUT, 'w') as f: f.write('\n'.join(recs))
print(f"Generated {OUT} with {len(recs)} sequences.")
EOF

python prep_deepfri_fasta.py

# Now switch to TF environment for actual DeepFRI inference
conda activate dgg_sota_tf

for ont in bp mf cc; do
    echo "Running DeepFRI for ${ont}..."
    python predict.py --fasta_fn test_seqs.fasta -ont ${ont} -o ../deepfri_results/deepfri_seq_${ont^^}
done


echo "========================================================"
echo "  [2/3] TransFun Inference"
echo "========================================================"
conda activate dgg_sota_torch
cd ${PROJECT_DIR}/SOTA/TransFun
for ont in bp mf cc; do
    case "${ont}" in
        "bp") full_ont="biological_process" ;;
        "mf") full_ont="molecular_function" ;;
        "cc") full_ont="cellular_component" ;;
    esac
    echo "Running TransFun for ${full_ont}..."
    python predict.py --data-path data --ontology ${full_ont} --input-type pdb --pdb-path ../../SOTA/TransFun/test_pdbs/ --output ${ont}_results.txt
done


echo "========================================================"
echo "  [3/3] DPFunc Inference"
echo "========================================================"
cd ${PROJECT_DIR}/SOTA/DPFunc

for ont in bp mf cc; do
    echo "Building workspace for ${ont}..."
    python DPFunc_demo_pipeline/scripts/build_data_demo.py \
        --manifest manifest.tsv \
        --workspace workspace_${ont} \
        --ontology ${ont} \
        --pdb-dir ${PROJECT_DIR}/SOTA/TransFun/test_pdbs \
        --inter-idx data/inter_idx.pkl \
        --go-obo data/go.obo \
        --esm-batch-size 1

    echo "Running DPFunc for ${ont}..."
    # DPFunc expects save_models and mlb inside the workspace or properly linked
    rm -rf workspace_${ont}/save_models workspace_${ont}/mlb
    ln -s $(pwd)/save_models workspace_${ont}/save_models
    ln -s $(pwd)/mlb workspace_${ont}/mlb

    python DPFunc_demo_pipeline/scripts/run_dpfunc.py predict \
        --workspace workspace_${ont} \
        --ontology ${ont} \
        --gpu -1 \
        --pre-name DPFunc_model

    # Copy results to SOTA_predictions for evaluate script
    cp workspace_${ont}/results/DPFunc_model_${ont}_final.pkl ${PROJECT_DIR}/SOTA_predictions/DPFunc/${ont}_results.pkl
    # DPFunc saves as pkl (pandas DF) usually, we need to convert to text for the evaluate script
    cat << 'EOF' > convert_dpfunc.py
import sys, pandas as pd
df = pd.read_pickle(sys.argv[1])
# DPFunc output format: Index is protein, columns are GO terms
with open(sys.argv[2], 'w') as f:
    for prot in df.index:
        for go_term in df.columns:
            score = df.loc[prot, go_term]
            f.write(f"{prot} {go_term} {score}\n")
EOF
    python convert_dpfunc.py ${PROJECT_DIR}/SOTA_predictions/DPFunc/${ont}_results.pkl ${PROJECT_DIR}/SOTA_predictions/DPFunc/${ont}_results.txt
done

echo "========================================================"
echo "  All Predictions Complete!"
echo "  End time  : $(date)"
echo "========================================================"
exit 0
