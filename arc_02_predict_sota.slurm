#!/bin/bash
# ==============================================================================
# arc_02_predict_sota.slurm
# ARC Cluster — Generate Inference Predictions for SOTA Baselines
#
# Runs inference for DeepFRI, TransFun, and DPFunc sequentially.
# Environments:
#   dgg_sota_torch — used for fasta prep (needs torch) and TransFun/DPFunc
#   dgg_sota_tf    — used for DeepFRI inference (needs tensorflow)
#
# Output paths (where evaluate_sota_5seeds.py and plot_sota_comparison.py look):
#   DeepFRI:  baselines/deepfri_results/deepfri_seq_{BP,MF,CC}_pred_scores.json
#   TransFun: SOTA/TransFun/data/{bp,mf,cc}_results.txt
#   DPFunc:   SOTA_predictions/DPFunc/{bp,mf,cc}_results.txt
# ==============================================================================

#SBATCH --job-name=predict_sota
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu-h100
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
# Create output dirs up front so later steps don't fail on mkdir
mkdir -p baselines/deepfri_results
mkdir -p SOTA_predictions/DPFunc
mkdir -p SOTA/TransFun/data

# ==============================================================================
# [1/3] DeepFRI
# ==============================================================================
echo "========================================================"
echo "  [1/3] DeepFRI Inference"
echo "========================================================"

# Prep step: extract test sequences from .pt files — requires torch
conda activate dgg_sota_torch
cd ${PROJECT_DIR}/baselines/DeepFRI

cat << 'PYEOF' > prep_deepfri_fasta.py
import os, pickle, sys
sys.path.append('../../')

PKL = '../../preprocessing/data/split_files/datasets.pkl'
OUT = 'test_seqs.fasta'

class _Up(pickle.Unpickler):
    def find_class(self, m, n):
        if m == 'numpy._core.multiarray':
            m = 'numpy.core.multiarray'
        return super().find_class(m, n)

import __main__
from preprocessing.create_batch_dataset import PDB_Dataset
__main__.PDB_Dataset = PDB_Dataset

with open(PKL, 'rb') as f:
    ds = _Up(f).load()

# All three ontologies share the same test proteins — use bp
test_prots = ds['biological_process']['test'].pdb_split_list

import torch
recs = []
for p in test_prots:
    pt = f'../../preprocessing/data/processed/{p}.pt'
    if not os.path.exists(pt):
        continue
    try:
        d = torch.load(pt, weights_only=False, map_location='cpu')
        seq = d.sequence if hasattr(d, 'sequence') else d['atoms'].sequence
        recs.append(f'>{p}\n{seq}')
    except Exception:
        pass

with open(OUT, 'w') as f:
    f.write('\n'.join(recs))
print(f'Generated {OUT} with {len(recs)} sequences.')
PYEOF

python prep_deepfri_fasta.py

# Inference step: requires tensorflow
conda activate dgg_sota_tf
cd ${PROJECT_DIR}/baselines/DeepFRI

for ont in bp mf cc; do
    echo "  Running DeepFRI for ${ont^^}..."
    python predict.py \
        --fasta_fn test_seqs.fasta \
        -ont ${ont} \
        -o ../deepfri_results/deepfri_seq_${ont^^}
done


# ==============================================================================
# [2/3] TransFun
# ==============================================================================
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
    echo "  Running TransFun for ${full_ont}..."
    # Results must go to data/ — that is where evaluate_sota_5seeds.py looks
    python predict.py \
        --data-path data \
        --ontology ${full_ont} \
        --input-type pdb \
        --pdb-path ${PROJECT_DIR}/SOTA/TransFun/test_pdbs/ \
        --output data/${ont}_results.txt
done


# ==============================================================================
# [3/3] DPFunc
# ==============================================================================
echo "========================================================"
echo "  [3/3] DPFunc Inference"
echo "========================================================"
conda activate dgg_sota_torch
cd ${PROJECT_DIR}/SOTA/DPFunc

for ont in bp mf cc; do
    echo "  Building DPFunc workspace for ${ont}..."
    python DPFunc_demo_pipeline/scripts/build_data_demo.py \
        --manifest manifest.tsv \
        --workspace workspace_${ont} \
        --ontology ${ont} \
        --pdb-dir ${PROJECT_DIR}/SOTA/TransFun/test_pdbs \
        --inter-idx data/inter_idx.pkl \
        --go-obo data/go.obo \
        --esm-batch-size 1

    echo "  Running DPFunc prediction for ${ont}..."
    # Symlink model files into workspace so DPFunc can find them
    rm -rf workspace_${ont}/save_models workspace_${ont}/mlb
    ln -sf $(pwd)/save_models workspace_${ont}/save_models
    ln -sf $(pwd)/mlb workspace_${ont}/mlb

    python DPFunc_pred.py \
        -d ${ont} \
        -n -1 \
        -p DPFunc_model \
        -w workspace_${ont}

    echo "  Converting DPFunc output pkl -> text for ${ont}..."
    PKL_PATH="${PROJECT_DIR}/SOTA/DPFunc/workspace_${ont}/results/DPFunc_model_${ont}_final.pkl"
    TXT_PATH="${PROJECT_DIR}/SOTA_predictions/DPFunc/${ont}_results.txt"

    python - <<PYEOF
import sys, pandas as pd, os

pkl = "${PKL_PATH}"
out = "${TXT_PATH}"

if not os.path.exists(pkl):
    print(f"[ERROR] DPFunc result not found: {pkl}", file=sys.stderr)
    sys.exit(1)

df = pd.read_pickle(pkl)
written = 0
with open(out, 'w') as f:
    for prot in df.index:
        for go_term in df.columns:
            score = float(df.loc[prot, go_term])
            if score > 0:
                f.write(f"{prot} {go_term} {score}\n")
                written += 1
print(f"Wrote {written} predictions to {out}")
PYEOF

done

echo "========================================================"
echo "  All SOTA Predictions Complete!"
echo "  End time  : $(date)"
echo "========================================================"
exit 0
