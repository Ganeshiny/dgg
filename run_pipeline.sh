#!/bin/bash
set -e

echo "=== 1. Extract Sequences, Cluster (MMseqs2) and Split ==="
python3 preprocessing/extract_seqs_from_cif.py

echo "=== 2. Generate Distance Maps (with pLDDT) ==="
python3 preprocessing/create_cmaps.py

echo "=== 3. Generate PyG Batch Datasets ==="
python3 preprocessing/create_batch_dataset.py

echo "=== Preprocessing Completed Successfully ==="

echo "=== 3.5 Run Hyperparameter Tuning ==="
bash run_tuning.sh
python3 aggregate_tuning.py

echo "=== 4. Run Baseline (BLAST) ==="
cd baselines/blast
python3 run_blast_baseline.py
cd ../../

echo "=== 5. Run Ablations & Plot Results ==="
python3 run_experiments.py

echo "=== Pipeline Completed! ==="
