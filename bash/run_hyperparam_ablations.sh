#!/usr/bin/env bash
# run_hyperparam_ablations.sh
# Runs sensitivity analysis for hyperparameters on the Hybrid architecture.

DATASET_PATH="${DATASET_PATH:-preprocessing/data/split_files/datasets.pkl}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ONTOLOGY="biological_process"
MODEL="Hybrid"
LOSS="Focal"
SEED=42

echo "========================================="
echo "  DeepGreenGO Hyperparameter Sweeps"
echo "========================================="
echo "Base configuration: Model=$MODEL, Loss=$LOSS, Ontology=$ONTOLOGY, Epochs=$EPOCHS, Seed=$SEED"

# Helper function
run_exp() {
    local param_name=$1
    local param_val=$2
    local extra_args=$3

    echo "------------------------------------------------------"
    echo "  Testing $param_name = $param_val"
    echo "------------------------------------------------------"

    python3 train.py \
        --model "$MODEL" \
        --loss "$LOSS" \
        --seed "$SEED" \
        --ontology "$ONTOLOGY" \
        --epochs "$EPOCHS" \
        --dataset_path "$DATASET_PATH" \
        $extra_args

    if [ $? -ne 0 ]; then
        echo "  [WARN] Run failed for $param_name=$param_val"
    fi
}

# 1. Learning Rate Sweep
for lr in 1e-4 5e-5 1e-5; do
    run_exp "Learning Rate" "$lr" "--lr $lr --batch_size $BATCH_SIZE"
done

# 2. Dropout Sweep
for dropout in 0.1 0.3 0.5; do
    run_exp "Dropout" "$dropout" "--dropout $dropout --batch_size $BATCH_SIZE"
done

# 3. Focal Gamma Sweep
for gamma in 1.0 2.0 4.0; do
    run_exp "Focal Gamma" "$gamma" "--focal_gamma $gamma --batch_size $BATCH_SIZE"
done

# 4. Hidden Sizes Sweep
for hs in "512,256" "1024,512" "1024,912"; do
    run_exp "Hidden Sizes" "$hs" "--hidden_sizes $hs --batch_size $BATCH_SIZE"
done

# 5. Batch Size Sweep
for bs in 16 32 64; do
    run_exp "Batch Size" "$bs" "--batch_size $bs"
done

echo ""
echo "========================================="
echo "  Hyperparameter sweep complete."
echo "========================================="
