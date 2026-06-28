#!/usr/bin/env bash
# run_ablations.sh
# Runs all Model × Loss combinations across 3 seeds for all three GO ontologies.
# Failures are logged but do NOT halt the whole sweep.

MODELS=("MLP" "GCN" "GAT" "Hybrid")
LOSSES=("BCE" "Focal")
SEEDS=(42 123 456)

# Default to all 3, but allow passing a specific ontology as an argument
if [ -n "$1" ]; then
    ONTOLOGIES=("$1")
    echo "Running ablations only for: $1"
else
    ONTOLOGIES=("biological_process" "molecular_function" "cellular_component")
fi

DATASET_PATH="${DATASET_PATH:-preprocessing/data/split_files/datasets.pkl}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"


echo "========================================="
echo "  DeepGreenGO  Ablation Sweep"
echo "========================================="
echo "Dataset: $DATASET_PATH"
echo "Epochs:  $EPOCHS  | Batch: $BATCH_SIZE"
echo ""

TOTAL=$(( ${#MODELS[@]} * ${#LOSSES[@]} * ${#SEEDS[@]} * ${#ONTOLOGIES[@]} ))
RUN=0
FAILED=0

for ont in "${ONTOLOGIES[@]}"; do
    # Default to base script variables
    BEST_LR=""
    BEST_DROPOUT=""
    BEST_BATCH_SIZE="$BATCH_SIZE"
    
    # Try to load best parameters if tuning was run
    if [ -f "tuning_runs/tuning_results_summary.csv" ]; then
        BEST_PARAMS=$(python3 get_best_hyperparams.py --ontology "$ont")
    # Dynamically fetch the best hyperparameters for this specific ontology
    BEST_PARAMS=$(python3 get_best_hyperparams.py --ontology "$ont")
    if [ -n "$BEST_PARAMS" ]; then
        eval "$BEST_PARAMS"
        echo "  [✓] Found Tuned Params: LR=$LR | Dropout=$DROPOUT | BatchSize=$BATCH_SIZE"
    else
        echo "  [!] No tuning results found. Falling back to defaults."
        LR=1e-5
        DROPOUT=0.3
        BATCH_SIZE=16
    fi
    echo "========================================="

    for model in "${MODELS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                RUN=$(( RUN + 1 ))
                RUN_NAME="${ont}_${model}_${loss}_s${seed}"
                
                echo "------------------------------------------------------"
                echo "  Run $RUN / $TOTAL"
                echo "  Run: $RUN_NAME"
                echo "------------------------------------------------------"

                python3 train.py \
                    --model "$model" \
                    --loss  "$loss" \
                    --seed  "$seed" \
                    --ontology "$ont" \
                    --epochs "$EPOCHS" \
                    --batch_size "$BATCH_SIZE" \
                    --lr "$LR" \
                    --dropout "$DROPOUT" \
                    --dataset_path "$DATASET_PATH"

                if [ $? -ne 0 ]; then
                    echo "  [WARN] Run failed — continuing sweep"
                    FAILED=$(( FAILED + 1 ))
                fi
            done
        done
    done
done

echo ""
echo "========================================="
echo "  Ablation sweep complete."
echo "  Completed: $(( TOTAL - FAILED ))  /  $TOTAL"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failed:    $FAILED"
fi
echo "========================================="
