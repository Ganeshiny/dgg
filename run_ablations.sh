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
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"


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
    for model in "${MODELS[@]}"; do
        for loss in "${LOSSES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                RUN=$(( RUN + 1 ))
                echo "------------------------------------------------------"
                echo "  Run $RUN / $TOTAL"
                echo "  Ontology: $ont | Model: $model | Loss: $loss | Seed: $seed"
                echo "------------------------------------------------------"

                python3 train.py \
                    --model "$model" \
                    --loss  "$loss" \
                    --seed  "$seed" \
                    --ontology "$ont" \
                    --epochs "$EPOCHS" \
                    --batch_size "$BATCH_SIZE" \
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
