#!/usr/bin/env bash
# run_tuning.sh
# Performs a grid search over key hyperparameters for DeepGreenGO.
# Usage: bash run_tuning.sh [ontology]
# If ontology is not provided, runs for all 3 ontologies sequentially.

LRS=(1e-5 5e-5 1e-4)
DROPOUTS=(0.2 0.3 0.4)
BATCH_SIZES=(16 32)
MODEL="Hybrid"
LOSS="Focal"
SEED=42

if [ -n "$1" ]; then
    ONTOLOGIES=("$1")
    echo "Running tuning only for: $1"
else
    ONTOLOGIES=("biological_process" "molecular_function" "cellular_component")
fi

DATASET_PATH="${DATASET_PATH:-preprocessing/data/split_files/datasets.pkl}"
EPOCHS="${EPOCHS:-100}"
OUT_DIR="tuning_runs"

echo "========================================="
echo "  DeepGreenGO Hyperparameter Tuning"
echo "========================================="
echo "Dataset: $DATASET_PATH"
echo "Output Directory: $OUT_DIR"
echo ""

TOTAL=$(( ${#ONTOLOGIES[@]} * ${#LRS[@]} * ${#DROPOUTS[@]} * ${#BATCH_SIZES[@]} ))
RUN=0
FAILED=0

mkdir -p "$OUT_DIR"

for ont in "${ONTOLOGIES[@]}"; do
    for lr in "${LRS[@]}"; do
        for drop in "${DROPOUTS[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
                RUN=$(( RUN + 1 ))
                echo "------------------------------------------------------"
                echo "  Tuning Run $RUN / $TOTAL"
                echo "  Ontology: $ont | LR: $lr | Dropout: $drop | Batch: $bs"
                echo "------------------------------------------------------"

                python3 train.py \
                    --model "$MODEL" \
                    --loss  "$LOSS" \
                    --seed  "$SEED" \
                    --ontology "$ont" \
                    --epochs "$EPOCHS" \
                    --lr "$lr" \
                    --dropout "$drop" \
                    --batch_size "$bs" \
                    --dataset_path "$DATASET_PATH" \
                    --output_dir "$OUT_DIR"

                if [ $? -ne 0 ]; then
                    echo "  [WARN] Tuning run failed — continuing sweep"
                    FAILED=$(( FAILED + 1 ))
                fi
            done
        done
    done
done

echo ""
echo "========================================="
echo "  Tuning sweep complete."
echo "  Completed: $(( TOTAL - FAILED ))  /  $TOTAL"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failed:    $FAILED"
fi
echo "========================================="
echo "Run 'python3 aggregate_tuning.py' to summarize results."
