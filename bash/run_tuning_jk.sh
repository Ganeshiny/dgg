#!/usr/bin/env bash
# run_tuning_jk.sh
# Performs a grid search over key hyperparameters for HybridGNN_JK.
# Usage: bash run_tuning_jk.sh [ontology]
# If ontology is not provided, runs for all 3 ontologies sequentially.

LRS=(1e-4 5e-4 1e-3)
DROPOUTS=(0.2 0.3 0.4)
BATCH_SIZES=(16 32)
LOSSES=("BCE" "Focal")
GAMMAS=(2.0 4.0)
MODEL="Hybrid_JK"
SEED=42

if [ -n "$1" ]; then
    ONTOLOGIES=("$1")
    echo "Running tuning only for: $1"
else
    ONTOLOGIES=("biological_process" "molecular_function" "cellular_component")
fi

DATASET_PATH="${DATASET_PATH:-preprocessing/data/split_files/datasets.pkl}"
EPOCHS="${EPOCHS:-1000}"
OUT_DIR="tuning_runs_jk"

echo "========================================="
echo "  DeepGreenGO Hyperparameter Tuning (JK)"
echo "========================================="
echo "Dataset: $DATASET_PATH"
echo "Output Directory: $OUT_DIR"
echo ""

# BCE has no gamma, Focal has 2 gammas => 3 loss combinations total per config
TOTAL=$(( ${#ONTOLOGIES[@]} * ${#LRS[@]} * ${#DROPOUTS[@]} * ${#BATCH_SIZES[@]} * 3 ))
RUN=0
FAILED=0

mkdir -p "$OUT_DIR"

for ont in "${ONTOLOGIES[@]}"; do
    for lr in "${LRS[@]}"; do
        for drop in "${DROPOUTS[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
                for loss in "${LOSSES[@]}"; do
                    # If BCE, we only run once (gamma doesn't matter, we just use 4.0 as dummy)
                    if [ "$loss" == "BCE" ]; then
                        CURRENT_GAMMAS=(4.0)
                    else
                        CURRENT_GAMMAS=("${GAMMAS[@]}")
                    fi
                    for gamma in "${CURRENT_GAMMAS[@]}"; do
                        RUN=$(( RUN + 1 ))
                        echo "------------------------------------------------------"
                        echo "  Tuning Run $RUN / $TOTAL"
                        echo "  Ontology: $ont | LR: $lr | Dropout: $drop | Batch: $bs | Loss: $loss | Gamma: $gamma"
                        echo "------------------------------------------------------"

                        python3 src/train.py \
                            --model "$MODEL" \
                            --loss  "$loss" \
                            --focal_gamma "$gamma" \
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
echo "Run 'python3 src/aggregate_tuning.py' to summarize results."
