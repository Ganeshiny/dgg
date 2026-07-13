#!/usr/bin/env bash
# run_jk_test.sh
# Runs the HybridGNN_JK model for a given ontology using its best hyperparameters.

if [ -z "$1" ]; then
    echo "Usage: bash run_jk_test.sh [ontology]"
    echo "Example: bash run_jk_test.sh biological_process"
    exit 1
fi

ONTOLOGY="$1"
echo "Fetching best hyperparameters for $ONTOLOGY..."

BEST_PARAMS=$(python3 get_best_hyperparams.py --ontology "$ONTOLOGY")

if [ -z "$BEST_PARAMS" ]; then
    echo "Error: Could not find best parameters for $ONTOLOGY in tuning summary."
    exit 1
fi

# Export LR, DROPOUT, BATCH_SIZE
eval "$BEST_PARAMS"
echo "Found Best Params -> LR: $LR, Dropout: $DROPOUT, Batch Size: $BATCH_SIZE"

OUT_DIR="runs_jk_test/${ONTOLOGY}"
mkdir -p "$OUT_DIR"

echo "Running HybridGNN_JK training test..."
python3 train.py \
    --model "Hybrid_JK" \
    --loss "Focal" \
    --seed 42 \
    --ontology "$ONTOLOGY" \
    --epochs 1000 \
    --lr "$LR" \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUT_DIR"
