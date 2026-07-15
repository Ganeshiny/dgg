#!/usr/bin/env bash
# run_ablation_input.sh
# Runs Input Modality ablation: Sequence-only, Structure-only, and Seq + Struct

MODALITIES=("seq_only" "struct_only" "full")
SEEDS=(42 123 456)
if [ -n "$1" ]; then
    ONTOLOGIES=("$1")
    echo "Running only for: $1"
else
    ONTOLOGIES=("biological_process" "molecular_function" "cellular_component")
fi

DATASET_PATH="${DATASET_PATH:-preprocessing/data/split_files/datasets.pkl}"
EPOCHS="${EPOCHS:-1000}"
OUT_DIR="runs_ablation_input"
MODEL="Hybrid_JK"
LOSS="Focal"

echo "========================================="
echo "  Input Modality Ablation (Hybrid_JK)"
echo "========================================="
echo "Output Directory: $OUT_DIR"

TOTAL=$(( ${#ONTOLOGIES[@]} * ${#MODALITIES[@]} * ${#SEEDS[@]} ))
RUN=0
FAILED=0

mkdir -p "$OUT_DIR"

for ont in "${ONTOLOGIES[@]}"; do
    BEST_PARAMS=$(python3 get_best_hyperparams.py --ontology "$ont" --summary_file "tuning_runs_jk/tuning_results_summary.csv")
    if [ -n "$BEST_PARAMS" ]; then
        eval "$BEST_PARAMS"
        echo "  [✓] Found Tuned Params: LR=$LR | Dropout=$DROPOUT | BatchSize=$BATCH_SIZE"
    else
        echo "  [!] No tuning results found. Falling back to defaults."
        LR=1e-5; DROPOUT=0.3; BATCH_SIZE=16
    fi
    echo "========================================="

    for mod in "${MODALITIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN=$(( RUN + 1 ))
            echo "------------------------------------------------------"
            echo "  Run $RUN / $TOTAL | $ont | modality=$mod | s$seed"
            echo "------------------------------------------------------"

            python3 train.py \
                --model "$MODEL" \
                --loss  "$LOSS" \
                --input_modality "$mod" \
                --seed  "$seed" \
                --ontology "$ont" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --dropout "$DROPOUT" \
                --dataset_path "$DATASET_PATH" \
                --output_dir "$OUT_DIR"

            if [ $? -ne 0 ]; then
                FAILED=$(( FAILED + 1 ))
            fi
        done
    done
done
echo "Completed: $(( TOTAL - FAILED )) / $TOTAL"
