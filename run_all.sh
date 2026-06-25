#!/usr/bin/env bash
# =============================================================================
#  DeepGreenGO — Master Pipeline Script
#  Usage:  bash run_all.sh [--skip-preprocess] [--skip-ablations]
#
#  All steps are run from the PROJECT ROOT (same directory as this script).
# =============================================================================

set -euo pipefail

# ─── Helpers ──────────────────────────────────────────────────────────────────
BOLD=$'\033[1m'
RESET=$'\033[0m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'

section() { echo; echo "${BOLD}${GREEN}══════════════════════════════════════════${RESET}"; echo "${BOLD}${GREEN}  $1${RESET}"; echo "${BOLD}${GREEN}══════════════════════════════════════════${RESET}"; }
info()    { echo "  ${YELLOW}▸${RESET} $*"; }
success() { echo "  ${GREEN}✔${RESET} $*"; }
warn()    { echo "  ${YELLOW}⚠${RESET}  $*"; }
die()     { echo "  ${RED}✘ $*${RESET}"; exit 1; }

# Ensure we are in the project root (directory that contains this file)
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
info "Project root: $PROJECT_ROOT"

# ─── Parse flags ──────────────────────────────────────────────────────────────
SKIP_PREPROCESS=false
SKIP_BASELINES=false
SKIP_ABLATIONS=false
SKIP_EVAL=false
SKIP_PLOTS=false
SKIP_CMAPS=false

for arg in "$@"; do
    case "$arg" in
        --skip-preprocess) SKIP_PREPROCESS=true ;;
        --skip-baselines)  SKIP_BASELINES=true  ;;
        --skip-ablations)  SKIP_ABLATIONS=true  ;;
        --skip-eval)       SKIP_EVAL=true       ;;
        --skip-plots)      SKIP_PLOTS=true      ;;
        --skip-cmaps)      SKIP_CMAPS=true      ;;
        --help|-h)
            echo "Usage: bash run_all.sh [--skip-preprocess] [--skip-cmaps] [--skip-baselines] [--skip-ablations] [--skip-eval] [--skip-plots]"
            exit 0
            ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

# ─── Configuration (override via env vars) ────────────────────────────────────
STRUC_DIR="${STRUC_DIR:-preprocessing/data/structure_files}"
SIFTS_FILE="${SIFTS_FILE:-preprocessing/data/pdb_chain_go.tsv_2024-06-25}"
OBO_FILE="${OBO_FILE:-preprocessing/data/go-basic_2024-06-25.obo}"
SEQS_FILE="${SEQS_FILE:-preprocessing/data/seqs_from_structure_dir.fasta}"
ANNOT_FILE="${ANNOT_FILE:-preprocessing/data/pdb2go.tsv}"
DATASET_PKL="${DATASET_PKL:-preprocessing/data/split_files/datasets.pkl}"

EPOCHS="${EPOCHS:-100}"          # Set to 200+ for final runs
BATCH_SIZE="${BATCH_SIZE:-32}"
MAIN_MODEL="${MAIN_MODEL:-Hybrid}"
MAIN_LOSS="${MAIN_LOSS:-Focal}"
MAIN_SEED="${MAIN_SEED:-42}"
MAIN_ONT="${MAIN_ONT:-biological_process}"
# Contact-map parallelism: 2 workers is safe on most WSL setups.
# Set to 1 if WSL still crashes (sequential mode, zero OOM risk).
CMAP_THREADS="${CMAP_THREADS:-2}"
CMAP_CHUNK="${CMAP_CHUNK:-30}"

info "Settings: EPOCHS=$EPOCHS  BATCH=$BATCH_SIZE  MODEL=$MAIN_MODEL  LOSS=$MAIN_LOSS"
info "Ontology: $MAIN_ONT"

# ─── Step 1: Preprocessing ────────────────────────────────────────────────────
if [ "$SKIP_PREPROCESS" = false ]; then
    section "Step 1: Data Preprocessing"

    info "1a. Extracting sequences from CIF files and building annotation TSV..."
    python3 preprocessing/extract_seqs_from_cif.py \
        -sifts  "$SIFTS_FILE" \
        -struc_dir "$STRUC_DIR" \
        -seqs   "$SEQS_FILE" \
        -obo    "$OBO_FILE" \
        -out    "preprocessing/data/"
    success "Annotations written to $ANNOT_FILE"

    info "1b. Homology clustering and cluster-aware split (MMseqs2)..."
    python3 preprocessing/cluster_and_split.py \
        -fasta  "$SEQS_FILE" \
        -prefix "preprocessing/data/"
    success "Train/valid/test splits written to preprocessing/data/split_files/"

    if [ "$SKIP_CMAPS" = false ]; then
        info "1c. Computing contact maps with pLDDT filtering..."
        info "    Workers: $CMAP_THREADS  |  Chunk: $CMAP_CHUNK  (override with CMAP_THREADS=1 for safe sequential mode)"
        python3 preprocessing/create_cmaps.py \
            -annot       "$ANNOT_FILE" \
            -seqs        "$SEQS_FILE" \
            -struc_dir   "$STRUC_DIR" \
            -num_threads "$CMAP_THREADS" \
            -chunk_size  "$CMAP_CHUNK"
        success "Contact maps written to $STRUC_DIR/tmp_cmap_files/"
    else
        warn "Skipping contact-map generation (--skip-cmaps)"
    fi

    info "1d. Building PyG graph datasets (ProtBERT embeddings)..."
    python3 preprocessing/create_batch_dataset.py
    success "Datasets pickled to $DATASET_PKL"
else
    warn "Skipping preprocessing (--skip-preprocess)"
fi

# ─── Step 2: Baselines ────────────────────────────────────────────────────────
if [ "$SKIP_BASELINES" = false ]; then
    section "Step 2: Baseline Models"

    info "2a. BLAST baseline..."
    python3 baselines/blast/run_blast_baseline.py || warn "BLAST baseline failed (is BLAST+ installed?)"

    info "2b. DIAMOND baseline..."
    python3 baselines/diamond/run_diamond_baseline.py || warn "DIAMOND baseline failed (is DIAMOND installed?)"

    info "2c. Naive frequency baseline..."
    python3 baselines/naive_frequency/run_naive_baseline.py
    success "Naive baseline done"
else
    warn "Skipping baselines (--skip-baselines)"
fi

# ─── Step 3: Ablation sweep ───────────────────────────────────────────────────
if [ "$SKIP_ABLATIONS" = false ]; then
    section "Step 3: Ablation Study (all Model×Loss×Seed×Ontology combos)"
    info "This can take many hours. Override epochs: EPOCHS=50 bash run_all.sh"
    EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" DATASET_PATH="$DATASET_PKL" bash run_ablations.sh
else
    warn "Skipping ablations (--skip-ablations)"
    info "If you only want one quick training run, execute:"
    info "  python3 train.py --model $MAIN_MODEL --loss $MAIN_LOSS --seed $MAIN_SEED --ontology $MAIN_ONT --epochs $EPOCHS"
fi

# ─── Step 4: Per-cluster evaluation ──────────────────────────────────────────
if [ "$SKIP_EVAL" = false ]; then
    section "Step 4: Per-Cluster Evaluation (all 3 ontologies)"

    declare -A ONT_SHORT_MAP
    ONT_SHORT_MAP[biological_process]="bp"
    ONT_SHORT_MAP[molecular_function]="mf"
    ONT_SHORT_MAP[cellular_component]="cc"

    for ONT in biological_process molecular_function cellular_component; do
        ONT_SHORT="${ONT_SHORT_MAP[$ONT]}"
        BEST_MODEL_PATH="runs/${ONT_SHORT}_${MAIN_MODEL}_${MAIN_LOSS}_s${MAIN_SEED}/best_model.pth"

        if [ -f "$BEST_MODEL_PATH" ]; then
            info "Per-cluster eval: $ONT"
            python3 per_cluster_eval.py \
                --model_path "$BEST_MODEL_PATH" \
                --model      "$MAIN_MODEL" \
                --ontology   "$ONT" \
                --dataset_path "$DATASET_PKL" \
                --output     "runs/cluster_performance_${ONT_SHORT}.csv"
            success "Cluster performance saved → runs/cluster_performance_${ONT_SHORT}.csv"
        else
            warn "Best model not found at $BEST_MODEL_PATH — skipping $ONT cluster eval"
        fi
    done
else
    warn "Skipping per-cluster evaluation (--skip-eval)"
fi

# ─── Step 5: Aggregation & Plotting ──────────────────────────────────────────
if [ "$SKIP_PLOTS" = false ]; then
    section "Step 5: Aggregate Results and Generate Plots"

    python3 aggregate_results.py
    python3 plot_results.py
    success "Plots saved to plots/"
else
    warn "Skipping plots (--skip-plots)"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
section "Pipeline Complete"
echo "  Trained models : runs/"
echo "  Figures        : plots/"
echo "  Aggregated CSV : runs/aggregated_results.csv"
echo ""
