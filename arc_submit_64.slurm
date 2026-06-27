#!/bin/bash
# ==============================================================================
# ARC Cluster Submission Script for DeepGreenGO
# University of Calgary ARC — gpu-l40 partition
#
# Submit with:  sbatch arc_submit.slurm
# Monitor with: squeue -u $USER
# ==============================================================================

####### Reserve computing resources ###########################################
#SBATCH --job-name=deepgreengo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --mem=128000
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/dgg/deep-green-GO/logs/deepgreengo_%j.out
#SBATCH --error=/home/%u/dgg/deep-green-GO/logs/deepgreengo_%j.err

####### Diagnostics ###########################################################
# PROJECT_DIR is the directory where you ran 'sbatch' from.
# Slurm automatically provides this as SLURM_SUBMIT_DIR.
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"


echo "========================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURM_NODELIST"
echo "CPUs          : $SLURM_CPUS_PER_TASK"
echo "Start time    : $(date)"
echo "Project dir   : ${PROJECT_DIR}"
echo "Log .out      : ${LOG_DIR}/deepgreengo_${SLURM_JOB_ID}.out"
echo "Log .err      : ${LOG_DIR}/deepgreengo_${SLURM_JOB_ID}.err"
echo "========================================================"

####### Load Software #########################################################
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepgreengo \
  || { echo "ERROR: 'deepgreengo' conda env not found. Aborting."; \
       echo "       Run 'conda env list' on ARC to check available environments."; \
       exit 1; }

echo "Python : $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE")')"

####### Job Execution #########################################################
SCRATCH_DIR="/scratch/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"

export CMAP_THREADS=$SLURM_CPUS_PER_TASK
export CMAP_CHUNK=50

echo "Scratch dir   : $SCRATCH_DIR"
echo "CMAP_THREADS  : $CMAP_THREADS"
echo "========================================================"

# Run the full DeepGreenGO pipeline from the project directory.
# --skip-cmaps is added because we already have ~16,000 cmaps and want to force
# the pipeline to move to the dataset building step.
cd "${PROJECT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: SLURM ARRAY JOBS FOR PARALLEL ONTOLOGY TRAINING
# Running all 3 ontologies sequentially takes >23 hours and will timeout.
# Instead, submit this script as a SLURM array: `sbatch --array=0-2 arc_submit.slurm`
# ─────────────────────────────────────────────────────────────────────────────
ONT_LIST=("biological_process" "molecular_function" "cellular_component")

# If submitted as an array, pick one ontology based on the array task ID.
# If run normally, it will just run all 3 sequentially (which may timeout).
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    CURRENT_ONT="${ONT_LIST[$SLURM_ARRAY_TASK_ID]}"
    echo "Array Job Detected: Running ONLY for $CURRENT_ONT"
    
    # Run hyperparameter tuning first for this ontology
    bash run_tuning_64.sh "$CURRENT_ONT"
    python3 aggregate_tuning.py

    # Run ablations for just this ontology
    bash run_ablations.sh "$CURRENT_ONT"
else
    echo "Running sequentially for all ontologies (Warning: May timeout)"
    
    # Run tuning before ablations sequentially
    bash run_all.sh --skip-preprocess --tune
fi

echo "========================================================"
echo "Job finished  : $(date)"
echo "========================================================"
