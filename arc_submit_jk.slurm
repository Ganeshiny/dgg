#!/bin/bash
# ==============================================================================
# ARC Cluster Submission Script for DeepGreenGO (JK MODEL)
# University of Calgary ARC — gpu-l40 partition
#
# Submit with:  sbatch --array=0-2 arc_submit_jk.slurm
# Monitor with: squeue -u $USER
# ==============================================================================

####### Reserve computing resources ###########################################
#SBATCH --job-name=deepgreengo_jk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --mem=128000
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/dgg/deep-green-GO/logs/deepgreengo_jk_%j.out
#SBATCH --error=/home/%u/dgg/deep-green-GO/logs/deepgreengo_jk_%j.err

####### Diagnostics ###########################################################
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURM_NODELIST"
echo "CPUs          : $SLURM_CPUS_PER_TASK"
echo "Start time    : $(date)"
echo "Project dir   : ${PROJECT_DIR}"
echo "Log .out      : ${LOG_DIR}/deepgreengo_jk_${SLURM_JOB_ID}.out"
echo "Log .err      : ${LOG_DIR}/deepgreengo_jk_${SLURM_JOB_ID}.err"
echo "========================================================"

####### Load Software #########################################################
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepgreengo \
  || { echo "ERROR: 'deepgreengo' conda env not found. Aborting."; exit 1; }

####### Job Execution #########################################################
SCRATCH_DIR="/scratch/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"

export CMAP_THREADS=$SLURM_CPUS_PER_TASK
export CMAP_CHUNK=50

cd "${PROJECT_DIR}"

ONT_LIST=("biological_process" "molecular_function" "cellular_component")

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    CURRENT_ONT="${ONT_LIST[$SLURM_ARRAY_TASK_ID]}"
    echo "Array Job Detected: Running ONLY for $CURRENT_ONT"
    
    # Run hyperparameter tuning first for this ontology using the JK model
    bash run_tuning_jk.sh "$CURRENT_ONT"
else
    echo "Running sequentially for all ontologies (Warning: May timeout)"
    # Fallback to run all if not run as array
    bash run_tuning_jk.sh
fi

echo "========================================================"
echo "Job finished  : $(date)"
echo "========================================================"
