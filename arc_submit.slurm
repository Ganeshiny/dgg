#!/bin/bash
# ==============================================================================
# ARC Cluster Submission Script for DeepGreenGO
# University of Calgary ARC — gpu-v100 partition
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
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/deep-green-GO/logs/deepgreengo_%j.out
#SBATCH --error=/home/%u/deep-green-GO/logs/deepgreengo_%j.err

####### Diagnostics ###########################################################
# Make sure the logs directory exists before anything else runs
mkdir -p /home/$USER/deep-green-GO/logs

echo "========================================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURM_NODELIST"
echo "CPUs          : $SLURM_CPUS_PER_TASK"
echo "Start time    : $(date)"
echo "Working dir   : $(pwd)"
echo "Log .out      : /home/$USER/deep-green-GO/logs/deepgreengo_${SLURM_JOB_ID}.out"
echo "Log .err      : /home/$USER/deep-green-GO/logs/deepgreengo_${SLURM_JOB_ID}.err"
echo "========================================================"

####### Load Software #########################################################
# Try newest Anaconda first; fall back to older versions available on ARC.
# If none load, abort immediately (do NOT silently continue — the job will
# fail anyway once 'conda' is not found).
module load python/anaconda3-2022.10 2>/dev/null \
  || module load python/anaconda3-2020.02 2>/dev/null \
  || module load python/anaconda3-2018.12 2>/dev/null \
  || { echo "ERROR: No Anaconda module found. Aborting."; exit 1; }

echo "Loaded module: $(module list 2>&1 | grep anaconda)"

# Initialize conda so 'conda activate' works inside batch scripts
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepgreengo \
  || { echo "ERROR: 'deepgreengo' conda env not found. Aborting."; exit 1; }

echo "Python : $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT AVAILABLE\")')"

####### Job Execution #########################################################
# ARC allocates a per-job scratch directory at /scratch/$SLURM_JOB_ID.
# Use it for large intermediate files (cmaps) to avoid filling /home (500GB limit).
SCRATCH_DIR="/scratch/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"

# Point the cmap output to scratch, and export the thread count.
# With 8 CPUs and 128 GB RAM, 8 parallel workers is safe.
export CMAP_THREADS=$SLURM_CPUS_PER_TASK
export CMAP_CHUNK=50

echo "Scratch dir   : $SCRATCH_DIR"
echo "CMAP_THREADS  : $CMAP_THREADS"
echo "========================================================"

# Run the full DeepGreenGO pipeline
bash run_all.sh

echo "========================================================"
echo "Job finished  : $(date)"
echo "========================================================"
