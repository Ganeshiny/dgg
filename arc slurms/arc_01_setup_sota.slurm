#!/usr/bin/env bash
# ==============================================================================
# arc_01_setup_sota.slurm
# ARC Cluster - strict, resumable SOTA baseline setup
#
# Submit from the repository root:
#   sbatch "arc slurms/arc_01_setup_sota.slurm"
# Override the project location with DGG_PROJECT_ROOT when needed.
# ==============================================================================

#SBATCH --job-name=setup_sota
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --output=logs/setup_sota_%j.out
#SBATCH --error=logs/setup_sota_%j.err

set -euo pipefail

PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
SETUP_SCRIPT="${PROJECT_DIR}/scripts/setup_arc_sota.sh"

cd "${PROJECT_DIR}"
mkdir -p logs

echo "========================================================"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : ${SLURM_NODELIST}"
echo "  Start time  : $(date)"
echo "  Project     : ${PROJECT_DIR}"
echo "  Methods     : ${DGG_SOTA_SETUP_METHODS:-deepfri,dpfunc,deepgoplus,deepgose,interproscan}"
echo "========================================================"

if [[ ! -x "${SETUP_SCRIPT}" ]]; then
    echo "[ERROR] Missing executable setup script: ${SETUP_SCRIPT}" >&2
    exit 1
fi

"${SETUP_SCRIPT}"
