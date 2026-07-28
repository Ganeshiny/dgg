#!/usr/bin/env bash
# Generate DPFunc-style stratification, plots, and supplementary tables.
# Uses completed normalized predictions; no GPU or inference is required.
#SBATCH --job-name=dgg_supp
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/supplementary_%j.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/supplementary_%j.err

set -euo pipefail
PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
WORKSPACE="${DGG_BENCHMARK_ROOT:-${PROJECT_DIR}/arc_benchmark/nominal_30_identity_80_coverage}"
ENV_NAME="${DGG_BASE_ENV:-deepgreengo}"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -m src.benchmark.run_stratified --workspace "${WORKSPACE}"
python src/plot_baselines_only.py --workspace "${WORKSPACE}"
python src/plot_stratified.py --workspace "${WORKSPACE}"
python src/export_supplementary_tables.py --workspace "${WORKSPACE}"

echo "Supplementary analysis complete"
echo "Plots: ${WORKSPACE}/plots/stratified"
echo "Tables: ${WORKSPACE}/plots/supplementary_tables"
