#!/usr/bin/env bash
# Render DeepGreenGO versus the locked baseline set into plots/figures/benchmark.
# Submit from anywhere inside the repository:
#   sbatch "arc slurms/run_baseline_figures.slurm"

#SBATCH --job-name=dgg_baseline_figures
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=baseline_figures_%j.out
#SBATCH --error=baseline_figures_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -n "${DGG_PROJECT_ROOT:-${DGG_PROJECT_DIR:-}}" ]]; then
    PROJECT_DIR="${DGG_PROJECT_ROOT:-${DGG_PROJECT_DIR}}"
elif PROJECT_DIR="$(git -C "${SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    :
elif [[ -d "/home/${USER:-ganeshiny.sridharan}/dgg/deep-green-GO" ]]; then
    PROJECT_DIR="/home/${USER:-ganeshiny.sridharan}/dgg/deep-green-GO"
elif [[ -d "/home/ganeshiny.sridharan/dgg/deep-green-GO" ]]; then
    PROJECT_DIR="/home/ganeshiny.sridharan/dgg/deep-green-GO"
else
    echo "ERROR: submit from inside the DeepGreenGO checkout or set DGG_PROJECT_DIR." >&2
    exit 2
fi

WORKSPACE="${DGG_BENCHMARK_ROOT:-${PROJECT_DIR}/arc_benchmark/nominal_30_identity_80_coverage}"
OUTPUT_DIR="${DGG_FIGURE_DIR:-${PROJECT_DIR}/plots/figures/benchmark}"
ENV_NAME="${DGG_FIGURE_ENV:-deepgreengo}"

cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

METRICS="${WORKSPACE}/results/benchmark_metrics.csv"
BOOTSTRAP="${WORKSPACE}/results/bootstrap_metrics.csv"
for required in "${METRICS}" "${BOOTSTRAP}"; do
    if [[ ! -s "${required}" ]]; then
        echo "ERROR: required benchmark result is missing or empty: ${required}" >&2
        exit 2
    fi
done

echo "Project:   ${PROJECT_DIR}"
echo "Workspace: ${WORKSPACE}"
echo "Output:    ${OUTPUT_DIR}"
echo "Git HEAD:  $(git rev-parse --short HEAD)"

python src/plot_benchmark.py \
    --workspace "${WORKSPACE}" \
    --output-dir "${OUTPUT_DIR}"

echo "Locked DeepGreenGO comparison figures complete: ${OUTPUT_DIR}"
