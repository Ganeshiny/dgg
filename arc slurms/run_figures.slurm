#!/bin/bash
# Build every manuscript figure on an ARC compute node.
#
# Figure rendering is CPU-only and reads JSON/CSV/PKL artifacts — it never
# loads a checkpoint — so this deliberately requests no GPU. Per the RCS
# guidelines the ARC login node is limited to light work (editing, submitting,
# short tests under ~15 minutes); the full figure set exceeds that, so it is
# submitted rather than run interactively.
#
#   sbatch 'arc slurms/run_figures.slurm'
#
# cpu2023: 7-day limit, 512 GB, 64 cores. One hour and 16 GB is ample here —
# request what is used so the job is scheduled quickly and does not idle a
# large allocation.

#SBATCH --job-name=dgg_figures
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/figures_%j.out
#SBATCH --error=logs/figures_%j.err

set -euo pipefail

PROJECT="${DGG_PROJECT_DIR:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
cd "$PROJECT"
mkdir -p logs

# Conda under a non-interactive Slurm shell: source the profile hook rather
# than relying on `conda activate` being defined.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate deepgreengo

# Headless node: no display, so force a non-interactive backend. Without this
# matplotlib can try an interactive backend and fail on a compute node.
export MPLBACKEND=Agg
# Keep BLAS from oversubscribing the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Target journal: bmc (85/170 mm) or nature (89/183 mm).
export DGG_JOURNAL="${DGG_JOURNAL:-bmc}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} journal=$DGG_JOURNAL"
echo "python=$(which python)"
python -c "import matplotlib; print('matplotlib', matplotlib.__version__)"

echo "=== [1/3] ablation + bin figures (main_text/ and supplementary/) ==="
python src/make_figures.py --output-dir plots/figures

echo "=== [2/3] hyperparameter tuning supplement ==="
python src/plot_arc_tuning.py \
    --tuning-root arc_tuning_cafa \
    --output-dir plots/figures/supplementary_tuning

echo "=== [3/3] baseline benchmark figures ==="
python src/plot_benchmark.py \
    --workspace arc_benchmark/nominal_30_identity_80_coverage \
    --output-dir plots/figures/benchmark

echo "=== done ==="
find plots/figures -name '*.pdf' | wc -l | xargs echo "vector figures written:"
du -sh plots/figures
