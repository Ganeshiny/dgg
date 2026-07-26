#!/bin/bash
# Redo the entire figure pipeline on ARC and verify it reproduces.
#
#   sbatch 'arc slurms/run_figure_verification.slurm'
#
# Pull the current code onto ARC first — this job runs whatever is checked out.
#
# What it does, in order:
#   0. preflight: report every input it needs and whether it is present
#   1. repair the benchmark BLAST predictions, then re-evaluate
#      (ARC still holds the all-zero BLAST artifact; the repair is idempotent,
#       so re-running it on an already-correct workspace is harmless)
#   2. render every figure set
#   3. write verification_report_arc.json and diff-ready summary
#
# CPU only, deliberately: figure rendering and the label transfer read
# JSON/CSV/PKL and never load a checkpoint, so a GPU would sit idle. Per RCS
# guidance the ARC login node is limited to short light work, so this is
# submitted rather than run interactively.

#SBATCH --job-name=dgg_fig_verify
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/figure_verify_%j.out
#SBATCH --error=logs/figure_verify_%j.err

set -euo pipefail

PROJECT="${DGG_PROJECT_DIR:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
cd "$PROJECT"
mkdir -p logs

# set -u must be lifted across conda: the deepgreengo env's
# activate.d/libblas_mkl_activate.sh dereferences an unset MKL_INTERFACE_LAYER
# and aborts the job under a strict shell.
set +u
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found under \$HOME/miniconda3 or \$HOME/anaconda3" >&2
    exit 1
fi
conda activate "${DGG_FIGURE_ENV:-deepgreengo}"
set -u

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export DGG_JOURNAL="${DGG_JOURNAL:-bmc}"

WORKSPACE="arc_benchmark/nominal_30_identity_80_coverage"

echo "======================================================================"
echo " host      : $(hostname)"
echo " job       : ${SLURM_JOB_ID:-none}"
echo " project   : $PROJECT"
echo " journal   : $DGG_JOURNAL"
echo " python    : $(which python)"
echo " git HEAD  : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "======================================================================"

echo
echo "=== [0/4] preflight: are the inputs present on this host? ==="
missing=0
check_path() {
    if [[ -e "$1" ]]; then
        echo "  present : $1"
    else
        echo "  MISSING : $1   ($2)"
        missing=$((missing + 1))
    fi
}
check_path "preprocessing/data_arc_rebuild_2026_07_14/datasets/threshold_30" "bin/ablation support audit"
check_path "preprocessing/data_arc_rebuild_2026_07_14/pdb_splits/threshold_30/blast_te_vs_tr.tsv" "leakage figure"
check_path "preprocessing/data_arc_rebuild_2026_07_14/go-basic.obo" "benchmark evaluation"
check_path "$WORKSPACE/results/benchmark_metrics.csv" "benchmark figures"
check_path "$WORKSPACE/raw/blast_hits.tsv" "BLAST repair"
check_path "arc_tuning_cafa/hybrid_search" "tuning figures"
check_path "logs" "ablation log fallback"
if [[ -f "$WORKSPACE/raw/blast_hits.tsv" ]]; then
    echo "  blast_hits.tsv lines: $(wc -l < "$WORKSPACE/raw/blast_hits.tsv")  (expected ~7108; 0 means the search must be re-run)"
fi
echo "  ablation result folders on disk: $(find arc_tuning_cafa/ablations -name test_metrics.json 2>/dev/null | wc -l)"
echo "  ablation SLURM logs found      : $(find logs -name 'arc_ablation_*.out' 2>/dev/null | wc -l)"
if (( missing > 0 )); then
    echo "  -> $missing input(s) missing; dependent stages will skip or fail."
fi

echo
echo "=== [1/4] repair benchmark BLAST predictions, then re-evaluate ==="
# Idempotent. run_search_once reuses a complete blast_hits.tsv and only
# re-derives the label transfer, so this corrects an all-zero artifact
# without repeating the search.
python -m src.benchmark.cli sequence-baselines \
    --workspace "$WORKSPACE" --methods blast
python -m src.benchmark.cli evaluate \
    --workspace "$WORKSPACE" --bootstraps 1000 --bootstrap-seed 20260720

echo
echo "=== [2/4] render every figure set ==="
python src/make_figures.py --output-dir plots/figures
python src/plot_arc_tuning.py --tuning-root arc_tuning_cafa \
    --output-dir plots/figures/supplementary_tuning
python src/plot_benchmark.py --workspace "$WORKSPACE" \
    --output-dir plots/figures/benchmark
python src/plot_reviewer_figures.py --output-dir plots/figures/reviewer

echo
echo "=== [3/4] verification ==="
# Do not let a failing check abort before the report is written.
set +e
python src/verify_figure_pipeline.py --output verification_report_arc.json
verify_status=$?
set -e

echo
echo "=== [4/4] summary ==="
echo "vector figures: $(find plots/figures -name '*.pdf' | wc -l)"
du -sh plots/figures
echo
if [[ $verify_status -eq 0 ]]; then
    echo "VERIFICATION PASSED — every runnable check matches the expected value."
else
    echo "VERIFICATION FAILED — see the MISMATCH lines above and the JSON report."
fi
echo
echo "Copy verification_report_arc.json back and diff it against the local"
echo "verification_report.json. Any differing value is a real reproducibility"
echo "gap; identical values mean the pipeline reproduces on ARC."
exit $verify_status
