#!/usr/bin/env bash
# Rebuild all manuscript, supplementary, tuning, reviewer, and baseline figures.
#SBATCH --job-name=dgg_all_figures
#SBATCH --partition=cpu2023,cpu2022,cpu2021,cpu2019
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=all_figures_%j.out
#SBATCH --error=all_figures_%j.err

set -euo pipefail

resolve_project() {
  local c r
  for c in "${DGG_PROJECT_ROOT:-}" "${DGG_PROJECT_DIR:-}" "${SLURM_SUBMIT_DIR:-}" "${PWD:-}" "/home/${USER:-ganeshiny.sridharan}/dgg/deep-green-GO" "/home/ganeshiny.sridharan/dgg/deep-green-GO"; do
    [[ -n "$c" && -d "$c" ]] || continue
    r="$(git -C "$c" rev-parse --show-toplevel 2>/dev/null || true)"
    [[ -n "$r" ]] || r="$c"
    [[ -f "$r/src/make_figures.py" && -f "$r/src/plot_benchmark.py" ]] && { printf '%s\n' "$r"; return; }
  done
  echo 'ERROR: DeepGreenGO checkout not found. Set DGG_PROJECT_ROOT to its absolute ARC path.' >&2
  return 2
}
PROJECT="$(resolve_project)"
echo "Resolved project root: $PROJECT"
[[ "${DGG_RESOLVE_ONLY:-0}" == 1 ]] && exit 0

DATA="${DGG_DATA_ROOT:-$PROJECT/preprocessing/data_arc_rebuild_2026_07_14}"
TUNING="${DGG_TUNING_ROOT:-$PROJECT/arc_tuning_cafa}"
WORKSPACE="${DGG_BENCHMARK_ROOT:-$PROJECT/arc_benchmark/nominal_30_identity_80_coverage}"
OUT="${DGG_FIGURE_DIR:-$PROJECT/plots/figures}"
ARCHIVE="${DGG_ABLATION_ARCHIVE:-$PROJECT/plots/arc_tuning_cafa/ablations/ablation_test_metrics.csv}"
BINCSV="${DGG_BIN_CSV:-$PROJECT/plots/arc_tuning_cafa/bin_evaluation/bin_metrics.csv}"
SUPPORT="${DGG_SUPPORT_ROOT:-$DATA/datasets/threshold_30}"
SPLIT="${DGG_SPLIT_ROOT:-$DATA/pdb_splits/threshold_30}"
METRICS="$WORKSPACE/results/benchmark_metrics.csv"
BOOTSTRAP="$WORKSPACE/results/bootstrap_metrics.csv"

cd "$PROJECT"
mkdir -p logs "$OUT/benchmark"
set +u
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo 'ERROR: conda initialization script not found.' >&2; exit 2
fi
conda activate "${DGG_FIGURE_ENV:-deepgreengo}"
set -u
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export DGG_JOURNAL="${DGG_JOURNAL:-bmc}"

required=("$ARCHIVE" "$BINCSV" "$TUNING/hybrid_search" "$PROJECT/plots/arc_tuning_cafa/trial_metrics.csv" "$SUPPORT" "$SPLIT/blast_te_vs_tr.tsv" "$METRICS" "$BOOTSTRAP")
for f in "${required[@]}"; do [[ -e "$f" && ( -d "$f" || -s "$f" ) ]] || { echo "ERROR: missing figure input: $f" >&2; exit 2; }; done

python - "$METRICS" "$BOOTSTRAP" <<'PY'
import sys, pandas as pd
allowed={'naive','blast','blast_max','diamond','diamond_max','foldseek','foldseek_max','deepfri_sequence','deepfri_structure','dpfunc','heal'}
for name in sys.argv[1:]:
    frame=pd.read_csv(name)
    missing=allowed-set(frame['method'].dropna().astype(str))
    if missing: raise SystemExit(f'ERROR: {name} missing requested baselines: {sorted(missing)}')
print('Benchmark preflight passed: all requested baselines are present.')
PY
[[ "${DGG_PREFLIGHT_ONLY:-0}" == 1 ]] && { echo 'Preflight only: passed.'; exit 0; }

python src/make_figures.py --output-dir "$OUT" --ablations-root "$TUNING/ablations" --archive "$ARCHIVE" --bin-csv "$BINCSV" --logs-dir "$PROJECT/logs" --support-root "$SUPPORT"
python src/plot_arc_tuning.py --tuning-root "$TUNING" --output-dir "$OUT/supplementary_tuning"
python src/plot_benchmark.py --workspace "$WORKSPACE" --output-dir "$OUT/benchmark"
python src/plot_reviewer_figures.py --split-dir "$SPLIT" --trials-csv "$PROJECT/plots/arc_tuning_cafa/trial_metrics.csv" --ablation-csv "$ARCHIVE" --output-dir "$OUT/reviewer"

expected=("$OUT/main_text/figure1_ablation_dynamite_micro_fmax.pdf" "$OUT/main_text/figure2_metric_family_micro_fmax_vs_micro_auroc.pdf" "$OUT/main_text/figure3_homology_micro_fmax.pdf" "$OUT/supplementary/figureS19_test_ic_macro_auroc.pdf" "$OUT/supplementary_tables/supp_table_ablation_seed_metrics.csv" "$OUT/supplementary_tuning/five_seed_validation_curves.pdf" "$OUT/benchmark/benchmark_metrics_plotted.csv" "$OUT/benchmark/benchmark_cafa_fmax.pdf" "$OUT/benchmark/benchmark_metric_grid.pdf" "$OUT/benchmark/benchmark_coverage_vs_fmax.pdf" "$OUT/benchmark/benchmark_bootstrap_boxplots.pdf" "$OUT/reviewer/figure_leakage_residual_identity.pdf")
for f in "${expected[@]}"; do [[ -s "$f" ]] || { echo "ERROR: expected output missing: $f" >&2; exit 3; }; done

python - "$OUT/benchmark/benchmark_metrics_plotted.csv" <<'PY'
import sys, pandas as pd
allowed={'naive','blast','blast_max','diamond','diamond_max','foldseek','foldseek_max','deepfri_sequence','deepfri_structure','dpfunc','heal'}
actual=set(pd.read_csv(sys.argv[1])['method'].dropna().astype(str))
if actual != allowed: raise SystemExit(f'ERROR: plotted benchmark methods differ: {sorted(actual)}')
print('Verified baseline-only plotted metrics:', ', '.join(sorted(actual)))
PY
echo "All figure sets complete: $OUT"
echo "PDF files: $(find "$OUT" -type f -name '*.pdf' | wc -l)"
echo "CSV files: $(find "$OUT" -type f -name '*.csv' | wc -l)"
