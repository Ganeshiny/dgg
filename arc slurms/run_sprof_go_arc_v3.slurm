#!/usr/bin/env bash
#SBATCH --job-name=dgg_sprof
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sprof_go_%j.out
#SBATCH --error=logs/sprof_go_%j.err
set -eo pipefail
DGG_ROOT=${DGG_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}
TUNING_ROOT=${DGG_TUNING_ROOT:-$DGG_ROOT/arc_tuning_cafa}
DATA_ROOT=${DGG_DATA_ROOT:-$DGG_ROOT/preprocessing/data_arc_rebuild_2026_07_14}
SPROF_ROOT=${SPROF_ROOT:-/home/ganeshiny.sridharan/dgg/external/SPROF-GO}
PROTT5_ROOT=${PROTT5_ROOT:-/home/ganeshiny.sridharan/dgg/external/prot_t5_xl_uniref50}
OUT=${SPROF_OUT:-$TUNING_ROOT/external_benchmarks/sprof_go}
DGG_METRICS=${DGG_METRICS:-$TUNING_ROOT/test_evaluation/metrics.json}
# ARC dataset pickles reference src.arc_dataset, so the project root must be
# importable in every preparation/evaluation process.
export PYTHONPATH="$DGG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/raw" "$OUT/evaluation"
source /home/ganeshiny.sridharan/miniconda3/etc/profile.d/conda.sh
conda activate "${SPROF_CONDA_ENV:-deepgreengo}"
set -u
python -c 'import src.arc_dataset, sentencepiece, sklearn, scipy, transformers, torch'
python "$DGG_ROOT/scripts/prepare_sprof_go_input.py" --tuning-root "$TUNING_ROOT" --all-fasta "$DATA_ROOT/all_sequences.fasta" --output "$OUT/arc_test.fa"
python "$DGG_ROOT/scripts/run_sprof_go_official.py" --sprof-root "$SPROF_ROOT" --prott5-root "$PROTT5_ROOT" --fasta "$OUT/arc_test.fa" --outpath "$OUT/raw" --feat-bs "${SPROF_FEAT_BS:-2}" --pred-bs "${SPROF_PRED_BS:-4}"
python "$DGG_ROOT/scripts/evaluate_sprof_go.py" --predictions "$OUT/raw/arc_test_all_preds.txt" --tuning-root "$TUNING_ROOT" --output-dir "$OUT/evaluation"
python "$DGG_ROOT/scripts/add_sprof_go_smin.py" --evaluation-dir "$OUT/evaluation" --tuning-root "$TUNING_ROOT"
if [[ -s "$DGG_METRICS" ]]; then
  python "$DGG_ROOT/src/plot_sprof_go_benchmark.py" --sprof "$OUT/evaluation/metrics.json" --deepgreengo "$DGG_METRICS" --output-dir "$DGG_ROOT/plots/sprof_go_benchmark"
else
  echo "Benchmark complete; plot skipped because DeepGreenGO metrics were not found at $DGG_METRICS"
fi
