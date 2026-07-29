# ARC commands for the SPROF-GO benchmark

From the DeepGreenGO repository on the ARC login node:

```bash
conda activate deepgreengo
python -m pip install -r requirements-sprof-go.txt
bash "arc slurms/setup_sprof_go_arc.sh"
sbatch "arc slurms/run_sprof_go_arc.slurm"
```

The canonical batch script is `run_sprof_go_arc.slurm`. It runs the complete
held-out test set in one SPROF-GO invocation, evaluates MF/BP/CC after aligning
GO vocabularies, computes macro/micro Fmax, AUPR, AUROC and Smin, and writes:

```text
arc_tuning_cafa/external_benchmarks/sprof_go/
  arc_test.fa
  raw/arc_test_all_preds.txt
  raw/arc_test_top_preds.txt
  evaluation/metrics.json
  evaluation/predictions_long.csv
```

If the DeepGreenGO metrics JSON is elsewhere, submit with:

```bash
sbatch --export=ALL,DGG_METRICS=/absolute/path/to/deepgreengo_metrics.json \
  "arc slurms/run_sprof_go_arc.slurm"
```
