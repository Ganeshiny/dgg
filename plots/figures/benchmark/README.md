# Baseline benchmark figures

`src/plot_benchmark.py` writes the baseline-only benchmark figures and its
filtered `benchmark_metrics_plotted.csv` here. The allowed methods are:

- CAFA naive
- BLAST (top-10 and maximum-identity transfer)
- DIAMOND (top-10 and maximum-identity transfer)
- Foldseek (top-10 and maximum-identity transfer)
- DeepFRI (sequence and structure modes)
- DPFunc
- HEAL

DeepGreenGO, DeepGO-SE, DeepGOPlus, InterProScan, GAT-GO, SPROF-GO, and every
other method are excluded from these outputs even if their results exist in
the benchmark workspace.
