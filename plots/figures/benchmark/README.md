# DeepGreenGO benchmark-comparison figures

`src/plot_benchmark.py` writes the locked comparison figures and filtered
`benchmark_metrics_plotted.csv` here. The retained methods are:

- DeepGreenGO (this work)
- CAFA naive
- BLAST (top-10 and maximum-identity transfer)
- DIAMOND (top-10 and maximum-identity transfer)
- Foldseek (top-10 and maximum-identity transfer)
- DeepFRI (sequence and structure modes)
- DPFunc
- HEAL

Each retained method and transfer variant has its own fixed color. Colors do
not encode evidence families. Plot labels use protein-centric Fmax/Smin and
term-centric AUPR terminology, without directional guidance annotations.

DeepGO-SE, DeepGOPlus, InterProScan, GAT-GO, SPROF-GO, and every other method
are excluded even if stale results exist in the benchmark workspace.
InterProScan may run internally to create DPFunc features, but it is not
evaluated or plotted as a method.
