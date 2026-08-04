# Generated figures

This directory is present in every clone so local and ARC runs use the same
output location. Rendered figures are intentionally ignored by Git because
PDF/PNG/TIFF files are generated artifacts and cause binary merge conflicts.

The GPU benchmark job performs inference and evaluation, then automatically
queues `run_all_figures.slurm` as a dependent CPU job. That CPU job rebuilds
`main_text`, `supplementary`, `supplementary_tables`,
`supplementary_tuning`, `reviewer`, and `benchmark`. Submit from any
directory by exporting the checkout path:

```bash
export DGG_PROJECT_ROOT="$PWD"
sbatch --export=ALL,DGG_PROJECT_ROOT="$DGG_PROJECT_ROOT" \
  "arc slurms/run_full_benchmark.slurm"
```

If all benchmark results already exist and only figures need regeneration:

```bash
sbatch --export=ALL,DGG_PROJECT_ROOT="$PWD" \
  "arc slurms/run_all_figures.slurm"
```

The figure job verifies at least 3 main-text PDFs, 19 supplementary PDFs,
10 supplementary-table CSVs, 6 tuning PDFs, 2 reviewer PDFs, and 4 benchmark
PDFs before reporting success.
