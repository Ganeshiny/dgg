# Generated figures

This directory is present in every clone so local and ARC runs use the same
output location. Rendered figures are intentionally ignored by Git because
PDF/PNG/TIFF files are generated artifacts and cause binary merge conflicts.

On ARC, generate the restricted benchmark first and then rebuild the complete
figure tree. Exporting the checkout path makes submission independent of the
directory from which `sbatch` is invoked:

```bash
export DGG_PROJECT_ROOT="$PWD"
BENCH_JOB=$(sbatch --parsable \
  --export=ALL,DGG_PROJECT_ROOT="$DGG_PROJECT_ROOT" \
  "arc slurms/run_full_benchmark.slurm")
sbatch --dependency="afterok:${BENCH_JOB}" \
  --export=ALL,DGG_PROJECT_ROOT="$DGG_PROJECT_ROOT" \
  "arc slurms/run_all_figures.slurm"
```

If complete benchmark CSV files already exist, submit only:

```bash
sbatch --export=ALL,DGG_PROJECT_ROOT="$PWD" \
  "arc slurms/run_all_figures.slurm"
```

`run_baseline_figures.slurm` remains available when only the four benchmark
plots need to be refreshed.
