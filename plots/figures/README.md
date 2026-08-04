# Generated figures

This directory is present in every clone so local and ARC runs use the same
output location. Rendered figures are intentionally ignored by Git because
PDF/PNG/TIFF files are generated artifacts and cause binary merge conflicts.

The baseline-only benchmark figures are written to `benchmark/` by:

```bash
sbatch "arc slurms/run_baseline_figures.slurm"
```
