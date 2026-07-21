# ARC sequence/structure GO benchmark

This benchmark compares DeepGreenGO with sequence, structure, domain, plant-specific, and modern deep-learning methods on the **locked nominal 30%-identity/80%-coverage split**. The test set must contain exactly **754 experimental PDB chains**. It does not use AlphaFold structures or pLDDT filters.

## One ARC submission

From `/home/ganeshiny.sridharan/dgg/deep-green-GO`:

```bash
sbatch 'arc slurms/run_full_benchmark.slurm'
```

ARC GPU partitions currently impose a 24-hour maximum wall time. This job requests 23 hours 55 minutes and is resumable: if it reaches the Slurm time limit, submit the same command again. A stage is skipped only after its `.done` marker has been written.

Outputs are written under:

```text
arc_benchmark/nominal_30_identity_80_coverage/
├── benchmark_manifest.json
├── inputs/
├── predictions/
├── raw/
├── results/
│   ├── benchmark_metrics.csv
│   ├── bootstrap_metrics.csv
│   ├── paired_differences_vs_deepgreengo.csv
│   └── coverage_and_mapping_audit.csv
└── plots/
    ├── 01_cafa_metrics.png/.pdf
    └── 02_prediction_coverage.png/.pdf
```

The primary metrics are protein-centric CAFA Fmax, conditional-information Smin, micro/macro AUPR, and coverage. Confidence intervals use 1,000 paired protein bootstraps. AUPR and AUROC are deliberately left undefined for binary annotation pipelines such as InterProScan, eggNOG-mapper, GOMAP, and Hayai.

The default comparison includes:

- frequency: training-label prevalence (naive);
- sequence homology: BLAST and DIAMOND, each with weighted top-10 and maximum-similarity GO transfer;
- structure homology: Foldseek with weighted top-10 and maximum-similarity GO transfer;
- conventional annotation: InterProScan and eggNOG-mapper;
- general deep learning: DeepFRI sequence, DeepFRI structure, TransFun, DPFunc, DeepGOPlus, and DeepGO-SE;
- plant-oriented annotation: current Hayai v3.2 and GOMAP;
- DeepGreenGO: the five best seed checkpoints, averaged as the primary ensemble.

A separate naive-plant or plant-reference-DIAMOND result is not fabricated here. The locked split records do not contain taxonomy identifiers, so they cannot be separated into plant and non-plant training subsets reproducibly. If a verified protein-to-NCBI-taxonomy manifest is added later, those controls should be added explicitly.

## Required installations and data

The single Slurm script orchestrates the work but does not silently download version-changing databases. Its preflight stage stops before computation and names every missing item.

Expected defaults:

| Method | Default location/environment |
|---|---|
| DeepGreenGO | checkpoints: `arc_tuning_cafa/five_seed_hybrid`; shared graphs: `arc_tuning/graphs_protbert`; environment `deepgreengo` |
| BLAST/DIAMOND/Foldseek | executables visible in `deepgreengo` |
| InterProScan | set `DGG_INTERPROSCAN=/absolute/path/interproscan.sh` |
| DeepFRI | `baselines/DeepFRI`, environment `dgg_sota_tf` |
| TransFun | `SOTA/TransFun`, environment `dgg_sota_torch` |
| DPFunc | `SOTA/DPFunc`, environment `dgg_sota_torch` |
| DeepGOPlus | `SOTA/deepgoplus/data`, environment `dgg_deepgoplus` |
| DeepGO-SE | `SOTA/deepgo2`, environment `dgg_deepgose` |
| eggNOG-mapper | `SOTA/eggnog-mapper`, database under its `data/`, environment `dgg_eggnog` |
| Hayai v3.2 | `SOTA/HayaiAnnotation`, environment `hayai_v3.2` |
| GOMAP | completed GAF or a site-specific command; see below |

Paths and environment names can be overridden without editing the Slurm file, for example:

```bash
DGG_INTERPROSCAN=/home/ganeshiny.sridharan/tools/interproscan/interproscan.sh \
DGG_DEEPGO_ROOT=/home/ganeshiny.sridharan/tools/deepgo2 \
sbatch 'arc slurms/run_full_benchmark.slurm'
```

## GOMAP limitation

The upstream GOMAP workflow cannot be made fully unattended from its documented release: it requires a MySQL-backed PANNZER setup and a manual Argot2.5 web submission between `pipeline1.py` and `pipeline2.py`. Labeling a reduced or rewritten workflow as “GOMAP” would be scientifically misleading.

Use either:

```bash
DGG_GOMAP_GAF=/absolute/path/completed_gomap_annotations.gaf \
sbatch 'arc slurms/run_full_benchmark.slurm'
```

or provide a cluster-specific command that writes `arc_benchmark/nominal_30_identity_80_coverage/raw/gomap_predictions.gaf`:

```bash
DGG_GOMAP_COMMAND='/absolute/path/run_site_gomap.sh' \
sbatch 'arc slurms/run_full_benchmark.slurm'
```

The default benchmark is strict: if GOMAP is requested and neither option is provided, preflight fails instead of silently omitting the method. To run a staged subset, set the comma-separated method list explicitly:

```bash
DGG_BENCHMARK_METHODS=hybrid,naive,blast,diamond,foldseek,interproscan,eggnog_mapper,deepfri_sequence,deepfri_structure,transfun,dpfunc,deepgoplus,deepgose,hayai \
sbatch 'arc slurms/run_full_benchmark.slurm'
```

This subset run is not the complete final comparison until GOMAP is added and evaluation is rerun.

## Fairness controls

- BLAST, DIAMOND, and Foldseek search only the locked training proteins/structures.
- Each reports a bitscore-weighted top-10 score and a separate maximum-normalized-identity score; all use identical 50% query/target coverage filters.
- Validation and test proteins are predicted together, but only validation labels are used for DeepGreenGO's fixed operating threshold.
- Every external output is mapped to the locked GO vocabulary, propagated to represented ancestors, and audited for protein and term coverage.
- Missing predictions remain zero in the full 754-protein analysis.
- Pretrained external models are not described as leakage-free because historical training overlap may be unknown.
- Structural quality analysis uses experimental-chain residue coverage, not pLDDT.

