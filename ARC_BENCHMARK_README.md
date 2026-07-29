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
|-- benchmark_manifest.json
|-- inputs/
|-- predictions/
|-- raw/
|-- results/
|   |-- benchmark_metrics.csv
|   |-- bootstrap_metrics.csv
|   |-- paired_differences_vs_deepgreengo.csv
|   `-- coverage_and_mapping_audit.csv
`-- plots/
    |-- 01_cafa_metrics.png/.pdf
    `-- 02_prediction_coverage.png/.pdf
```

The primary metrics are protein-centric CAFA Fmax, conditional-information Smin, micro/macro AUPR, and coverage. Confidence intervals use 1,000 paired identical-sequence-cluster bootstraps. AUPR and AUROC are deliberately left undefined for binary annotation pipelines such as InterProScan, eggNOG-mapper, GOMAP, and Hayai.

The deadline-safe default comparison includes:

- frequency: training-label prevalence (naive);
- sequence homology: BLAST and DIAMOND, each with weighted top-10 and maximum-similarity GO transfer;
- structure homology: Foldseek with weighted top-10 and maximum-similarity GO transfer;
- conventional annotation: InterProScan;
- verified deep learning: DeepFRI sequence, DeepFRI structure, DPFunc, DeepGOPlus, and DeepGO-SE;
- DeepGreenGO: the five best seed checkpoints, averaged as the primary ensemble.

TransFun, eggNOG-mapper, Hayai, and GOMAP remain supported as explicit opt-ins, but they are not part of the deadline default because their ARC environments/data or manual upstream steps have not passed end-to-end verification. They must not be named as completed comparisons unless their prediction files are actually present and evaluated.

A separate naive-plant or plant-reference-DIAMOND result is not fabricated here. The locked split records do not contain taxonomy identifiers, so they cannot be separated into plant and non-plant training subsets reproducibly. If a verified protein-to-NCBI-taxonomy manifest is added later, those controls should be added explicitly.

## Required installations and data

The single Slurm script orchestrates the work but does not silently download version-changing databases. Its preflight stage stops before computation and names every missing item.

Install and verify the five resolved deep-learning baselines and InterProScan first:

```bash
sbatch 'arc slurms/arc_01_setup_sota.slurm'
```

The setup job is strict and resumable. Its default creates or verifies separate environments for DeepFRI, DPFunc, DeepGOPlus, DeepGO-SE, and InterProScan's Java 11 runtime; downloads their official pretrained data/models; verifies required imports and files; and exits non-zero if any check fails. Existing multi-gigabyte archives are reused. A successful log ends with `[SETUP COMPLETE] All requested SOTA dependencies and model files passed verification`.

After setup succeeds, submit the deadline-safe benchmark profile directly:

```bash
sbatch 'arc slurms/run_full_benchmark.slurm'
```

Expected defaults:

| Method | Default location/environment |
|---|---|
| DeepGreenGO | checkpoints: `arc_tuning_cafa/five_seed_hybrid`; shared graphs: `arc_tuning/graphs_protbert`; environment `deepgreengo` |
| BLAST/DIAMOND | executables visible in `deepgreengo` |
| Foldseek | executable visible in `DGG_FOLDSEEK_ENV` (defaults to `dgg_foldseek`, then falls back to `deepgreengo`) |
| InterProScan | `SOTA/interproscan/interproscan.sh`; Java environment `dgg_interproscan_java11` |
| DeepFRI | `baselines/DeepFRI`, environment `dgg_sota_tf` |
| TransFun | `SOTA/TransFun`, environment `dgg_transfun` |
| DPFunc | `SOTA/DPFunc`, environment `dgg_dpfunc` |
| DeepGOPlus | `SOTA/deepgoplus/data`, environment `dgg_deepgoplus_py37` (Python 3.7) |
| DeepGO-SE | `SOTA/deepgo2`, environment `dgg_deepgose` |
| HEAL | `SOTA/HEAL`, environment `dgg_heal`; ESM-1b under `SOTA/torch_cache` |
| Struct2GO | `SOTA/Struct2GO`; externally prepared ARC score files are required |
| eggNOG-mapper | `SOTA/eggnog-mapper`, database under its `data/`, environment `dgg_eggnog` |
| Hayai v3.2 | `SOTA/HayaiAnnotation`, environment `hayai_v3.2` |
| GOMAP | completed GAF or a site-specific command; see below |

Paths and environment names can be overridden without editing the Slurm file, for example:

```bash
DGG_INTERPROSCAN=/home/ganeshiny.sridharan/tools/interproscan/interproscan.sh \
DGG_DEEPGO_ROOT=/home/ganeshiny.sridharan/tools/deepgo2 \
sbatch 'arc slurms/run_full_benchmark.slurm'
```

Foldseek can be kept in a small dedicated environment instead of modifying the trained-model environment:

```bash
conda create -y -n dgg_foldseek -c conda-forge -c bioconda foldseek
sbatch 'arc slurms/run_full_benchmark.slurm'
```

The launcher resolves the executable to its absolute path during preflight, so
the Foldseek stage can still be driven by the main `deepgreengo` Python
environment. Set `DGG_FOLDSEEK_ENV` only if you chose a different environment
name.

## GOMAP limitation

The upstream GOMAP workflow cannot be made fully unattended from its documented release: it requires a MySQL-backed PANNZER setup and a manual Argot2.5 web submission between `pipeline1.py` and `pipeline2.py`. Labeling a reduced or rewritten workflow as "GOMAP" would be scientifically misleading.

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

The launcher is strict by default. If any explicitly requested method fails preflight, it exits before expensive computation instead of silently dropping that comparator. GOMAP therefore fails preflight when requested without a completed GAF or site command. Use `DGG_BENCHMARK_SKIP_UNAVAILABLE=1` only for a deliberately incomplete diagnostic run, and report the dropped methods recorded in `preflight_dropped_methods.txt`.

TransFun can be retried later as an explicit opt-in after its archived PyTorch/PyG environment is repaired independently:

```bash
DGG_SOTA_SETUP_METHODS=transfun sbatch 'arc slurms/arc_01_setup_sota.slurm'
```

It is not necessary for the deadline-safe five-model deep-learning comparison.

## HEAL opt-in

HEAL is runnable on the locked experimental PDB chains with its released
PDBch-only checkpoints. Install the repository, modernized GPU environment,
and ESM-1b checkpoint once:

```bash
DGG_SOTA_SETUP_METHODS=heal sbatch 'arc slurms/arc_01_setup_sota.slurm'
```

Then include `heal` in the requested method list and submit the normal
benchmark. The default variant is `pdb-only` (upstream `*CL.pt` checkpoints).

```bash
export DGG_BENCHMARK_METHODS=hybrid,naive,blast,diamond,foldseek,interproscan,deepfri_sequence,deepfri_structure,dpfunc,deepgoplus,deepgose,heal
sbatch 'arc slurms/run_full_benchmark.slurm'
```

The combined PDBch+AFch checkpoints are not labeled as this benchmark method.

HEAL uses ESM-1b, whose released model accepts at most 1,022 residues. The
current validation+test FASTA has 40 of 1,508 chains above that limit. The
runner does not truncate them: it records them in `raw/heal/manifest.json`
and their predictions remain zero. Per-protein score caches make the stage
resumable across ARC's 24-hour job limit.

## Struct2GO opt-in and upstream limitation

The Struct2GO repository can be installed for provenance and checkpoint
inspection:

```bash
DGG_SOTA_SETUP_METHODS=struct2go sbatch 'arc slurms/arc_01_setup_sota.slurm'
```

The released Struct2GO code is not a general PDB predictor. Its evaluation
entry point consumes pickled, precomputed human-protein datasets containing
SeqVec features, contact maps, Node2Vec structural features, and label
networks. Its documented raw-data workflow additionally calls a Spark/Angel
Node2Vec job through IntelliJ. Running its bundled test pickle would therefore
not predict the locked ARC proteins and must not be reported as an ARC result.

The benchmark supports genuine externally generated Struct2GO scores in
three tab-separated files with rows `protein_id GO:nnnnnnn score`:

```bash
export DGG_STRUCT2GO_MF=/absolute/path/struct2go_mf.tsv
export DGG_STRUCT2GO_BP=/absolute/path/struct2go_bp.tsv
export DGG_STRUCT2GO_CC=/absolute/path/struct2go_cc.tsv
export DGG_BENCHMARK_METHODS=hybrid,naive,blast,diamond,foldseek,interproscan,deepfri_sequence,deepfri_structure,dpfunc,deepgoplus,deepgose,struct2go
sbatch 'arc slurms/run_full_benchmark.slurm'
```

Preflight fails if any file is missing, so an unavailable Struct2GO run cannot
silently appear in plots.

## SProf-GO

Use the tracked v3 launcher:

```bash
sbatch 'arc slurms/run_sprof_go_arc_v3.slurm'
```

All three dataset readers (input preparation, evaluation, and Smin) use a
narrow compatibility unpickler that maps NumPy 2's `numpy._core` pickle
namespace to NumPy 1's `numpy.core`. This keeps the legacy SProf-GO runtime
unchanged while allowing it to read datasets serialized in the newer
DeepGreenGO environment.

## Fairness controls

- BLAST, DIAMOND, and Foldseek search only the locked training proteins/structures.
- Each reports a bitscore-weighted top-10 score and a separate maximum-normalized-identity score; all use identical 50% query/target coverage filters.
- Validation and test proteins are predicted together, but only validation labels are used for DeepGreenGO's fixed operating threshold.
- Every output is mapped to the locked GO vocabulary and propagated to represented ancestors at evaluation time under one common true-path rule; external adapters may also pre-propagate, which is idempotent. Protein and term coverage are audited.
- Missing predictions remain zero in the full 754-protein analysis.
- Pretrained external models are not described as leakage-free because historical training overlap may be unknown. In particular, DeepGOPlus combines a CNN with DIAMOND transfer from its own released Swiss-Prot-derived reference set, DeepGO-SE uses released models trained on an external Swiss-Prot corpus, and HEAL/Struct2GO use released checkpoints trained outside the locked ARC split. Their scores are descriptive unless those original corpora are explicitly audited against the locked test sequences.
- The locked test set contains 754 PDB chains but only 140 unique amino-acid sequences. Point estimates remain chain-level for compatibility with the locked benchmark; confidence intervals use an identical-sequence cluster bootstrap so duplicate chains are not treated as independent evidence.
- Structural quality analysis uses experimental-chain residue coverage, not pLDDT.
