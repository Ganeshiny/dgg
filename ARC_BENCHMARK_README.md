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

TransFun, eggNOG-mapper, Hayai, GOMAP, HEAL, GAT-GO, and DeepGraphGO remain supported as explicit opt-ins. They are not part of the deadline default because of runtime, released-data, or manual upstream constraints documented below. They must not be named as completed comparisons unless their prediction files are actually present and evaluated.

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
| GAT-GO | `SOTA/GAT-GO`, environment `dgg_gat_go`; official feature/model bundle required |
| DeepGraphGO | `SOTA/DeepGraphGO`, exact legacy CPU environment `dgg_deepgraphgo` |
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

## GAT-GO opt-in and strict feature audit

Install the official repository, model, GO index, and precomputed feature
bundle:

```bash
DGG_SOTA_SETUP_METHODS=gat_go sbatch 'arc slurms/arc_01_setup_sota.slurm'
```

Setup pins the audited upstream revision
`90ec6d1067a893d4a51be715e41daf9fa4732952`.

The upstream GAT-GO release predicts only from serialized per-protein objects
containing PSSMs, ESM-1b embeddings, contact edges, and sequence features. It
does not publish the pipeline needed to create these objects for arbitrary new
proteins. The ARC adapter therefore audits all 1,508 validation+test proteins
against the official feature bundle, verifies identifier and tensor dimensions,
and aborts if even one feature file is absent or incompatible. It explicitly
does not read the annotation `label` field stored in upstream feature files.
A subset result is never normalized or plotted.

After setup succeeds, request GAT-GO alone first:

```bash
export DGG_BENCHMARK_METHODS=gat_go
sbatch 'arc slurms/run_full_benchmark.slurm'
```

If the released feature bundle does not cover all locked proteins, the stage
writes `raw/gat_go/preflight.json` and exits. Generating substitute features
would no longer be a reproducible run of the released method.

## DeepGraphGO opt-in

DeepGraphGO supports new FASTA queries through its released one-iteration
PSI-BLAST mapping into the fixed PPI network. Install its repository, split
data archive, three checkpoints per ontology, and exact legacy runtime:

```bash
DGG_SOTA_SETUP_METHODS=deepgraphgo sbatch 'arc slurms/arc_01_setup_sota.slurm'
```

Setup pins the audited upstream revision
`efdb1cb9425f4f48e4613c0a89e603f5542bcb19`.

The published PyTorch 1.6/DGL 0.4 code cannot execute on ARC's L40 GPU stack.
Setup therefore creates a CPU-only legacy environment and changes only device
placement and the BLAST thread count; model layers, weights, graph sampling,
and the three-model ensemble are unchanged. ARC protein IDs are replaced by
synthetic query IDs before BLAST so a coincidental identifier match cannot
bypass sequence mapping.

Run it independently so a GAT-GO feature-coverage failure cannot prevent this
valid sequence-to-network evaluation:

```bash
export DGG_BENCHMARK_METHODS=deepgraphgo
sbatch 'arc slurms/run_full_benchmark.slurm'
```

## SProf-GO

Create the isolated secure runtime, then submit the v3 launcher only after the
setup job succeeds:

```bash
mkdir -p logs
setup_job=$(sbatch --parsable 'arc slurms/setup_sprof_go_arc.sh')
sbatch --dependency=afterok:${setup_job} 'arc slurms/run_sprof_go_arc_v3.slurm'
```

The setup job creates `dgg_sprof_go` without modifying the main `deepgreengo`
environment. It pins PyTorch 2.6.0 with CUDA 11.8 and performs a full local
ProtT5 load before marking setup complete. PyTorch versions below 2.6 are
rejected because the bundled `.bin` checkpoint otherwise reaches the
`torch.load` path affected by CVE-2025-32434.

All three dataset readers (input preparation, evaluation, and Smin) use a
narrow compatibility unpickler that maps NumPy 2's `numpy._core` pickle
namespace to NumPy 1's `numpy.core`. The official SProf-GO prediction code
remains unchanged while the preparation and evaluation stages run in the
project environment.

## Fairness controls

- BLAST, DIAMOND, and Foldseek search only the locked training proteins/structures.
- Each reports a bitscore-weighted top-10 score and a separate maximum-normalized-identity score; all use identical 50% query/target coverage filters.
- Validation and test proteins are predicted together, but only validation labels are used for DeepGreenGO's fixed operating threshold.
- Every output is mapped to the locked GO vocabulary and propagated to represented ancestors at evaluation time under one common true-path rule; external adapters may also pre-propagate, which is idempotent. Protein and term coverage are audited.
- Missing predictions remain zero in the full 754-protein analysis.
- Pretrained external models are not described as leakage-free because historical training overlap may be unknown. In particular, DeepGOPlus combines a CNN with DIAMOND transfer from its own released Swiss-Prot-derived reference set, DeepGO-SE, HEAL, GAT-GO, and DeepGraphGO use released checkpoints or reference networks trained outside the locked ARC split. Their scores are descriptive unless those original corpora are explicitly audited against the locked test sequences.
- The locked test set contains 754 PDB chains but only 140 unique amino-acid sequences. Point estimates remain chain-level for compatibility with the locked benchmark; confidence intervals use an identical-sequence cluster bootstrap so duplicate chains are not treated as independent evidence.
- Structural quality analysis uses experimental-chain residue coverage, not pLDDT.
