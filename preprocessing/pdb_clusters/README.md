# PDB Cluster Splits — Viridiplantae

This directory contains scripts for splitting the Viridiplantae protein dataset using
PDB's weekly DIAMOND sequence clusters, addressing all correctness issues with the
original MMseqs2-based split approach.

## Files

| Script | Purpose |
|---|---|
| `split_by_pdb_clusters.py` | Download PDB cluster files, union-find merge, bin-pack split, GO diagnostics |
| `analyse_and_plot.py` | Generate all 12 diagnostic plots |
| `plots/` | Output directory for all generated figures |

## Usage (in order)

```bash
# Fresh ARC run: use the Slurm orchestrator from the repository root.
sbatch "arc slurms/run_pdb_cluster_rebuild.slurm"

# Equivalent manual order for debugging:
python3 preprocessing/pdb_clusters/fetch_data.py --data-dir "$DGG_DATA_ROOT"
python3 preprocessing/pdb_clusters/prepare_dataset.py --data-dir "$DGG_DATA_ROOT"
python3 preprocessing/create_cmaps.py -annot "$DGG_DATA_ROOT/pdb2go.tsv" -seqs "$DGG_DATA_ROOT/all_sequences.fasta" -struc_dir "$DGG_DATA_ROOT/structure_files"
python3 preprocessing/pdb_clusters/split_by_pdb_clusters.py --all
python3 preprocessing/pdb_clusters/verify_splits.py
python3 preprocessing/pdb_clusters/analyse_and_plot.py
```

## Design Decisions

### Union-Find (not naive merge)
The split script uses a proper transitive union-find over chain IDs:
1. All chains of the **same PDB ID** are pre-merged into one component.
2. All chains whose entity IDs fall in the **same PDB cluster** are merged.
Since a PDB ID can span multiple entity clusters (different protein types),
and the same cluster can contain entities from multiple PDB IDs,
a single-pass merge would miss transitive connections. Union-find guarantees
that all transitively connected chains end up in the same super-cluster.

### Bin-packing by protein count
Clusters are randomly shuffled (seed=42) and greedily assigned to test/valid/train
by protein count — not cluster count. This avoids the skewed-cluster-size trap where
one enormous cluster landing in train would make the "80/10/10" ratio meaningless.
Achieved vs target fractions are logged in `split_log.json`.

### Unclustered IDs
Chain IDs that cannot be mapped to any PDB cluster (missing entity map entry or
entity not present in the RCSB weekly file) are treated as **singletons** and
assigned to whichever split needs filling, starting with test. Their count is
logged in `split_log.json` as `unclustered_chains`.

### Thresholds
Runs at 30%, 40%, 50%, 70%, 90%, 95%. The 100% file is excluded — it provides
no biologically meaningful separation (near-identical sequences to self).

### Seed
Fixed seed = 42. Logged in `split_log.json`.

## Output per threshold

```
preprocessing/data/pdb_splits/threshold_<threshold>/
├── _train.txt
├── _valid.txt
├── _test.txt
├── _train_sequences.fasta
├── _valid_sequences.fasta
├── _test_sequences.fasta
├── split_log.json           # sizes, achieved fracs, GO coverage summary
└── go_coverage_full.pkl     # per-term counts for plotting
```

## Non-destructive
All outputs go to `preprocessing/data/pdb_splits/threshold_<threshold>/`.
The legacy `split_files/` tree is not read by this pipeline. Outputs are isolated under `pdb_splits/` and `datasets/`.

## ARC rebuild

Run `arc slurms/run_pdb_cluster_rebuild.slurm` from the repository root. It retrieves fresh inputs, builds canonical sequences and annotations, regenerates contact maps, creates all PDB-cluster thresholds, writes ontology/threshold/split pickle datasets under `preprocessing/data/datasets/`, validates every split, and then generates plots. Set `DGG_DATA_ROOT` to a `/work/...` or `/scratch/...` location when the home quota is insufficient.
