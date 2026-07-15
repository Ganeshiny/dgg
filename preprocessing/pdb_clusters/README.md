# PDB Cluster Splits — Viridiplantae

This directory contains scripts for splitting the Viridiplantae protein dataset using
PDB's weekly DIAMOND sequence clusters, addressing all correctness issues with the
original MMseqs2-based split approach.

## Files

| Script | Purpose |
|---|---|
| `build_entity_map.py` | Parse all local CIF.gz files to build `entity_map.json`: auth_chain → entity_id |
| `split_by_pdb_clusters.py` | Download PDB cluster files, union-find merge, bin-pack split, GO diagnostics |
| `analyse_and_plot.py` | Generate all 12 diagnostic plots |
| `plots/` | Output directory for all generated figures |

## Usage (in order)

```bash
# Step 1: build the auth-chain → entity-number map (run once, ~2-3 min)
python3 preprocessing/pdb_clusters/build_entity_map.py

# Step 2: generate splits for all thresholds (downloads 6 cluster files)
python3 preprocessing/pdb_clusters/split_by_pdb_clusters.py --all

# Step 3: generate all diagnostic plots
python3 preprocessing/pdb_clusters/analyse_and_plot.py

# Step 4 (optional, requires blastp): BLAST identity analysis + comparison plots
python3 preprocessing/pdb_clusters/analyse_and_plot.py --skip_blast  # parse cached TSV
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
The existing split files in `preprocessing/data/split_files/` are **not modified**.
To revert: delete the `pdb_split_*` directories.

## ARC rebuild

Run `arc slurms/run_pdb_cluster_rebuild.slurm` from the repository root. It retrieves fresh inputs, builds canonical sequences and annotations, regenerates contact maps, creates all PDB-cluster thresholds, writes ontology/threshold/split pickle datasets under `preprocessing/data/datasets/`, validates every split, and then generates plots. Set `DGG_DATA_ROOT` to a `/work/...` or `/scratch/...` location when the home quota is insufficient.
