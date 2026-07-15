# ARC Hybrid tuning on the homology-controlled split

The workflow uses the **nominal 30% identity / 80% coverage split** (also called
the 30%-cluster homology-aware split). It must not be described as leakage-free:
the independent audit in `blast_leakage.json` reports 116 of 754 test sequences
(15.38%) with a qualifying test-to-train match.

## Launch

From `/home/ganeshiny.sridharan/dgg/deep-green-GO` on ARC:

```bash
sbatch "arc slurms/run_hybrid_tuning.slurm"
```

That CPU controller performs the following gated sequence:

1. verifies `strict_mmseqs/manifest_30.json` (30% identity, 0.80 coverage), the
   strict cluster file, split verification, 6,026/754/754 counts, disjoint IDs,
   FASTA/record sequence agreement, ontology keys, contact-map availability,
   label dimensions, and the residual-similarity denominator;
2. builds shared ProtBERT/PyG graph files on CPU and ontology/split PKLs under
   `preprocessing/data_arc_rebuild_2026_07_14/arc_tuning/`;
3. runs a PKL/dataloader smoke batch;
4. generates 40 seeded random-search configurations and submits them to
   `gpu-l40`, with at most four concurrent tasks;
5. selects each ontology's parameters using validation micro-Fmax only; and
6. launches five seeded Hybrid repeats using the selected parameters.

The tuning entry point only opens the train and validation PKLs. The test PKLs
are built and schema-checked but remain reserved for final evaluation.

The search includes log-uniform learning rate (`1e-5` to `3e-3`) and weight
decay (`1e-7` to `1e-2`), plus the requested dropout, hidden dimension, batch
size, gradient clipping, patience, BCE/focal-loss, and focal-gamma choices.
Positive weights and rare-term masks are computed from training labels only.
Each result includes validation micro-Fmax, rare-term micro-Fmax, Brier score,
10-bin expected calibration error, threshold sensitivity, configuration,
history, and the best checkpoint.

## Hybrid-JK confirmation

After `selected_hybrid.json` exists, reuse those Hybrid parameters for the
five-seed Hybrid-JK confirmation (no second search):

```bash
DGG_MODE=confirm_jk sbatch "arc slurms/run_hybrid_tuning.slurm"
```

No mode downloads data, reclusters PDBs, regenerates structures/contact maps,
or uses a GPU for preprocessing. Higher homology thresholds remain outside this
tuning workflow for a later homology-split ablation.
