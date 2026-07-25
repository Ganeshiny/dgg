# Main-text figure legends

**Figure 1.** Architecture and input-modality ablation. Micro-F$_{max}$ on the held-out test split for five architectures under three input modalities, across the three GO sub-ontologies. Points Rows are input modality and columns are ontology; points are individual training seeds (n = 5) and the horizontal bar is the mean ± s.d. Modality is faceted rather than encoded as a marker shape, so no symbol decoding is required. Full and sequence-only are near-identical for every architecture, while structure-only collapses — the sequence representation carries the signal.

**Figure 2.** Model ranking depends on the metric family. Identical layout and colour mapping in both rows; only the metric changes. F$_{max}$ (top) favours the graph-aware Hybrid variants, while AUROC (bottom) favours the MLP/GCN baselines. Neither family alone supports a single best-model claim, and the two must be reported together.

**Figure 3.** Generalisation across sequence-homology bins. Micro-F$_{max}$ by maximum BLAST identity of each test protein against the training set, per input modality (rows) and ontology (columns). Marker area is proportional to the square root of the number of test proteins in the bin; points are not connected across bins with insufficient support.
