# Homology-split CAFA plots

Available held-out test splits: 30%, 40%, 50%, 70%, 90%, 95%

Missing held-out test splits: none

The primary leakage-safe classification metric is micro F1 evaluated at the
validation-selected threshold. Fmax values sweep thresholds on the test set and
are therefore labelled descriptive. Smin is lower-is-better. Points represent
the five fixed seeds; intervals are 95% t intervals over those seeds.

The nominal split threshold is a clustering setting, not the measured identity
of every test protein to training. Figure 04 uses the independent BLAST audit.
The split memberships and GO-label composition change across thresholds, so
cross-split trends are descriptive rather than a controlled causal effect of
homology.

Regenerate from the repository root:

    python3 src/plot_split_cafa_metrics.py

Use `--require-complete` after every threshold test evaluation has finished.
