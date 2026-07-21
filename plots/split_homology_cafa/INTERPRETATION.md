# Homology-split metric interpretation

All six held-out test evaluations (30%, 40%, 50%, 70%, 90%, and 95%) are now
included. Each split contains three ontologies and five fixed training seeds.

- `test_micro_f1_at_validation_threshold` is the primary classification result:
  its decision threshold was selected on validation data and frozen for test.
- `test_micro_fmax` and `test_macro_fmax` sweep thresholds on the held-out test
  predictions. They are descriptive CAFA metrics rather than frozen-threshold
  estimates.
- `test_smin` is lower-is-better within a split. Information content is derived
  independently from each split's training labels, so raw Smin magnitudes are
  not strictly comparable when train/test membership changes.
- Seed points are the five fixed seeds. Error bars are 95% t intervals across
  those five runs; they represent training-seed variation, not uncertainty from
  rebuilding the split.
- Nominal cluster thresholds do not imply that every test sequence has that
  identity to a training sequence. Figure 04 uses measured BLAST similarity.
