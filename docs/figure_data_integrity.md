# Figure data-integrity audit

Resolved before figure aesthetics, because each item changes what a figure is
allowed to claim. Every number below was reproduced from the split labels in
`preprocessing/data_arc_rebuild_2026_07_14/datasets/threshold_30/` and the
archived metric tables; none are estimates.

Status key: **[confirmed]** reproduced numerically here · **[code]** established
by reading the implementation · **[blocked]** needs an ARC re-run to resolve.

---

## 1. Macro-AUPRC is computed with a biased estimator (affects every model)

**[code]** `get_auprc(..., average="macro")` in `src/evals.py` computes
`auc(recall, precision)` — trapezoidal integration of the PR curve — instead of
`average_precision_score`. Trapezoidal integration of a PR curve interpolates
between operating points, which is not valid for PR space and is optimistic.

For a predictor that emits the **same score for every protein**, the PR curve
collapses to two points, `(recall=1, precision=p)` and `(recall=0, precision=1)`,
whose trapezoid area is exactly **(1 + p) / 2** — a floor near 0.5 regardless of
how uninformative the model is.

**[confirmed]** Predicted vs. observed, MLP / structure-only:

| Ontology | mean prevalence p | predicted (1+p)/2 | observed Macro-AUPRC | correct AP |
|---|---|---|---|---|
| molecular_function | 0.0292 | 0.5146 | **0.514622** | 0.0292 |
| cellular_component | 0.0834 | 0.5417 | **0.541695** | 0.0834 |
| biological_process | 0.0280 | 0.5140 | 0.487260 | 0.0280 |

MF and CC match to four decimal places. BP is near-but-not-exact because its
MLP/structure-only output is not perfectly constant (its Macro-AUROC is 0.49927
rather than exactly 0.50000, i.e. float32 pooling noise breaks ties).

**Consequence.** The MLP structure-only "spike" is not a macro-averaging
artifact over a few low-support terms — it is a metric-implementation artifact,
and it inflates **all** Macro-AUPRC values in the archived tables, not only the
degenerate cells. `src/evals.py` has been fixed to use `average_precision_score`.
**[blocked]** Existing Macro-AUPRC numbers were produced by the old estimator and
must be regenerated on ARC (no checkpoints exist locally: 0 `best_checkpoint.pt`
found). Until then every figure labels Macro-AUPRC as provisional.

## 2. MLP + structure-only is a constant predictor, not a weak model

**[code]** `transform(batch, "struct_only")` sets `batch.x = torch.zeros_like(batch.x)`
(`src/train_arc_ablation.py`), and `MLPModel.forward` ignores `edge_index`
entirely (`src/model.py:167`, comment: "Ignore edge_index for MLP"). With zeroed
features and no edge use, the pooled representation is identical for every
protein, so the output is the bias vector alone.

**[confirmed]** Macro-AUROC = **0.50000** exactly (MF and CC), 0.49927 (BP) —
chance, i.e. zero per-term discrimination.

**Correction to a prior reading of these figures:** MLP/structure-only is *not*
"near 0.70" on Macro-AUROC. The ~0.70 value is **Micro**-AUROC (MF 0.696,
BP 0.709, CC 0.636). Micro-AUROC pools every (protein, term) pair, so the
per-term bias ordering alone produces apparent signal: frequent terms get higher
bias and are likelier to be positive. That is a term-frequency prior — the naive
CAFA baseline — not structural capability. Report it as such, or the split
between Micro-AUROC 0.70 and Macro-AUROC 0.50 will read as a contradiction.

## 3. Structure-only Smin: two opposite degenerate behaviours, both real

**[confirmed]** MLP, GAT and GCN under structure-only produce Smin *exactly*
equal to the "predict all-negative" value (total IC burden of the test set):

| Ontology | Smin if all-negative | observed (MLP / GAT / GCN) |
|---|---|---|
| molecular_function | 26.784569 | 26.784569 / 26.784569 / 26.784569 |
| biological_process | 101.055691 | 101.038864 / 101.055690 / 100.858367 |
| cellular_component | 29.926645 | 29.926645 / 29.763933 / 29.926645 |

These models emit sub-threshold probabilities everywhere, so the best threshold
predicts nothing and Smin reduces to the remaining-uncertainty term.

Hybrid and Hybrid_JK show the mirror-image failure: Smin 1580–4670 (vs <105 for
every other cell). This is **not** a units or normalisation bug. `get_smin`
scans thresholds `np.arange(0.01, 1.0, 0.01)`; a saturated constant output above
0.99 is predicted positive at *every* scanned threshold, so the misinformation
term never falls and Smin explodes. Genuine pathological behaviour, but the
magnitude is partly an artifact of a threshold grid that stops at 0.99 — state
the grid bound wherever these numbers appear.

## 4. Bin cells with Macro-AUROC exactly 0.0 — archived evaluator artifact + small N

**[confirmed]** 58 rows have Macro-AUROC exactly 0.0. Macro-AUPRC has **zero**
such rows. Distribution:

| Ontology | bin | N | rows | models affected |
|---|---|---|---|---|
| molecular_function | homology 40–60% | **5** | 49 | all five |
| molecular_function | homology ≥60% | 31 | 3 | GCN |
| cellular_component | homology <30% | 26 | 4 | GCN, Hybrid, Hybrid_JK |
| biological_process | homology 40–60% | **5** | 1 | Hybrid_JK |
| biological_process | ic 2–4 bits | 10 | 1 | Hybrid |

The flagged GAT/GCN, MF, 40–60% case is **confirmed as a genuine small-N
artifact compounded by an evaluator bug**. Ground truth for that bin: n=5
proteins, 3 terms with any positive, and only **1 term two-class** — the other 2
are positive in all 5 proteins, so AUROC is undefined for them. The archived
evaluator returned 0.0 for the whole macro average when it hit an undefined
term; the current `get_auroc` skips one-class terms and returns NaN if none
remain. These 0.0 values are therefore **not model scores** and are masked to
NaN rather than plotted. **[blocked]** Real values require the ARC re-run.

## 5. Empty bins are genuinely empty, not silently dropped

**[confirmed]** Homology bin sizes are identical across all three ontologies —
612 / 26 / 80 / 5 / 31 for no_hit / <30% / 30–40% / 40–60% / ≥60% — summing to
754, the full test set, as expected since homology does not depend on ontology.
IC bins are ontology-specific and also sum to 754 (MF 219+100+41+43+351;
BP 206+0+10+16+522; CC 231+22+71+89+341).

Exactly one (ontology, bin_type, bin) combination is absent from the metrics
table: **biological_process / ic / <2_bits**, which has 0 test proteins by
construction. It is a true empty bin and renders as an explicit "no data" gap.

## 6. Per-seed values are available for every bin cell

**[confirmed]** Every (ontology, model, modality, bin_type, bin) cell has
exactly **5** rows (min = max = 5) across 225 distinct checkpoints, so mean ± SD
across seeds is computable everywhere. Error bars in the bin figures are real
seed-level dispersion, not placeholders.

---

## 7. BLAST benchmark rows were a stale artifact — root-caused and REPAIRED

**[confirmed]** In `arc_benchmark/.../results/benchmark_metrics.csv`, BLAST and
BLAST-max record CAFA Fmax **0.000** with protein coverage **0.000** and 0
predicted terms in all three ontologies, while DIAMOND — the same
homology-transfer method — scores 0.13–0.22.

That zero is not reproducible from the pipeline's own inputs:

| Check | BLAST | DIAMOND |
|---|---|---|
| raw hits on disk | 7,108 alignments | 3,822 |
| test queries resolving to label IDs | 195 | 154 |
| train targets resolving | 414 | 214 |
| hits passing both coverage filters | 3,616 | 3,023 |
| **transfer recomputed from stored hits** | **2,619 nonzero / 172 proteins** | 2,306 / 124 |
| **stored `predictions/*/mf.npz`** | **0 nonzero** | 2,306 nonzero |

DIAMOND's recomputation matches its stored array exactly (2,306 = 2,306),
which shows the transfer code is correct. `canonicalize_similarity_id` already
handles BLAST's `pdb|7FHK|A` mangling, and the ID overlap confirms it works.
The only inconsistent artifact is BLAST's stored prediction array, which is
empty despite its hit file being the largest of the three. The most probable
cause is that the hit file was empty when the prediction stage ran and the
`.done` marker then suppressed recomputation on resume.

**[confirmed] Root cause.** `arc_benchmark_methods.py` guarded each search with
`if not hits.is_file()`. blastp creates its `-out` file the instant it starts,
so a run killed by the Slurm wall clock (the benchmark requests 23 h 55 min and
is resumable) leaves a zero-byte or truncated `blast_hits.tsv`. On resume that
file is indistinguishable from a finished search: blastp is skipped, the
transfer reads nothing, all-zero predictions are written, and the stage `.done`
marker locks the empty result in permanently. DIAMOND and Foldseek carried the
identical guard and were simply never interrupted.

**Fix.** `run_search_once` now writes to a `.partial` path and `os.replace`s it
into position only after the process exits successfully, so the final filename
can never name an incomplete search; an existing zero-byte file is discarded
and re-run, and an empty result raises rather than being recorded. Applied to
all three searches.

**[confirmed] Repaired.** The transfer and evaluation are pure NumPy, so the
repair ran locally from the intact hit file — no ARC time and no re-search:

```
python scripts/run_arc_benchmark.py sequence-baselines --workspace <ws> --methods blast
python scripts/run_arc_benchmark.py evaluate --workspace <ws> --bootstraps 1000 --bootstrap-seed 20260720
```

| CAFA Fmax | MF | BP | CC |
|---|---|---|---|
| BLAST before | 0.0000 | 0.0000 | 0.0000 |
| **BLAST after** | **0.2777** | **0.1352** | **0.1158** |
| BLAST-max after | 0.2830 | 0.1338 | 0.1948 |
| DIAMOND (for reference) | 0.2192 | 0.1296 | 0.0752 |

Coverage rose from 0.000 to 0.19–0.23, and all 3,000 BLAST bootstrap replicates
are now nonzero. BLAST now exceeds DIAMOND on all three ontologies, which is the
expected ordering for a more sensitive search and is an independent check that
the repair is correct rather than merely non-zero.

Re-evaluation also required a portability fix: the manifest records an absolute
ARC `data_root`, so `go-basic.obo` could not be found on any other host.
`resolve_data_root` now falls back to the same directory name under the local
checkout, and `evaluate` accepts `--data-root`.

**Separately worth noting [confirmed]:** the naive frequency prior *beats*
DeepGreenGO on cellular component (0.3690 vs 0.3461 CAFA Fmax) and is close on
biological process (0.1670 vs 0.2171). Whatever the BLAST re-run shows, that
comparison belongs in the results text rather than only in a figure.

---

## What this means for the manuscript

1. Do not report Macro-AUPRC from the archived tables. Re-run on ARC with the
   corrected `get_auprc` first.
2. Do not describe MLP/structure-only as a weak-but-real model. It is a constant
   predictor and belongs in the text as a null control.
3. The "AUROC/AUPRC favour MLP/GCN" reading is partly an artifact of items 1–2.
   After the re-run, re-check whether the metric-family split survives; if it
   does not, the framing in the results section changes.
4. The homology 40–60% bin (n=5) cannot support any per-model claim. Keep it in
   figures as an explicitly flagged low-support point, or drop the bin.

---

## 8. Reviewer-requested evidence: what exists and what does not

Mapping the BMC Bioinformatics reviewer comments onto available data.

**R1.3 / R2.3 — redundancy and data leakage. [confirmed, figure available]**
`figure_leakage_residual_identity` quantifies it: 142 of 754 test proteins have
any BLAST hit to training, 116 (15.4%) retain >=30% identity, and **31 (4.1%)
sit at or above 60%** — visibly a spike near 100% identity. The split is
homology-controlled, not leakage-free, and must not be described as the latter.
Homology-binned performance is reported separately so the low-identity result
can be read independently of the residual.

**R1.10 — micro- and macro-AUROC reported as identical. [confirmed: not a
current bug]** Across all 225 ablation runs, **0 rows** have numerically equal
micro and macro AUROC; the mean gap is 0.10 (BP), 0.12 (CC), 0.13 (MF). The
identical values in the submitted Table 2 therefore came from the older
evaluation, not from this code. Regenerate Table 2 from current results.

**R2.6 — loss-function ablation (BCE vs focal). [blocked — does not exist]**
No controlled loss ablation exists on the locked split. The current ARC
ablation varies architecture and input modality only; the archived
`old run/runs_ablation_loss` was produced on the superseded pre-homology split
and is not comparable. Loss *was* one of nine hyperparameters in the random
search, so `figure_loss_bce_vs_focal_search` reports that evidence
(19 BCE vs 21 focal trials per ontology) — but it is **observational and
confounded**: those trials also differ in learning rate, dropout, hidden
dimension and batch size, so the gap cannot be attributed to the loss. Focal
leads on CC (0.3445 vs 0.3186) and is indistinguishable on MF and BP.
Answering the reviewer properly needs an ARC run holding every other
hyperparameter at the selected configuration and varying only the loss,
5 seeds x 3 ontologies.

**R2.5 — PR curves inconsistent with Table 2. [not yet built]** The benchmark
prediction arrays are on disk, so per-method PR curves are computable without
ARC; not generated in this pass.

**R1.6 / R2.8 — pLDDT filtering of AlphaFold structures. [n/a for this split]**
The locked benchmark uses 754 experimental PDB chains, not AlphaFold models, so
the concern applies to the rice prediction stage rather than to training or
test. That distinction should be stated explicitly in the response.
