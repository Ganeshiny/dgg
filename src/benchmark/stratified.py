#!/usr/bin/env python3
"""Stratified benchmark evaluation: homology bins, IC bins, GO depth, IC-weighted PR.

Computes the DPFunc-style analyses (their Fig. 1a/b/c/d/e/f) for every method
in an arc_benchmark workspace, directly from the stored per-protein score
matrices in predictions/<method>/<ontology>.npz.

Three stratification axes, all defined WITHOUT reference to any model's
predictions so no method can be favoured by the binning itself:

  homology  max BLAST identity of each test protein against the training set
            -> "does performance survive when the nearest training relative
               is remote?"  (protein-level stratification)
  ic        information content of each GO term, -log2(p) from TRAINING label
            frequency -> "does performance survive on rare, informative
            terms?"  (term-level stratification)
  depth     shortest path from the ontology root through is_a/part_of
            -> "does performance survive on specific, deep terms?"
            (term-level stratification)

Uncertainty: DeepGreenGO has five independent training seeds on disk, so it
carries a real mean +/- s.d. across seeds. The similarity baselines are
deterministic given the split -- BLAST/DIAMOND/Foldseek/naive produce one
answer, not a distribution -- so they are single points with no error bar.
That asymmetry is reported rather than papered over with a fabricated
interval.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from .core import ONTOLOGIES, ROOT_TERMS, load_label_npz, parse_obo

# Protein-level homology bins, ordered by increasing similarity to training.
HOMOLOGY_BINS = ["no hit", "<30%", "30-40%", "40-60%", ">=60%"]

# Term-level IC bins, ordered by increasing rarity/informativeness.
IC_BINS = ["<2 bits", "2-4 bits", "4-6 bits", ">=6 bits"]

# Depth bins. DPFunc uses per-ontology deep-term cutoffs (MF>8, CC>6, BP>8);
# these shallow/mid/deep bands keep the same idea while staying interpretable
# when a bin would otherwise be empty.
DEPTH_BINS = ["1-3", "4-6", "7-9", ">=10"]

def max_identity_by_protein(homology_tsv: Path, protein_ids) -> dict[str, float]:
    """Best percent identity of each test protein against any training protein."""
    best = {str(pid): 0.0 for pid in protein_ids}
    if not homology_tsv.is_file():
        raise FileNotFoundError(f"Homology table not found: {homology_tsv}")
    with homology_tsv.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            query = fields[0]
            if query not in best:
                continue
            try:
                best[query] = max(best[query], float(fields[2]))
            except ValueError:
                continue
    return best


def homology_bin_of(identity: float) -> str:
    if identity <= 0:
        return "no hit"
    if identity < 30:
        return "<30%"
    if identity < 40:
        return "30-40%"
    if identity < 60:
        return "40-60%"
    return ">=60%"


def information_content(train_labels: np.ndarray) -> np.ndarray:
    """IC per term as -log2(frequency) measured on TRAINING labels only.

    Using training frequency (not test) keeps the term stratification
    independent of the evaluation set, so a method cannot look better simply
    because the test set happens to be skewed.
    """
    n_train = max(int(train_labels.shape[0]), 1)
    counts = np.asarray(train_labels.sum(axis=0), dtype=np.float64)
    frequency = np.clip(counts / n_train, 1e-12, 1.0)
    ic = -np.log2(frequency)
    # A term never seen in training has undefined IC, not infinite IC.
    ic[counts == 0] = np.nan
    return ic


def ic_bin_of(value: float) -> str | None:
    if not np.isfinite(value):
        return None
    if value < 2:
        return "<2 bits"
    if value < 4:
        return "2-4 bits"
    if value < 6:
        return "4-6 bits"
    return ">=6 bits"


def term_depths(terms, parents: dict[str, set[str]], root: str) -> dict[str, int]:
    """Shortest is_a/part_of distance from the ontology root, by BFS.

    Shortest path is the conventional GO "depth"; the DAG admits several paths
    to the root and the minimum is the standard, reproducible choice.
    """
    children: dict[str, set[str]] = {}
    for child, child_parents in parents.items():
        for parent in child_parents:
            children.setdefault(parent, set()).add(child)

    depth = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in children.get(node, ()):
            if child not in depth:
                depth[child] = depth[node] + 1
                queue.append(child)
    return {str(term): depth[str(term)] for term in map(str, terms) if str(term) in depth}


def depth_bin_of(depth: int) -> str:
    if depth <= 3:
        return "1-3"
    if depth <= 6:
        return "4-6"
    if depth <= 9:
        return "7-9"
    return ">=10"


def protein_centric_fmax(y_true: np.ndarray, scores: np.ndarray,
                         thresholds: np.ndarray | None = None) -> float:
    """CAFA protein-centric Fmax over a fixed threshold grid.

    Precision is averaged only over proteins with at least one prediction at
    the threshold (the CAFA convention); recall is averaged over all proteins
    that have at least one true annotation.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    has_truth = y_true.sum(axis=1) > 0
    if not has_truth.any():
        return float("nan")
    truth = y_true[has_truth].astype(bool)
    pred_scores = scores[has_truth]
    truth_counts = truth.sum(axis=1)

    best = 0.0
    for threshold in thresholds:
        predicted = pred_scores >= threshold
        predicted_counts = predicted.sum(axis=1)
        intersection = np.logical_and(predicted, truth).sum(axis=1)
        covered = predicted_counts > 0
        if not covered.any():
            continue
        precision = float(np.mean(intersection[covered] / predicted_counts[covered]))
        recall = float(np.mean(intersection / truth_counts))
        if precision + recall > 0:
            best = max(best, 2 * precision * recall / (precision + recall))
    return float(best)


def term_centric_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mean average precision over terms that have at least one positive."""
    from sklearn.metrics import average_precision_score

    values = []
    for index in range(y_true.shape[1]):
        column = y_true[:, index]
        if column.sum() == 0:
            continue
        values.append(average_precision_score(column, scores[:, index]))
    return float(np.mean(values)) if values else float("nan")


def ic_weighted_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    ic: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """IC-weighted precision/recall across a threshold sweep.

    Each (protein, term) decision is weighted by the term's information
    content, so correctly recovering a rare, specific term counts for more
    than recovering a near-universal one. This is the curve underlying
    DPFunc's Fig. 1b/d/e.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    weights = np.where(np.isfinite(ic), ic, 0.0)
    truth = y_true.astype(bool)
    weighted_truth = (truth * weights).sum()
    if weighted_truth <= 0:
        empty = np.full(len(thresholds), np.nan)
        return empty, empty

    precisions, recalls = [], []
    for threshold in thresholds:
        predicted = scores >= threshold
        true_positive = float((np.logical_and(predicted, truth) * weights).sum())
        predicted_weight = float((predicted * weights).sum())
        precisions.append(true_positive / predicted_weight if predicted_weight > 0 else np.nan)
        recalls.append(true_positive / weighted_truth)
    return np.asarray(precisions), np.asarray(recalls)


def load_scores(workspace: Path, method: str, ontology_short: str,
                protein_ids, go_terms) -> np.ndarray | None:
    """Load a method's score matrix, aligned to the evaluation label axes."""
    path = workspace / "predictions" / method / f"{ontology_short}.npz"
    if not path.is_file():
        return None
    payload = np.load(path, allow_pickle=True)
    scores = np.asarray(payload["scores"], dtype=np.float32)
    stored_proteins = [str(value) for value in payload["protein_ids"]]
    stored_terms = [str(value) for value in payload["go_terms"]]
    if stored_proteins == [str(v) for v in protein_ids] and stored_terms == [str(v) for v in go_terms]:
        return scores
    # Realign rather than assume identical ordering.
    protein_index = {name: i for i, name in enumerate(stored_proteins)}
    term_index = {name: i for i, name in enumerate(stored_terms)}
    rows = [protein_index.get(str(name)) for name in protein_ids]
    columns = [term_index.get(str(name)) for name in go_terms]
    if any(r is None for r in rows) or any(c is None for c in columns):
        raise ValueError(f"{path} does not cover the evaluation label axes")
    return scores[np.ix_(rows, columns)]


def seed_score_matrices(workspace: Path, ontology_short: str,
                        protein_ids, go_terms) -> list[np.ndarray]:
    """Per-seed DeepGreenGO scores, used for genuine across-seed variability."""
    directories = sorted(
        (workspace / "predictions").glob("deepgreengo_seed_*_test")
    )
    matrices = []
    for directory in directories:
        path = directory / f"{ontology_short}.npz"
        if not path.is_file():
            continue
        payload = np.load(path, allow_pickle=True)
        scores = np.asarray(payload["scores"], dtype=np.float32)
        stored_proteins = [str(value) for value in payload["protein_ids"]]
        stored_terms = [str(value) for value in payload["go_terms"]]
        protein_index = {name: i for i, name in enumerate(stored_proteins)}
        term_index = {name: i for i, name in enumerate(stored_terms)}
        rows = [protein_index[str(name)] for name in protein_ids]
        columns = [term_index[str(name)] for name in go_terms]
        matrices.append(scores[np.ix_(rows, columns)])
    return matrices
