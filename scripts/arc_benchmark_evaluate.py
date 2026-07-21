#!/usr/bin/env python3
"""CAFA-style evaluation and plotting for standardized ARC predictions."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from arc_benchmark import ONTOLOGIES, load_label_npz, load_prediction, parse_obo


def align_prediction(path: Path, target_ids: list[str], target_terms: list[str]):
    ids, terms, source = load_prediction(path)
    if len(set(ids)) != len(ids) or len(set(terms)) != len(terms):
        raise ValueError(f"{path}: duplicate proteins or GO terms")
    id_index = {value: index for index, value in enumerate(ids)}
    term_index = {value: index for index, value in enumerate(terms)}
    aligned = np.zeros((len(target_ids), len(target_terms)), dtype=np.float32)
    mapped_ids = [protein for protein in target_ids if protein in id_index]
    mapped_terms = [term for term in target_terms if term in term_index]
    source_rows = np.asarray([id_index[value] for value in mapped_ids])
    source_cols = np.asarray([term_index[value] for value in mapped_terms])
    target_id_index = {value: index for index, value in enumerate(target_ids)}
    target_term_index = {value: index for index, value in enumerate(target_terms)}
    target_rows = np.asarray([target_id_index[value] for value in mapped_ids])
    target_cols = np.asarray([target_term_index[value] for value in mapped_terms])
    if len(source_rows) and len(source_cols):
        aligned[np.ix_(target_rows, target_cols)] = source[np.ix_(source_rows, source_cols)]
    return aligned, len(mapped_ids), len(mapped_terms)


def information_accretion(train_labels: np.ndarray, terms: list[str], obo_path: Path) -> np.ndarray:
    """Compute -log2 P(term | all represented parents) from training labels."""
    parents, _ = parse_obo(obo_path)
    term_index = {term: index for index, term in enumerate(terms)}
    values = np.zeros(len(terms), dtype=np.float64)
    counts = train_labels.sum(axis=0, dtype=np.float64)
    for index, term in enumerate(terms):
        represented = [term_index[parent] for parent in parents.get(term, ()) if parent in term_index]
        if represented:
            parent_mask = np.all(train_labels[:, represented] > 0, axis=1)
            denominator = float(parent_mask.sum())
        else:
            denominator = float(train_labels.shape[0])
        numerator = float(counts[index])
        if numerator > 0 and denominator > 0:
            probability = min(max(numerator / denominator, 1e-12), 1.0)
            values[index] = -math.log2(probability)
    return values


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    # A fixed grid makes every method and every bootstrap directly comparable.
    return np.linspace(0.0, 1.0, 101, dtype=np.float32)


def per_protein_curves(y_true: np.ndarray, scores: np.ndarray, ia: np.ndarray):
    thresholds = threshold_grid(scores)
    n, _ = y_true.shape
    precision_contrib = np.zeros((n, len(thresholds)), np.float64)
    recall_contrib = np.zeros_like(precision_contrib)
    has_prediction = np.zeros_like(precision_contrib)
    ru = np.zeros_like(precision_contrib)
    mi = np.zeros_like(precision_contrib)
    true_counts = y_true.sum(axis=1)
    for column, threshold in enumerate(thresholds):
        predicted = scores > threshold if threshold == 0 else scores >= threshold
        tp = np.logical_and(predicted, y_true > 0).sum(axis=1)
        pred_counts = predicted.sum(axis=1)
        covered = pred_counts > 0
        precision_contrib[covered, column] = tp[covered] / pred_counts[covered]
        has_prediction[covered, column] = 1.0
        nonempty_truth = true_counts > 0
        recall_contrib[nonempty_truth, column] = tp[nonempty_truth] / true_counts[nonempty_truth]
        ru[:, column] = (np.logical_and(y_true > 0, ~predicted) * ia).sum(axis=1)
        mi[:, column] = (np.logical_and(y_true == 0, predicted) * ia).sum(axis=1)
    return thresholds, precision_contrib, recall_contrib, has_prediction, ru, mi


def aggregate_curves(curves, weights: np.ndarray | None = None):
    thresholds, precision_c, recall_c, covered_c, ru_c, mi_c = curves
    if weights is None:
        precision_sum = precision_c.sum(axis=0, keepdims=True)
        recall_sum = recall_c.sum(axis=0, keepdims=True)
        covered_sum = covered_c.sum(axis=0, keepdims=True)
        ru_sum = ru_c.sum(axis=0, keepdims=True)
        mi_sum = mi_c.sum(axis=0, keepdims=True)
        sample_size = precision_c.shape[0]
    else:
        precision_sum = weights @ precision_c
        recall_sum = weights @ recall_c
        covered_sum = weights @ covered_c
        ru_sum = weights @ ru_c
        mi_sum = weights @ mi_c
        sample_size = weights.sum(axis=1, keepdims=True)
    precision = precision_sum / np.maximum(covered_sum, 1.0)
    recall = recall_sum / np.maximum(sample_size, 1.0)
    fscore = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    semantic = np.sqrt((ru_sum / sample_size) ** 2 + (mi_sum / sample_size) ** 2)
    coverage = covered_sum / sample_size
    best_f = np.nanmax(fscore, axis=1)
    best_f_index = np.nanargmax(fscore, axis=1)
    best_s = np.nanmin(semantic, axis=1)
    best_s_index = np.nanargmin(semantic, axis=1)
    return {
        "fmax": best_f,
        "fmax_threshold": thresholds[best_f_index],
        "smin": best_s,
        "smin_threshold": thresholds[best_s_index],
        "coverage_at_fmax": coverage[np.arange(len(best_f)), best_f_index],
    }


def scalar_metrics(y_true: np.ndarray, scores: np.ndarray, score_type: str):
    from sklearn.metrics import average_precision_score, roc_auc_score

    observed = y_true.sum(axis=0) > 0
    result = {}
    if score_type == "binary":
        result.update(micro_aupr=np.nan, macro_aupr=np.nan, micro_auroc=np.nan, macro_auroc=np.nan)
    else:
        result["micro_aupr"] = float(average_precision_score(y_true.ravel(), scores.ravel()))
        per_term = [average_precision_score(y_true[:, j], scores[:, j]) for j in np.where(observed)[0]]
        result["macro_aupr"] = float(np.mean(per_term)) if per_term else np.nan
        try:
            result["micro_auroc"] = float(roc_auc_score(y_true, scores, average="micro"))
        except ValueError:
            result["micro_auroc"] = np.nan
        auc_terms = []
        for j in np.where(observed)[0]:
            if np.unique(y_true[:, j]).size == 2:
                auc_terms.append(roc_auc_score(y_true[:, j], scores[:, j]))
        result["macro_auroc"] = float(np.mean(auc_terms)) if auc_terms else np.nan
    result["protein_coverage_any_score"] = float(np.mean(np.any(scores > 0, axis=1)))
    result["predicted_term_coverage"] = int(np.any(scores > 0, axis=0).sum())
    return result


def fixed_threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold if threshold > 0 else scores > 0
    true_mask = y_true > 0
    tp = np.logical_and(predicted, true_mask).sum(axis=1)
    pred_counts = predicted.sum(axis=1)
    true_counts = true_mask.sum(axis=1)
    covered = pred_counts > 0
    precision = float(np.mean(tp[covered] / pred_counts[covered])) if covered.any() else 0.0
    recall = float(np.mean(tp[true_counts > 0] / true_counts[true_counts > 0]))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "test_precision_at_validation_threshold": precision,
        "test_recall_at_validation_threshold": recall,
        "test_f1_at_validation_threshold": f1,
        "test_coverage_at_validation_threshold": float(covered.mean()),
        "mean_terms_per_protein_at_validation_threshold": float(pred_counts.mean()),
    }


def calibration_metrics(y_true: np.ndarray, scores: np.ndarray, kind: str, bins: int = 15) -> dict:
    if kind == "binary":
        return {"brier_score": np.nan, "expected_calibration_error": np.nan}
    truth = y_true.ravel().astype(np.float64)
    probabilities = scores.ravel().astype(np.float64)
    brier = float(np.mean((probabilities - truth) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            ece += selected.mean() * abs(probabilities[selected].mean() - truth[selected].mean())
    return {"brier_score": brier, "expected_calibration_error": float(ece)}


def score_type(path: Path) -> str:
    metadata = path.with_name(path.stem + ".metadata.json")
    if metadata.is_file():
        return str(json.loads(metadata.read_text()).get("score_type", "continuous"))
    return "continuous"


def discover_methods(workspace: Path) -> list[str]:
    methods = []
    for directory in sorted((workspace / "predictions").iterdir()):
        if not directory.is_dir() or directory.name.endswith("_valid") or "_seed_" in directory.name:
            continue
        if all((directory / f"{short}.npz").is_file() for short in ONTOLOGIES):
            methods.append(directory.name)
    return methods


def evaluate(args) -> None:
    workspace = args.workspace.resolve()
    manifest = json.loads((workspace / "benchmark_manifest.json").read_text())
    obo_path = Path(manifest["data_root"]) / "go-basic.obo"
    methods = discover_methods(workspace)
    required = [value.strip() for value in args.require_methods.split(",") if value.strip()]
    missing = sorted(set(required) - set(methods))
    if missing:
        raise SystemExit("Required standardized predictions are missing: " + ", ".join(missing))
    if not methods:
        raise SystemExit("No complete standardized prediction sets were found")

    result_dir = workspace / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, bootstrap_rows, delta_rows, audit_rows = [], [], [], []
    rng = np.random.default_rng(args.bootstrap_seed)
    for short, ontology in ONTOLOGIES.items():
        train_ids, train_terms, train_labels = load_label_npz(workspace, short, "train")
        valid_ids, valid_terms, valid_true = load_label_npz(workspace, short, "valid")
        test_ids, terms, y_true = load_label_npz(workspace, short, "test")
        if train_terms != terms or valid_terms != terms:
            raise ValueError(f"{short}: train/valid/test term universes differ")
        ia = information_accretion(train_labels, terms, obo_path)
        n = len(test_ids)
        draws = rng.integers(0, n, size=(args.bootstraps, n), endpoint=False)
        weights = np.zeros((args.bootstraps, n), dtype=np.float32)
        row_indices = np.repeat(np.arange(args.bootstraps), n)
        np.add.at(weights, (row_indices, draws.ravel()), 1.0)
        method_bootstrap = {}
        for method in methods:
            path = workspace / "predictions" / method / f"{short}.npz"
            scores, mapped_ids, mapped_terms = None, None, None
            scores, mapped_ids, mapped_terms = (*align_prediction(path, test_ids, terms),)
            kind = score_type(path)
            validation_path = workspace / "predictions" / f"{method}_valid" / f"{short}.npz"
            if not validation_path.is_file():
                raise FileNotFoundError(
                    f"Validation predictions are required for fixed-threshold reporting: {validation_path}"
                )
            validation_scores, _, _ = align_prediction(validation_path, valid_ids, terms)
            validation_point = aggregate_curves(
                per_protein_curves(valid_true, validation_scores, ia)
            )
            validation_threshold = float(validation_point["fmax_threshold"][0])
            curves = per_protein_curves(y_true, scores, ia)
            point = aggregate_curves(curves)
            boot = aggregate_curves(curves, weights)
            method_bootstrap[method] = boot
            scalar = scalar_metrics(y_true, scores, kind)
            fixed = fixed_threshold_metrics(y_true, scores, validation_threshold)
            calibration = calibration_metrics(y_true, scores, kind)
            row = {
                "method": method, "ontology": ontology, "ontology_short": short,
                "score_type": kind, "test_proteins": n, "test_terms": len(terms),
                "mapped_proteins": mapped_ids, "mapped_terms": mapped_terms,
                "cafa_fmax": float(point["fmax"][0]),
                "cafa_fmax_threshold": float(point["fmax_threshold"][0]),
                "cafa_smin": float(point["smin"][0]),
                "cafa_smin_threshold": float(point["smin_threshold"][0]),
                "coverage_at_fmax": float(point["coverage_at_fmax"][0]),
                "validation_threshold": validation_threshold,
                "validation_cafa_fmax": float(validation_point["fmax"][0]),
                **fixed,
                **scalar,
                **calibration,
            }
            for metric in ("fmax", "smin"):
                values = boot[metric]
                row[f"cafa_{metric}_ci_low"] = float(np.quantile(values, 0.025))
                row[f"cafa_{metric}_ci_high"] = float(np.quantile(values, 0.975))
            summary_rows.append(row)
            audit_rows.append({key: row[key] for key in (
                "method", "ontology", "score_type", "test_proteins", "test_terms",
                "mapped_proteins", "mapped_terms", "protein_coverage_any_score", "predicted_term_coverage")})
            for index in range(args.bootstraps):
                bootstrap_rows.append({"bootstrap": index, "method": method, "ontology": ontology,
                                       "cafa_fmax": float(boot["fmax"][index]),
                                       "cafa_smin": float(boot["smin"][index])})
        if "deepgreengo" in method_bootstrap:
            reference = method_bootstrap["deepgreengo"]
            for method, boot in method_bootstrap.items():
                if method == "deepgreengo":
                    continue
                for metric in ("fmax", "smin"):
                    delta = reference[metric] - boot[metric]
                    if metric == "smin":
                        delta = -delta  # positive always means DeepGreenGO is better
                    delta_rows.append({
                        "ontology": ontology, "competitor": method, "metric": f"cafa_{metric}",
                        "deepgreengo_advantage_mean": float(np.mean(delta)),
                        "ci_low": float(np.quantile(delta, 0.025)),
                        "ci_high": float(np.quantile(delta, 0.975)),
                        "probability_deepgreengo_better": float(np.mean(delta > 0)),
                    })
    write_csv(result_dir / "benchmark_metrics.csv", summary_rows)
    write_csv(result_dir / "bootstrap_metrics.csv", bootstrap_rows)
    write_csv(result_dir / "paired_differences_vs_deepgreengo.csv", delta_rows)
    write_csv(result_dir / "coverage_and_mapping_audit.csv", audit_rows)
    (result_dir / "evaluation_manifest.json").write_text(json.dumps({
        "methods": methods, "bootstraps": args.bootstraps,
        "bootstrap_seed": args.bootstrap_seed,
        "notes": [
            "CAFA Fmax is protein-centric; precision averages over covered targets and recall over all targets.",
            "Smin uses training-derived conditional information accretion and is minimized over the same threshold grid.",
            "Bootstrap resampling is paired by protein across methods.",
            "AUPR/AUROC and calibration are not reported for binary annotation pipelines.",
            "Fixed-threshold test precision, recall and F1 use thresholds selected only on validation proteins.",
            "The split is nominal 30% identity/80% coverage and is not described as leakage-free.",
        ],
    }, indent=2) + "\n")
    print(f"Evaluated {len(methods)} methods; results: {result_dir}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(args) -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    workspace = args.workspace.resolve()
    results = pd.read_csv(workspace / "results" / "benchmark_metrics.csv")
    out_dir = workspace / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    preferred = [
        "deepgreengo", "dpfunc", "transfun", "deepgose", "deepgoplus",
        "deepfri_structure", "deepfri_sequence", "gomap", "hayai",
        "eggnog_mapper", "interproscan", "foldseek", "foldseek_max",
        "blast", "blast_max", "diamond", "diamond_max", "naive",
    ]
    methods = [method for method in preferred if method in set(results.method)]
    methods += sorted(set(results.method) - set(methods))
    labels = {"deepgreengo": "DeepGreenGO", "deepfri_structure": "DeepFRI (structure)",
              "deepfri_sequence": "DeepFRI (sequence)", "deepgose": "DeepGO-SE",
              "deepgoplus": "DeepGOPlus", "eggnog_mapper": "eggNOG-mapper",
              "interproscan": "InterProScan", "foldseek_max": "Foldseek (max)",
              "blast_max": "BLAST (max)", "diamond_max": "DIAMOND (max)"}
    colors = {method: plt.cm.tab20(index % 20) for index, method in enumerate(methods)}
    colors["deepgreengo"] = "#9C1C3A"

    fig, axes = plt.subplots(3, 3, figsize=(20, 14), constrained_layout=True)
    for row_index, (short, ontology) in enumerate(ONTOLOGIES.items()):
        subset = results[results.ontology == ontology].set_index("method").reindex(methods)
        x = np.arange(len(methods))
        for column, (metric, title, lower) in enumerate((
            ("cafa_fmax", "CAFA Fmax", "higher is better"),
            ("micro_aupr", "Micro AUPR", "higher is better"),
            ("cafa_smin", "CAFA Smin", "lower is better"),
        )):
            ax = axes[row_index, column]
            values = subset[metric].to_numpy(float)
            ax.bar(x, values, color=[colors[m] for m in methods], edgecolor="white")
            if metric in ("cafa_fmax", "cafa_smin"):
                low = subset[metric + "_ci_low"].to_numpy(float)
                high = subset[metric + "_ci_high"].to_numpy(float)
                ax.errorbar(x, values, yerr=np.vstack([values - low, high - values]),
                            fmt="none", ecolor="#333333", capsize=2, linewidth=0.8)
            ax.set_title(f"{ontology.replace('_', ' ').title()} — {title}")
            ax.set_ylabel(lower)
            ax.set_xticks(x, [labels.get(m, m) for m in methods], rotation=55, ha="right")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"01_cafa_metrics.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    for ax, (short, ontology) in zip(axes, ONTOLOGIES.items()):
        subset = results[results.ontology == ontology].set_index("method").reindex(methods)
        x = np.arange(len(methods))
        ax.bar(x, subset.protein_coverage_any_score, color=[colors[m] for m in methods])
        ax.set_title(ontology.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of 754 test proteins")
        ax.set_xticks(x, [labels.get(m, m) for m in methods], rotation=55, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"02_prediction_coverage.{suffix}", dpi=300)
    plt.close(fig)

