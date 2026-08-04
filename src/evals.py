import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, average_precision_score
import math

def get_micro_fmax(y_true, y_pred_probs):
    """Compute Micro-averaged Fmax."""
    precisions, recalls, thresholds = precision_recall_curve(y_true.flatten(), y_pred_probs.flatten())
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    return np.max(f1_scores)

def get_macro_fmax(y_true, y_pred_probs):
    """Compute Macro-averaged Fmax (mean of per-class Fmax)."""
    fmax_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) == 0:
            continue
        precisions, recalls, thresholds = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        fmax_list.append(np.max(f1_scores))
    if len(fmax_list) == 0:
        return 0.0
    return np.mean(fmax_list)

def get_auroc(y_true, y_pred_probs, average="macro"):
    """Compute AUROC without turning undefined one-class terms into zero.

    A bin can contain only positives (or only negatives) for a GO term. In
    that case AUROC is undefined for that term; returning 0.0 for the whole
    macro average makes a small-support bin look like a catastrophic model
    failure. Macro AUROC therefore averages only terms with both classes and
    returns NaN when none are defined. Micro AUROC remains undefined when the
    flattened bin contains one class.
    """
    import warnings

    y_true = np.asarray(y_true)
    y_pred_probs = np.asarray(y_pred_probs)
    if average == "macro":
        scores = []
        for index in range(y_true.shape[1]):
            column = y_true[:, index]
            if np.unique(column).size < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores.append(roc_auc_score(column, y_pred_probs[:, index]))
        return float(np.mean(scores)) if scores else float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_true, y_pred_probs, average=average))

def get_auprc(y_true, y_pred_probs, average="macro"):
    """Area under the precision-recall curve, via average precision.

    Uses average_precision_score rather than auc(recall, precision).
    Trapezoidal integration of a PR curve linearly interpolates between
    operating points, which is not valid in PR space and is optimistically
    biased. The bias is unbounded for tied scores: a predictor emitting an
    identical score for every example collapses the curve to the two points
    (recall=1, precision=p) and (recall=0, precision=1), whose trapezoid area
    is (1 + p) / 2 — roughly 0.5 no matter how uninformative the model is.

    That is not hypothetical here. MLP under the structure-only ablation zeroes
    its input features and ignores edge_index, so it emits a constant score;
    the old estimator scored it 0.5146 (MF) and 0.5417 (CC), matching (1+p)/2
    to four decimals, while its true average precision is 0.0292 and 0.0834.
    Average precision uses the step-wise sum that PR curves require and returns
    exactly p for a constant predictor.

    See docs/figure_data_integrity.md. Metrics produced before this fix are not
    comparable with metrics produced after it; regenerate rather than mix.
    """
    if average == "micro":
        return float(average_precision_score(y_true.ravel(), y_pred_probs.ravel()))
    if average == "macro":
        scores = []
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i]) == 0:
                # No positives: average precision is undefined, not zero.
                continue
            scores.append(average_precision_score(y_true[:, i], y_pred_probs[:, i]))
        return float(np.mean(scores)) if scores else float("nan")
    return float("nan")

def compute_ic(y_train):
    """Training-frequency IC with a one-count floor for unseen GO terms.

    For terms observed in training this is exactly -log2(n_j / N_train).
    A vocabulary term with n_j = 0 would otherwise have infinite IC and make
    Smin undefined. Such terms receive the maximum finite training-derived IC,
    -log2(1 / N_train), without changing the weight of any observed term.
    """
    labels = np.asarray(y_train)
    if labels.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional training-label matrix, got {labels.shape}"
        )
    n_train = int(labels.shape[0])
    if n_train <= 0:
        raise ValueError("Cannot compute information content from an empty training set")
    counts = np.sum(labels > 0, axis=0, dtype=np.int64)
    effective_counts = np.maximum(counts, 1)
    return -np.log2(effective_counts / n_train)

def get_smin(y_true, y_pred_probs, ic):
    """
    Compute Smin (CAFA metric) vectorized.
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    s_min = float('inf')
    N = y_true.shape[0]
    
    # Vectorized computation of Smin
    for t in thresholds:
        preds = (y_pred_probs >= t).astype(int)
        
        # False negatives (True=1, Pred=0)
        fn_mask = (y_true == 1) & (preds == 0)
        ru = np.sum(fn_mask * ic) / N
        
        # False positives (True=0, Pred=1)
        fp_mask = (y_true == 0) & (preds == 1)
        mi = np.sum(fp_mask * ic) / N
        
        s = math.sqrt(ru**2 + mi**2)
        if s < s_min:
            s_min = s
            
    return s_min

def evaluate_all(y_true, y_pred_probs, ic):
    """
    Run all evaluations and return a dict of metrics.
    """
    metrics = {}
    metrics['Micro_Fmax'] = get_micro_fmax(y_true, y_pred_probs)
    metrics['Macro_Fmax'] = get_macro_fmax(y_true, y_pred_probs)
    metrics['Macro_AUROC'] = get_auroc(y_true, y_pred_probs, average='macro')
    metrics['Micro_AUROC'] = get_auroc(y_true, y_pred_probs, average='micro')
    metrics['Macro_AUPRC'] = get_auprc(y_true, y_pred_probs, average='macro')
    metrics['Micro_AUPRC'] = get_auprc(y_true, y_pred_probs, average='micro')
    metrics['Smin'] = get_smin(y_true, y_pred_probs, ic)
    return metrics
