import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
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
    """Compute Area Under the Precision-Recall Curve."""
    if average == "micro":
        precision, recall, _ = precision_recall_curve(y_true.flatten(), y_pred_probs.flatten())
        return auc(recall, precision)
    elif average == "macro":
        auprc_list = []
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i]) == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
            auprc_list.append(auc(recall, precision))
        if len(auprc_list) == 0:
            return 0.0
        return np.mean(auprc_list)
    return 0.0

def compute_ic(y_train):
    """
    Compute Information Content for each term based on frequency in the training set.
    """
    N = y_train.shape[0]
    counts = np.sum(y_train, axis=0)
    ic = np.zeros_like(counts, dtype=float)
    valid_mask = counts > 0
    ic[valid_mask] = -np.log2(counts[valid_mask] / N)
    return ic

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
