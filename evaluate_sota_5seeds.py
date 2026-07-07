#!/usr/bin/env python3
"""
evaluate_sota_5seeds.py
=======================
Unified evaluation script for the seeded comparison study.

What this script does:
  1. Loads the ground-truth test set from the preprocessed pickle.
  2. For SOTA models (DeepFRI, TransFun, DPFunc) — which are deterministic —
     it runs 5 BOOTSTRAP resamples of the test set to produce mean ± SE.
  3. For your models (Hybrid, Hybrid_JK) — which were trained 5 times with
     different seeds — it loads each seed's saved test_y_pred.npy and
     test_y_true.npy and computes metrics per seed, then aggregates.
  4. Writes all results to  runs_5seeds/evaluation_results.csv

Usage (run from project root):
    python evaluate_sota_5seeds.py
    python evaluate_sota_5seeds.py --bootstrap_seeds 5  # adjust number of bootstraps
    python evaluate_sota_5seeds.py --dry_run            # validate paths only

Layout of expected files
  SOTA predictions (deterministic, one file per ontology):
    SOTA/TransFun/data/{bp,mf,cc}_results.txt   — format: "PROT GOTERM SCORE"
    baselines/deepfri_results/deepfri_seq_{BP,MF,CC}_pred_scores.json
    SOTA/DPFunc/workspace_{bp,mf,cc}/result/     — DPFunc output parquet/pkl

  Your model predictions (saved by train.py):
    runs_5seeds/{ont}/Hybrid/{seed}/test_y_pred.npy
    runs_5seeds/{ont}/Hybrid/{seed}/test_y_true.npy
    runs_5seeds/{ont}/Hybrid_JK/{seed}/test_y_pred.npy
    runs_5seeds/{ont}/Hybrid_JK/{seed}/test_y_true.npy
"""

import os
import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
import math
import warnings

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PKL = os.path.join(PROJECT_DIR, 'preprocessing', 'data', 'split_files', 'datasets.pkl')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'runs_5seeds')
OUT_CSV     = os.path.join(RESULTS_DIR, 'evaluation_results.csv')

SEEDS = [42, 43, 44, 45, 46]
N_BOOTSTRAPS = 5   # overridden by --bootstrap_seeds

ONTOLOGIES = {
    'biological_process': 'bp',
    'molecular_function': 'mf',
    'cellular_component': 'cc',
}

# ── metric helpers ────────────────────────────────────────────────────────────

def get_micro_fmax(y_true, y_pred):
    prec, rec, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    return float(np.max(f1))

def get_macro_fmax(y_true, y_pred):
    vals = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true[:, j], y_pred[:, j])
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        vals.append(np.max(f1))
    return float(np.mean(vals)) if vals else 0.0

def get_micro_auroc(y_true, y_pred):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return float(roc_auc_score(y_true, y_pred, average='micro'))
    except:
        return float('nan')

def get_micro_auprc(y_true, y_pred):
    prec, rec, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    return float(auc(rec, prec))

def compute_ic(y_train):
    N = y_train.shape[0]
    counts = np.sum(y_train, axis=0)
    ic = np.zeros(counts.shape, dtype=float)
    mask = counts > 0
    ic[mask] = -np.log2(counts[mask] / N)
    return ic

def get_smin(y_true, y_pred, ic):
    thresholds = np.arange(0.01, 1.0, 0.01)
    s_min = float('inf')
    N = y_true.shape[0]
    for t in thresholds:
        preds = (y_pred >= t).astype(int)
        fn_mask = (y_true == 1) & (preds == 0)
        ru = np.sum(fn_mask * ic) / N
        fp_mask = (y_true == 0) & (preds == 1)
        mi = np.sum(fp_mask * ic) / N
        s = math.sqrt(ru**2 + mi**2)
        if s < s_min:
            s_min = s
    return float(s_min)

def all_metrics(y_true, y_pred, ic):
    return {
        'Micro_Fmax':  get_micro_fmax(y_true, y_pred),
        'Macro_Fmax':  get_macro_fmax(y_true, y_pred),
        'Micro_AUROC': get_micro_auroc(y_true, y_pred),
        'Micro_AUPRC': get_micro_auprc(y_true, y_pred),
        'Smin':        get_smin(y_true, y_pred, ic),
    }

# ── prediction parsers ────────────────────────────────────────────────────────

def parse_transfun(result_file, prot_list, goterms):
    """TransFun format: PROT GOTERM SCORE  (space-separated)"""
    y_pred = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    if not os.path.exists(result_file):
        print(f"  [MISSING] {result_file}")
        return y_pred
    prot_idx = {p: i for i, p in enumerate(prot_list)}
    term_idx = {t: i for i, t in enumerate(goterms)}
    with open(result_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                p, t, s = parts[0], parts[1], float(parts[2])
                if p in prot_idx and t in term_idx:
                    y_pred[prot_idx[p], term_idx[t]] = s
    return y_pred

def parse_deepfri_json(json_file, prot_list, goterms):
    """DeepFRI JSON: {pdb_chains, Y_hat, goterms, gonames}"""
    y_pred = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    if not os.path.exists(json_file):
        print(f"  [MISSING] {json_file}")
        return y_pred
    with open(json_file) as f:
        d = json.load(f)
    deepfri_chains = d['pdb_chains']
    deepfri_terms  = d['goterms']
    Y_hat          = np.array(d['Y_hat'], dtype=np.float32)  # shape: (n_df_prots, n_df_terms)

    prot_idx  = {p: i for i, p in enumerate(prot_list)}
    term_idx  = {t: i for i, t in enumerate(goterms)}
    df_term_idx = {t: i for i, t in enumerate(deepfri_terms)}

    for di, dprot in enumerate(deepfri_chains):
        if dprot not in prot_idx:
            continue
        pi = prot_idx[dprot]
        row = Y_hat[di]
        for dt, dj in df_term_idx.items():
            if dt in term_idx:
                y_pred[pi, term_idx[dt]] = row[dj]
    return y_pred

def parse_dpfunc(result_file, prot_list, goterms):
    """DPFunc format: PROT GOTERM SCORE  (space-separated, same as TransFun)"""
    return parse_transfun(result_file, prot_list, goterms)

# ── dataset loading ───────────────────────────────────────────────────────────

def load_datasets():
    print(f"Loading datasets from {DATASET_PKL} ...")
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'numpy._core.multiarray':
                module = 'numpy.core.multiarray'
            return super().find_class(module, name)
    # register PDB_Dataset so pickle can find it
    import __main__
    from preprocessing.create_batch_dataset import PDB_Dataset
    __main__.PDB_Dataset = PDB_Dataset
    with open(DATASET_PKL, 'rb') as f:
        return _Unpickler(f).load()

# ── bootstrap for deterministic models ───────────────────────────────────────

def bootstrap_eval(y_true, y_pred, ic, n=5, rng=None):
    """
    Draw `n` bootstrap resamples and return per-seed metric dicts.
    Uses fixed seeds for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    N = y_true.shape[0]
    records = []
    for i in range(n):
        idx = rng.choice(N, N, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        if yt.sum() == 0:
            continue
        m = all_metrics(yt, yp, ic)
        m['seed'] = SEEDS[i] if i < len(SEEDS) else i
        records.append(m)
    return records

# ── aggregate list of per-seed dicts ─────────────────────────────────────────

def aggregate(records, model, ontology):
    """Convert list of {metric: val} to summary row with mean/std/se."""
    if not records:
        return {}
    keys = [k for k in records[0] if k != 'seed']
    row = {'Model': model, 'Ontology': ontology, 'N_seeds': len(records)}
    for k in keys:
        vals = [r[k] for r in records if not math.isnan(r.get(k, float('nan')))]
        row[f'{k}_mean'] = float(np.mean(vals)) if vals else float('nan')
        row[f'{k}_std']  = float(np.std(vals))  if vals else float('nan')
        row[f'{k}_se']   = float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return row

# ── your model evaluation ─────────────────────────────────────────────────────

def eval_your_model(model_name, ont_full, ont_short, y_true, ic):
    """
    Loads test_y_pred.npy for each seed from
      runs_5seeds/{ont_short}/{model_name}/{seed}/
    train.py creates a run_name subdirectory inside output_dir, so we
    search recursively for test_y_pred.npy under each seed directory.
    """
    records = []
    for seed in SEEDS:
        seed_dir = os.path.join(RESULTS_DIR, ont_short, model_name, str(seed))

        # train.py saves files inside output_dir/run_name/, so search one level deeper
        pred_path = None
        true_path = None
        if os.path.exists(seed_dir):
            # First check directly in seed_dir (in case someone ran with flat structure)
            if os.path.exists(os.path.join(seed_dir, 'test_y_pred.npy')):
                pred_path = os.path.join(seed_dir, 'test_y_pred.npy')
                true_path = os.path.join(seed_dir, 'test_y_true.npy')
            else:
                # Search one level deeper (run_name subdir created by train.py)
                for sub in os.listdir(seed_dir):
                    candidate = os.path.join(seed_dir, sub, 'test_y_pred.npy')
                    if os.path.exists(candidate):
                        pred_path = candidate
                        true_path = os.path.join(seed_dir, sub, 'test_y_true.npy')
                        break

        if pred_path is None:
            print(f"  [MISSING] test_y_pred.npy under {seed_dir}")
            continue

        yp = np.load(pred_path)
        yt = np.load(true_path) if (true_path and os.path.exists(true_path)) else y_true

        if yt.shape != y_true.shape:
            print(f"  [SHAPE MISMATCH] {true_path}: {yt.shape} vs {y_true.shape}")
            continue

        m = all_metrics(yt, yp, ic)
        m['seed'] = seed
        records.append(m)
    return records

# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bootstrap_seeds', type=int, default=N_BOOTSTRAPS,
                   help='Number of bootstrap resamples for deterministic SOTA models')
    p.add_argument('--dry_run', action='store_true',
                   help='Only check file paths, do not compute metrics')
    return p.parse_args()

def _find_cached_npy(ont_short, key):
    """
    Finds a cached test_y_true.npy or train_labels.npy without loading
    the heavy PyG .pt graph files.  Searches tuning_runs_jk first, then
    tuning_runs, then runs_5seeds.
    key is either 'test_y_true' or 'train_labels'.
    """
    if key == 'train_labels':
        candidates = [
            os.path.join(PROJECT_DIR, f'{ont_short}_train_labels.npy'),
            os.path.join(PROJECT_DIR, f'{_FULL_ONT[ont_short]}_train_labels.npy'),
        ]
        for c in candidates:
            if os.path.exists(c):
                return np.load(c)
        return None

    # test_y_true — find any completed run for this ontology
    for runs_dir in ('tuning_runs_jk', 'tuning_runs', 'runs_jk_test', 'runs_5seeds'):
        search_root = os.path.join(PROJECT_DIR, runs_dir)
        if not os.path.isdir(search_root):
            continue
        for root, dirs, files in os.walk(search_root):
            if 'test_y_true.npy' in files:
                # Make sure this is the right ontology
                if os.path.basename(root).startswith(ont_short + '_'):
                    return np.load(os.path.join(root, 'test_y_true.npy'))
    return None

_FULL_ONT = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}

def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets = load_datasets()
    all_rows = []
    rng = np.random.default_rng(0)

    for ont_full, ont_short in ONTOLOGIES.items():
        print(f"\n{'='*60}")
        print(f"  Ontology: {ont_full.upper()}")
        print(f"{'='*60}")

        # ── metadata from pickle (no .pt files needed) ─────────────────────
        test_ds  = datasets[ont_full]['test']
        goterms   = test_ds.y_labels        # list of GO term IDs
        prot_list = test_ds.pdb_split_list  # list of PDB chain IDs

        print(f"  Test proteins: {len(prot_list)}, GO terms: {len(goterms)}")

        # ── load y_true from a pre-saved run (avoids needing .pt files) ────
        y_true = _find_cached_npy(ont_short, 'test_y_true')
        if y_true is None:
            print(f"  [ERROR] No cached test_y_true.npy found for {ont_short}.")
            print(f"  Run at least one training job for {ont_full} first.")
            continue
        print(f"  y_true shape: {y_true.shape}")

        # ── load IC from pre-saved train labels ────────────────────────────
        y_train_raw = _find_cached_npy(ont_short, 'train_labels')
        if y_train_raw is not None:
            ic = compute_ic(y_train_raw)
        else:
            # Fallback: approximate IC from y_true (not ideal but functional)
            print(f"  [WARN] train_labels.npy not found for {ont_short}, approximating IC from test set.")
            ic = compute_ic(y_true)

        if args.dry_run:
            print(f"  [DRY RUN] Paths OK for {ont_short}")
            continue

        # ── TransFun ─────────────────────────────────────────────────────────
        print("  [TransFun] Bootstrapping ...")
        tf_file = os.path.join(PROJECT_DIR, 'SOTA', 'TransFun', 'data', f'{ont_short}_results.txt')
        yp_tf = parse_transfun(tf_file, prot_list, goterms)
        tf_records = bootstrap_eval(y_true, yp_tf, ic, n=args.bootstrap_seeds, rng=rng)
        all_rows.append(aggregate(tf_records, 'TransFun', ont_short.upper()))

        # ── DeepFRI ──────────────────────────────────────────────────────────
        print("  [DeepFRI] Bootstrapping ...")
        df_file = os.path.join(PROJECT_DIR, 'baselines', 'deepfri_results',
                               f'deepfri_seq_{ont_short.upper()}_pred_scores.json')
        yp_df = parse_deepfri_json(df_file, prot_list, goterms)
        df_records = bootstrap_eval(y_true, yp_df, ic, n=args.bootstrap_seeds, rng=rng)
        all_rows.append(aggregate(df_records, 'DeepFRI', ont_short.upper()))

        # ── DPFunc ───────────────────────────────────────────────────────────
        print("  [DPFunc] Bootstrapping ...")
        dpfunc_file = os.path.join(PROJECT_DIR, 'SOTA_predictions', 'DPFunc',
                                   f'{ont_short}_results.txt')
        yp_dpf = parse_dpfunc(dpfunc_file, prot_list, goterms)
        dpf_records = bootstrap_eval(y_true, yp_dpf, ic, n=args.bootstrap_seeds, rng=rng)
        all_rows.append(aggregate(dpf_records, 'DPFunc', ont_short.upper()))

        # ── Hybrid ───────────────────────────────────────────────────────────
        print("  [Hybrid] Loading 5-seed predictions ...")
        hybrid_recs = eval_your_model('Hybrid', ont_full, ont_short, y_true, ic)
        all_rows.append(aggregate(hybrid_recs, 'Hybrid', ont_short.upper()))

        # ── Hybrid_JK ────────────────────────────────────────────────────────
        print("  [Hybrid_JK] Loading 5-seed predictions ...")
        jk_recs = eval_your_model('Hybrid_JK', ont_full, ont_short, y_true, ic)
        all_rows.append(aggregate(jk_recs, 'Hybrid_JK', ont_short.upper()))

    # ── save ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame([r for r in all_rows if r])
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Results saved to {OUT_CSV}")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
