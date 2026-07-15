#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
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

def parse_json_predictions(json_path, test_prots, go_terms, is_list_format=False):
    import json
    import numpy as np
    y_pred = np.zeros((len(test_prots), len(go_terms)), dtype=np.float32)
    covered_proteins = set()
    if not os.path.exists(json_path):
        print(f"  [MISSING] {json_path}")
        return y_pred, covered_proteins
    with open(json_path, 'r') as f:
        data = json.load(f)
    term2idx = {g: i for i, g in enumerate(go_terms)}
    for i, prot in enumerate(test_prots):
        if prot in data:
            for ont_key, preds in data[prot].items():
                if is_list_format:
                    if len(preds) > 0: covered_proteins.add(prot)
                    for t in preds:
                        if t in term2idx:
                            y_pred[i, term2idx[t]] = 1.0
                else:
                    if len(preds) > 0: covered_proteins.add(prot)
                    for t, score in preds.items():
                        if t in term2idx:
                            y_pred[i, term2idx[t]] = score
    return y_pred, covered_proteins

def parse_transfun(result_file, prot_list, goterms):
    """TransFun format: PROT GOTERM SCORE  (space-separated)
    Returns (y_pred, covered_proteins) where covered_proteins is the set of
    protein IDs that appeared in the output file (regardless of term mapping).
    """
    y_pred = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    covered_proteins = set()
    if not os.path.exists(result_file):
        print(f"  [MISSING] {result_file}")
        return y_pred, covered_proteins
    prot_idx = {p: i for i, p in enumerate(prot_list)}
    term_idx = {t: i for i, t in enumerate(goterms)}
    with open(result_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                p, t, s = parts[0], parts[1], float(parts[2])
                # Track ALL proteins that appear in the file
                if p in prot_idx:
                    covered_proteins.add(p)
                    if t in term_idx:
                        y_pred[prot_idx[p], term_idx[t]] = s
    return y_pred, covered_proteins

def parse_deepfri_json(json_file, prot_list, goterms):
    """DeepFRI JSON: {pdb_chains, Y_hat, goterms, gonames}
    Returns (y_pred, covered_proteins) where covered_proteins is the set of
    protein IDs from pdb_chains that are in our test set.
    """
    y_pred = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    covered_proteins = set()
    if not os.path.exists(json_file):
        print(f"  [MISSING] {json_file}")
        return y_pred, covered_proteins
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
        # Protein appeared in DeepFRI output → covered
        covered_proteins.add(dprot)
        pi = prot_idx[dprot]
        row = Y_hat[di]
        for dt, dj in df_term_idx.items():
            if dt in term_idx:
                y_pred[pi, term_idx[dt]] = row[dj]
    return y_pred, covered_proteins

def parse_dpfunc(result_file, prot_list, goterms):
    """DPFunc format: PROT GOTERM SCORE  (space-separated, same as TransFun)
    Returns (y_pred, covered_proteins).
    """
    return parse_transfun(result_file, prot_list, goterms)


def _coverage_diagnostic(model_name, covered_proteins, prot_list, goterms, y_pred):
    """Compute coverage diagnostic breakdown for a SOTA model.
    Returns dict with: Coverage_Failures, Label_Mismatch, Covered_Mapped.
    - Coverage_Failures: proteins NOT in the output file at all
    - Label_Mismatch: proteins in the output but with zero mapped terms (all-zero row)
    - Covered_Mapped: proteins in the output with ≥1 mapped term (non-zero row)
    """
    prot_set = set(prot_list)
    prot_idx = {p: i for i, p in enumerate(prot_list)}
    coverage_failures = len(prot_set - covered_proteins)
    label_mismatch = 0
    covered_mapped = 0
    for p in covered_proteins:
        if p in prot_idx:
            row = y_pred[prot_idx[p]]
            if np.any(row != 0):
                covered_mapped += 1
            else:
                label_mismatch += 1
    return {
        'Model': model_name,
        'Total_Test': len(prot_list),
        'Coverage_Failures': coverage_failures,
        'Label_Mismatch': label_mismatch,
        'Covered_Mapped': covered_mapped,
    }

def _find_robust(filename_variants, req_dir_part, search_roots):
    for sdir in search_roots:
        r = os.path.join(PROJECT_DIR, sdir)
        if not os.path.isdir(r): continue
        for root, _, files in os.walk(r):
            if req_dir_part.lower() in root.lower() or req_dir_part == '':
                for f in files:
                    if f in filename_variants:
                        return os.path.join(root, f)
    return ""


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
    p.add_argument('--mode', type=str, default='dl_only', choices=['dl_only', 'baselines_only', 'all'])
    p.add_argument('--common_subset', action='store_true',
                   help='Evaluate only on the common subset of proteins that ALL models '
                        'successfully processed (per ontology). Results saved separately.')
    return p.parse_args()



_FULL_ONT = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}

def main():
    import random
    np.random.seed(42)
    random.seed(42)
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets = load_datasets()
    if args.mode != 'dl_only':
        OUT_CSV = os.path.join(RESULTS_DIR, f'evaluation_results_{args.mode}.csv')
    else:
        OUT_CSV = os.path.join(RESULTS_DIR, 'evaluation_results.csv')
    
    all_rows = []
    common_subset_rows = []  # rows for --common_subset mode
    coverage_diag_rows = []  # coverage diagnostic table
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

        # ── generate y_true from loaded datasets.pkl ───────────────────────
        y_true = np.array([test_ds.prot2annot[p][ont_full] for p in prot_list])
        print(f"  y_true shape: {y_true.shape}")

        # ── generate IC from train set in datasets.pkl ─────────────────────
        train_ds = datasets[ont_full]['train']
        y_train_raw = np.array([train_ds.prot2annot[p][ont_full] for p in train_ds.pdb_split_list])
        ic = compute_ic(y_train_raw)

        if args.dry_run:
            print(f"  [DRY RUN] Paths OK for {ont_short}")
            continue

        # ── TransFun ─────────────────────────────────────────────────────────
        print("  [TransFun] Parsing ...")
        tf_file = _find_robust([f'{ont_short}_results.txt'], 'TransFun', ['SOTA', 'SOTA_predictions'])
        if not tf_file: tf_file = os.path.join(PROJECT_DIR, 'SOTA', 'TransFun', 'data', f'{ont_short}_results.txt')
        yp_tf, cov_tf = parse_transfun(tf_file, prot_list, goterms)
        print(f"    Coverage: {len(cov_tf)}/{len(prot_list)} proteins")

        # ── DeepFRI Seq ──────────────────────────────────────────────────────
        print("  [DeepFRI_Seq] Parsing ...")
        df_seq_variants = [f'deepfri_seq_{ont_short.upper()}_pred_scores.json', f'deepfri_seq_{ont_short.upper()}_{ont_short.upper()}_pred_scores.json']
        df_seq_file = _find_robust(df_seq_variants, 'deepfri', ['baselines', 'baselines/deepfri_results'])
        if not df_seq_file: df_seq_file = os.path.join(PROJECT_DIR, 'baselines', 'deepfri_results', f'deepfri_seq_{ont_short.upper()}_pred_scores.json')
        yp_df_seq, cov_df_seq = parse_deepfri_json(df_seq_file, prot_list, goterms)
        print(f"    Coverage: {len(cov_df_seq)}/{len(prot_list)} proteins")

        # ── DeepFRI Cmap ─────────────────────────────────────────────────────
        print("  [DeepFRI_Cmap] Parsing ...")
        df_cmap_variants = [f'deepfri_cmap_{ont_short.upper()}_pred_scores.json', f'deepfri_cmap_{ont_short.upper()}_{ont_short.upper()}_pred_scores.json']
        df_cmap_file = _find_robust(df_cmap_variants, 'deepfri', ['baselines', 'baselines/deepfri_results'])
        if not df_cmap_file: df_cmap_file = os.path.join(PROJECT_DIR, 'baselines', 'deepfri_results', f'deepfri_cmap_{ont_short.upper()}_pred_scores.json')
        yp_df_cmap, cov_df_cmap = parse_deepfri_json(df_cmap_file, prot_list, goterms)
        print(f"    Coverage: {len(cov_df_cmap)}/{len(prot_list)} proteins")

        # ── DPFunc ───────────────────────────────────────────────────────────
        print("  [DPFunc] Parsing ...")
        dpfunc_file = _find_robust([f'{ont_short}_results.txt'], 'DPFunc', ['SOTA_predictions', 'SOTA'])
        if not dpfunc_file: dpfunc_file = os.path.join(PROJECT_DIR, 'SOTA_predictions', 'DPFunc', f'{ont_short}_results.txt')
        yp_dpf, cov_dpf = parse_dpfunc(dpfunc_file, prot_list, goterms)
        print(f"    Coverage: {len(cov_dpf)}/{len(prot_list)} proteins")

        yp_blast, cov_blast = parse_json_predictions(os.path.join(PROJECT_DIR, 'baselines', 'blast', 'blast_predictions.json'), prot_list, goterms, True)
        yp_diamond, cov_diamond = parse_json_predictions(os.path.join(PROJECT_DIR, 'baselines', 'diamond', 'diamond_predictions.json'), prot_list, goterms, True)
        yp_naive, cov_naive = parse_json_predictions(os.path.join(PROJECT_DIR, 'baselines', 'naive_frequency', 'naive_predictions.json'), prot_list, goterms, False)


        # ── Coverage diagnostic ──────────────────────────────────────────────
        diag_models = []
        if args.mode in ('dl_only', 'all'):
            diag_models += [('TransFun', yp_tf, cov_tf), ('DeepFRI_Seq', yp_df_seq, cov_df_seq), ('DeepFRI_Cmap', yp_df_cmap, cov_df_cmap), ('DPFunc', yp_dpf, cov_dpf)]
        if args.mode in ('baselines_only', 'all'):
            diag_models += [('BLAST', yp_blast, cov_blast), ('DIAMOND', yp_diamond, cov_diamond), ('Naive', yp_naive, cov_naive)]
        for mname, yp, cov in diag_models:
            diag = _coverage_diagnostic(mname, cov, prot_list, goterms, yp)
            diag['Ontology'] = ont_short.upper()
            coverage_diag_rows.append(diag)

        # ── Full test set evaluation (Mode 1 — always runs) ──────────────────
        rng = np.random.default_rng(0)  # reset per ontology for consistency
        
        if args.mode in ('dl_only', 'all'):
            print("  [TransFun] Bootstrapping (full test set) ...")
            tf_records = bootstrap_eval(y_true, yp_tf, ic, n=args.bootstrap_seeds, rng=rng)
            all_rows.append(aggregate(tf_records, 'TransFun', ont_short.upper()))

            print("  [DeepFRI_Seq] Bootstrapping (full test set) ...")
            df_seq_records = bootstrap_eval(y_true, yp_df_seq, ic, n=args.bootstrap_seeds, rng=rng)
            all_rows.append(aggregate(df_seq_records, 'DeepFRI_Seq', ont_short.upper()))

            print("  [DeepFRI_Cmap] Bootstrapping (full test set) ...")
            df_cmap_records = bootstrap_eval(y_true, yp_df_cmap, ic, n=args.bootstrap_seeds, rng=rng)
            all_rows.append(aggregate(df_cmap_records, 'DeepFRI_Cmap', ont_short.upper()))

            print("  [DPFunc] Bootstrapping (full test set) ...")
            dpf_records = bootstrap_eval(y_true, yp_dpf, ic, n=args.bootstrap_seeds, rng=rng)
            all_rows.append(aggregate(dpf_records, 'DPFunc', ont_short.upper()))

        if args.mode in ('baselines_only', 'all'):
            for name, yp in [('BLAST', yp_blast), ('DIAMOND', yp_diamond), ('Naive', yp_naive)]:
                print(f"  [{name}] Bootstrapping ...")
                recs = bootstrap_eval(y_true, yp, ic, n=args.bootstrap_seeds, rng=rng)
                all_rows.append(aggregate(recs, name, ont_short.upper()))

        print("  [Hybrid] Loading 5-seed predictions ...")
        hybrid_recs = eval_your_model('Hybrid', ont_full, ont_short, y_true, ic)
        all_rows.append(aggregate(hybrid_recs, 'Hybrid', ont_short.upper()))

        print("  [Hybrid_JK] Loading 5-seed predictions ...")
        jk_recs = eval_your_model('Hybrid_JK', ont_full, ont_short, y_true, ic)
        all_rows.append(aggregate(jk_recs, 'Hybrid_JK', ont_short.upper()))

        # ── Common subset evaluation (Mode 2 — only when flag is set) ────────
        if args.common_subset:
            # Your models always cover all test proteins
            cov_hybrid = set(prot_list)
            cov_hybrid_jk = set(prot_list)

            # Intersection across active models
            common_prots = cov_hybrid & cov_hybrid_jk
            if args.mode in ('dl_only', 'all'):
                common_prots &= cov_tf & cov_df_seq & cov_df_cmap & cov_dpf
            if args.mode in ('baselines_only', 'all'):
                common_prots &= cov_blast & cov_diamond & cov_naive
            n_common = len(common_prots)
            n_total = len(prot_list)
            print(f"\n  [Common Subset] {ont_short.upper()}: {n_common}/{n_total} test proteins")

            # Build boolean mask aligned to prot_list ordering
            common_mask = np.array([p in common_prots for p in prot_list], dtype=bool)

            if common_mask.sum() < 5:
                print(f"  [WARN] Common subset too small ({common_mask.sum()}), skipping {ont_short.upper()}")
                continue

            # Filter
            y_true_cs = y_true[common_mask]
            yp_tf_cs  = yp_tf[common_mask]
            yp_df_seq_cs  = yp_df_seq[common_mask]
            yp_df_cmap_cs = yp_df_cmap[common_mask]
            yp_dpf_cs = yp_dpf[common_mask]
            
            rng_cs = np.random.default_rng(100) # diff seed for subset
            
            # Recompute IC on common subset if desired, or keep original
            ic_cs = ic 

            tf_cs_records = bootstrap_eval(y_true_cs, yp_tf_cs, ic_cs, n=args.bootstrap_seeds, rng=rng_cs)
            common_subset_rows.append(aggregate(tf_cs_records, 'TransFun', ont_short.upper()))

            df_seq_cs_records = bootstrap_eval(y_true_cs, yp_df_seq_cs, ic_cs, n=args.bootstrap_seeds, rng=rng_cs)
            common_subset_rows.append(aggregate(df_seq_cs_records, 'DeepFRI_Seq', ont_short.upper()))
            
            df_cmap_cs_records = bootstrap_eval(y_true_cs, yp_df_cmap_cs, ic_cs, n=args.bootstrap_seeds, rng=rng_cs)
            common_subset_rows.append(aggregate(df_cmap_cs_records, 'DeepFRI_Cmap', ont_short.upper()))

            print("  [DPFunc] Bootstrapping (common subset) ...")
            cs_dpf_recs = bootstrap_eval(y_true_cs, yp_dpf_cs, ic, n=args.bootstrap_seeds, rng=rng_cs)
            common_subset_rows.append(aggregate(cs_dpf_recs, 'DPFunc', ont_short.upper()))

            # Your models: reload per-seed preds and filter to common subset
            for mname in ('Hybrid', 'Hybrid_JK'):
                print(f"  [{mname}] Evaluating (common subset) ...")
                cs_records = []
                for seed in SEEDS:
                    seed_dir = os.path.join(RESULTS_DIR, ont_short, mname, str(seed))
                    pred_path = None
                    true_path = None
                    if os.path.exists(seed_dir):
                        if os.path.exists(os.path.join(seed_dir, 'test_y_pred.npy')):
                            pred_path = os.path.join(seed_dir, 'test_y_pred.npy')
                            true_path = os.path.join(seed_dir, 'test_y_true.npy')
                        else:
                            for sub in os.listdir(seed_dir):
                                candidate = os.path.join(seed_dir, sub, 'test_y_pred.npy')
                                if os.path.exists(candidate):
                                    pred_path = candidate
                                    true_path = os.path.join(seed_dir, sub, 'test_y_true.npy')
                                    break
                    if pred_path is None:
                        continue
                    yp = np.load(pred_path)[common_mask]
                    yt = np.load(true_path)[common_mask] if (true_path and os.path.exists(true_path)) else y_true_cs
                    if yt.shape[0] == 0:
                        continue
                    m = all_metrics(yt, yp, ic)
                    m['seed'] = seed
                    cs_records.append(m)
                common_subset_rows.append(aggregate(cs_records, mname, ont_short.upper()))

    # ── save full test set results (always) ───────────────────────────────────
    df = pd.DataFrame([r for r in all_rows if r])
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Results saved to {OUT_CSV}")
    print(df.to_string(index=False))

    # ── save coverage diagnostic CSV ─────────────────────────────────────────
    if coverage_diag_rows:
        diag_csv = os.path.join(RESULTS_DIR, 'coverage_diagnostic.csv')
        df_diag = pd.DataFrame(coverage_diag_rows)
        col_order = ['Model', 'Ontology', 'Total_Test', 'Coverage_Failures',
                     'Label_Mismatch', 'Covered_Mapped']
        df_diag = df_diag[[c for c in col_order if c in df_diag.columns]]
        df_diag.to_csv(diag_csv, index=False)
        print(f"\n✓ Coverage diagnostic saved to {diag_csv}")
        print(df_diag.to_string(index=False))

    # ── save common subset results (when --common_subset) ────────────────────
    if args.common_subset and common_subset_rows:
        cs_csv = OUT_CSV.replace('.csv', '_common_subset.csv')
        df_cs = pd.DataFrame([r for r in common_subset_rows if r])
        df_cs.to_csv(cs_csv, index=False)
        print(f"\n✓ Common subset results saved to {cs_csv}")
        print(df_cs.to_string(index=False))

if __name__ == '__main__':
    main()
