#!/usr/bin/env python3
"""
plot_sota_comparison.py
=======================
Generates publication-quality DPFunc-style comparison figures after
evaluate_sota_5seeds.py has been run.

Figures produced (saved to plots_sota_comparison/):
  A.  Fmax vs Sequence Identity bins (hard proteins)
  B.  IC-weighted PR curves — MF
  C.  Fmax vs IC bins (rare GO terms)
  D.  IC-weighted PR curves — CC
  E.  IC-weighted PR curves — BP
  F.  Fmax vs GO term Depth bins
  G.  Coverage (unique predicted GO terms per method)

Usage (from project root):
    python plot_sota_comparison.py

Requirements:
  - runs_5seeds/evaluation_results.csv          (from evaluate_sota_5seeds.py)
  - runs_5seeds/{ont}/{model}/{seed}/test_y_pred.npy
  - SOTA/TransFun/data/{bp,mf,cc}_results.txt
  - baselines/deepfri_results/deepfri_seq_{BP,MF,CC}_pred_scores.json
  - SOTA_predictions/DPFunc/{bp,mf,cc}_results.txt
  - preprocessing/data/split_files/datasets.pkl
  - Optional: preprocessing/data/blast_identity.csv   (for plot A)
              preprocessing/data/go_properties.csv    (for plots C, F)
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import precision_recall_curve
import warnings

warnings.filterwarnings('ignore')

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'runs_5seeds')
OUT_DIR     = os.path.join(PROJECT_DIR, 'plots_sota_comparison')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]

ONTOLOGIES = {
    'biological_process': 'bp',
    'molecular_function': 'mf',
    'cellular_component': 'cc',
}

# ── colour palette (Paul Tol vibrant, high-contrast) ──────────────────────────
PALETTE = {
    'Hybrid_JK': '#EE3377',   # magenta
    'Hybrid':    '#CC3311',   # red
    'TransFun':  '#0077BB',   # blue
    'DPFunc':    '#009988',   # teal
    'DeepFRI':   '#EE7733',   # orange
    'BLAST':     '#882255',   # wine
    'DIAMOND':   '#44AA99',   # dark teal
    'Naive':     '#999933',   # olive
}
MODEL_ORDER_PERFORMANCE = ['Hybrid', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']
MODEL_ORDER_COVERAGE = ['Hybrid', 'DPFunc', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']
MODEL_ORDER_SUPPLEMENTARY = ['Hybrid_JK', 'Hybrid', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']

# Will be set in main()
MODEL_ORDER = MODEL_ORDER_PERFORMANCE
MODEL_COLORS = {
    'Hybrid': '#1f77b4',
    'Hybrid_JK': '#ff7f0e',
    'DPFunc': '#2ca02c',
    'TransFun': '#9467bd',
    'DeepFRI_Seq': '#d62728',
    'DeepFRI_Cmap': '#8c564b',
    'BLAST': '#882255',
    'DIAMOND': '#44AA99',
    'Naive': '#999933'
}

# ── matplotlib config ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 4,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.edgecolor': '#999999',
    'grid.color':     '#DDDDDD',
    'grid.linewidth': 0.6,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#AAAAAA')
    ax.spines['bottom'].set_color('#AAAAAA')
    ax.grid(axis='y', alpha=0.5)
    ax.set_axisbelow(True)

def save_fig(name):
    global args
    if args and args.common_subset:
        name += '_common_subset' 
    if args and getattr(args, 'supplementary', False):
        name += '_supp'
        out_dir = os.path.join(OUT_DIR, 'supplementary')
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = OUT_DIR
    path = os.path.join(out_dir, f'{name}.png')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'✓ Saved {path}')

def load_datasets():
    class _Up(pickle.Unpickler):
        def find_class(self, m, n):
            if m == 'numpy._core.multiarray':
                m = 'numpy.core.multiarray'
            return super().find_class(m, n)
    import __main__
    from preprocessing.create_batch_dataset import PDB_Dataset
    __main__.PDB_Dataset = PDB_Dataset
    pkl = os.path.join(PROJECT_DIR, 'preprocessing', 'data', 'split_files', 'datasets.pkl')
    with open(pkl, 'rb') as f:
        return _Up(f).load()

_FULL_ONT = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}

def _load_valid_mask(ont_short):
    mask_file = os.path.join(PROJECT_DIR, f"{ont_short}_valid_mask.npy")
    if os.path.exists(mask_file):
        return np.load(mask_file)
    return None

def _load_y_true(ont_short, datasets):
    """Extract test_y_true directly from datasets.pkl"""
    ont_full = _FULL_ONT[ont_short]
    test_ds = datasets[ont_full]['test']
    return np.array([test_ds.prot2annot[p][ont_full] for p in test_ds.pdb_split_list])

def _load_ic(ont_short, datasets):
    """Extract training IC directly from datasets.pkl"""
    ont_full = _FULL_ONT[ont_short]
    train_ds = datasets[ont_full]['train']
    y_train = np.array([train_ds.prot2annot[p][ont_full] for p in train_ds.pdb_split_list])
    return compute_ic(y_train)

def compute_ic(y_train):
    N = y_train.shape[0]
    counts = np.sum(y_train, axis=0)
    ic = np.zeros(counts.shape, dtype=float)
    m = counts > 0
    ic[m] = -np.log2(counts[m] / N)
    return ic


# ── prediction loaders ────────────────────────────────────────────────────────

def _json_to_matrix(json_path, prot_list, goterms, is_list_format=False):
    import json
    y_pred = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    cov_sets = set()
    if not os.path.exists(json_path):
        print(f"  [MISSING] {json_path}")
        return y_pred, cov_sets
    with open(json_path, 'r') as f:
        data = json.load(f)
    prot_idx = {p: i for i, p in enumerate(prot_list)}
    term_idx = {t: i for i, t in enumerate(goterms)}
    
    for p in prot_list:
        if p in data:
            for ont_key, preds in data[p].items():
                if is_list_format:
                    if len(preds) > 0: cov_sets.add(p)
                    for t in preds:
                        if t in term_idx:
                            y_pred[prot_idx[p], term_idx[t]] = 1.0
                else:
                    if len(preds) > 0: cov_sets.add(p)
                    for t, score in preds.items():
                        if t in term_idx:
                            y_pred[prot_idx[p], term_idx[t]] = float(score)
    return y_pred, cov_sets

def _txt_to_matrix(path, prot_list, goterms):
    """Load space-separated 'PROT TERM SCORE' file."""
    y = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    cov = set()
    if not os.path.exists(path):
        print(f'  [MISSING] {path}')
        return y, cov
    pi = {p: i for i, p in enumerate(prot_list)}
    ti = {t: i for i, t in enumerate(goterms)}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] in pi:
                cov.add(parts[0])
                if parts[1] in ti:
                    y[pi[parts[0]], ti[parts[1]]] = float(parts[2])
    return y, cov

def _deepfri_to_matrix(path, prot_list, goterms):
    if not os.path.exists(path):
        print(f'  [MISSING] {path}')
        return np.zeros((len(prot_list), len(goterms)), dtype=np.float32), set()
    with open(path) as f:
        d = json.load(f)
    chains = d.get('pdb_chains', [])
    Yhat   = np.array(d.get('Y_hat', []), dtype=np.float32)
    
    # If pdb_chains is empty or not present, try to find another key with the protein names
    if len(chains) != Yhat.shape[0]:
        for k, v in d.items():
            if isinstance(v, list) and len(v) == Yhat.shape[0] and len(v) > 0 and isinstance(v[0], str):
                chains = v
                break

    dterms = d.get('goterms', [])
    y = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    
    pi = {p: i for i, p in enumerate(prot_list)}
    ti = {t: i for i, t in enumerate(goterms)}
    dti = {t: i for i, t in enumerate(dterms)}
    
    match_chains = set()
    
    for di, dp in enumerate(chains):
        candidates = [dp, dp.replace('-', '_'), dp.split('_')[0] + '_' + dp.split('_')[-1] if '_' in dp else dp]
        best_p = next((c for c in candidates if c in pi), None)
        
        if not best_p:
            continue
            
        match_chains.add(best_p)
        row = Yhat[di]
        for dt, dj in dti.items():
            dt_fix = dt if dt.startswith('GO:') else f'GO:{dt}'
            if dt_fix in ti:
                y[pi[best_p], ti[dt_fix]] = row[dj]

    match_terms = sum(1 for dt in dterms if (dt if dt.startswith('GO:') else f'GO:{dt}') in ti)
    print(f"  [DeepFRI Debug] {os.path.basename(path)} -> matched {len(match_chains)}/{len(chains)} test proteins, {match_terms}/{len(dterms)} terms. Max score: {np.max(y):.4f}")
    return y, match_chains

def _find_robust(filename_variants, req_dir_part, search_roots):
    for sdir in search_roots:
        r = os.path.join(PROJECT_DIR, sdir)
        if not os.path.isdir(r): continue
        for root, _, files in os.walk(r):
            if req_dir_part.lower() in root.lower() or req_dir_part == '':
                for f in files:
                    if f in filename_variants:
                        return os.path.join(root, f)
    return None

def load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=None):
    """Returns (preds, cov_sets) dicts mapping model to predictions and coverage."""
    preds = {}
    cov_sets = {}

    # TransFun
    tf_file = _find_robust([f'{ont_short}_results.txt'], 'TransFun', ['SOTA', 'SOTA_predictions'])
    if not tf_file: tf_file = os.path.join(PROJECT_DIR, 'SOTA', 'TransFun', 'data', f'{ont_short}_results.txt')
    preds['TransFun'], cov_sets['TransFun'] = _txt_to_matrix(tf_file, prot_list, goterms)

    # DeepFRI Sequence Mode
    df_seq_variants = [f'deepfri_seq_{ont_short.upper()}_pred_scores.json', f'deepfri_seq_{ont_short.upper()}_{ont_short.upper()}_pred_scores.json']
    df_seq_file = _find_robust(df_seq_variants, 'deepfri', ['baselines', 'baselines/deepfri_results'])
    if not df_seq_file: df_seq_file = os.path.join(PROJECT_DIR, 'baselines', 'deepfri_results', f'deepfri_seq_{ont_short.upper()}_pred_scores.json')
    preds['DeepFRI_Seq'], cov_sets['DeepFRI_Seq'] = _deepfri_to_matrix(df_seq_file, prot_list, goterms)

    # DeepFRI Structure Mode (Cmap)
    df_cmap_variants = [f'deepfri_cmap_{ont_short.upper()}_pred_scores.json', f'deepfri_cmap_{ont_short.upper()}_{ont_short.upper()}_pred_scores.json']
    df_cmap_file = _find_robust(df_cmap_variants, 'deepfri', ['baselines', 'baselines/deepfri_results'])
    if not df_cmap_file: df_cmap_file = os.path.join(PROJECT_DIR, 'baselines', 'deepfri_results', f'deepfri_cmap_{ont_short.upper()}_pred_scores.json')
    preds['DeepFRI_Cmap'], cov_sets['DeepFRI_Cmap'] = _deepfri_to_matrix(df_cmap_file, prot_list, goterms)

    # DPFunc
    dpf_file = _find_robust([f'{ont_short}_results.txt'], 'DPFunc', ['SOTA_predictions', 'SOTA'])
    if not dpf_file: dpf_file = os.path.join(PROJECT_DIR, 'SOTA_predictions', 'DPFunc', f'{ont_short}_results.txt')
    preds['DPFunc'], cov_sets['DPFunc'] = _txt_to_matrix(dpf_file, prot_list, goterms)

    # Your models — average over available seeds
    for mname in ('Hybrid', 'Hybrid_JK'):
        cov_sets[mname] = set(prot_list)
        seed_preds_list = load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list), valid_mask=valid_mask)
        seed_preds = [arr for s, arr in seed_preds_list]
        if seed_preds:
            preds[mname] = np.mean(np.stack(seed_preds, axis=0), axis=0)
        else:
            preds[mname] = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
            print(f'  [MISSING] No seeds found for {mname}/{ont_short}')

    global args
    if args and args.common_subset:
        common = set(prot_list)
        for mname, c in cov_sets.items():
            if mname != 'DeepFRI_Cmap':
                common = common.intersection(c)
        mask = np.array([p in common for p in prot_list])
        print(f"  [Common Subset] {ont_short}: {len(common)}/{len(prot_list)}")
        for m in preds:
            preds[m] = preds[m][mask]
        return preds, mask
    
    return preds, None

def load_per_seed_preds(ont_short, mname, goterms_len, n_prots, mask=None, valid_mask=None):
    """Load per-seed y_pred arrays for your models.
    train.py creates output_dir/run_name/ so we search one level deep.
    """
    out = []
    for s in SEEDS:
        seed_dir = os.path.join(RESULTS_DIR, ont_short, mname, str(s))
        pred_path = None
        if os.path.exists(seed_dir):
            direct = os.path.join(seed_dir, 'test_y_pred.npy')
            if os.path.exists(direct):
                pred_path = direct
            else:
                for sub in os.listdir(seed_dir):
                    candidate = os.path.join(seed_dir, sub, 'test_y_pred.npy')
                    if os.path.exists(candidate):
                        pred_path = candidate
                        break
        if pred_path is not None:
            sp = np.load(pred_path)
            if valid_mask is not None:
                sp = sp[:, valid_mask]
            if mask is not None:
                sp = sp[mask]
            out.append((s, sp))
    return out


# ── PLOT A: Fmax vs sequence identity bins ────────────────────────────────────

def plot_A_sequence_identity(datasets):
    """
    Requires preprocessing/data/blast_identity.csv with columns:
        prot, max_identity  (float 0-1)
    Generated by: preprocessing/calc_blast_identity.sh
    """
    blast_csv = os.path.join(PROJECT_DIR, 'preprocessing', 'data', 'blast_identity.csv')
    
    # We use BP as the representative ontology for this figure (like DPFunc paper)
    ont_full, ont_short = 'biological_process', 'bp'
    test_ds  = datasets[ont_full]['test']
    prot_list = test_ds.pdb_split_list
    goterms   = test_ds.y_labels

    if not os.path.exists(blast_csv):
        # Auto-generate it from BLAST or Diamond results if possible
        import glob
        tsv_matches = glob.glob(os.path.join(PROJECT_DIR, 'baselines', '*', '*_results.tsv'))
        # If glob fails, try a robust walk
        if not tsv_matches:
            for root, _, files in os.walk(os.path.join(PROJECT_DIR, 'baselines')):
                for f in files:
                    if f.endswith('_results.tsv') or f.endswith('_results.txt') or f.endswith('.m8'):
                        tsv_matches.append(os.path.join(root, f))
        
        if tsv_matches:
            print(f'  [INFO] Auto-generating blast_identity.csv from {tsv_matches[0]} ...')
            df_align = pd.read_csv(tsv_matches[0], sep='\t', header=None, usecols=[0, 2], names=['prot', 'pident'])
            max_id = df_align.groupby('prot')['pident'].max() / 100.0
            id_map = {p: max_id.get(p, 0.0) for p in prot_list}
            pd.DataFrame({'prot': list(id_map.keys()), 'max_identity': list(id_map.values())}).to_csv(blast_csv, index=False)
        else:
            print(f'  [SKIP plot A] {blast_csv} not found and no baseline alignment file found.')
            print(f'                Run `python preprocessing/calc_blast_identity.py` to generate it.')
            return

    id_df = pd.read_csv(blast_csv)   # columns: prot, max_identity
    id_map = dict(zip(id_df['prot'], id_df['max_identity']))

    bins   = [(0, 0.2, '<20%'), (0.2, 0.4, '20–40%'),
              (0.4, 0.6, '40–60%'), (0.6, 1.01, '>60%')]
    y_true    = _load_y_true(ont_short, datasets)
    if y_true is None:
        print('  [SKIP plot A] No cached y_true for bp.')
        return
        
    valid_mask = _load_valid_mask(ont_short)
    if valid_mask is not None:
        y_true = y_true[:, valid_mask]
        goterms = [gt for gt, v in zip(goterms, valid_mask) if v]
        
    ic_raw    = _load_ic(ont_short, datasets)
    if ic_raw is not None and valid_mask is not None:
        ic_raw = ic_raw[:, valid_mask]
    ic        = ic_raw if ic_raw is not None else compute_ic(y_true)
    preds, mask = load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
    if mask is not None:
        y_true = y_true[mask]
        # Depending on context, ic or prot_identity or ic_bins may need masking.
        # It's better to mask them directly after this call.

    # Assign each test protein to an identity bin
    prot_identity = np.array([id_map.get(p, 1.0) for p in prot_list])
    if mask is not None:
        prot_identity = prot_identity[mask]
        if ic is not None and len(ic) == len(mask):
            ic = ic[mask]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    active_bins = [b for b in bins if ((prot_identity >= b[0]) & (prot_identity < b[1])).sum() > 0]
    x_positions = np.arange(len(active_bins))
    bar_width   = 0.15
    n_models    = len(MODEL_ORDER)
    ont_stats   = [{'bin': b[2]} for b in active_bins]

    for mi, mname in enumerate(MODEL_ORDER):
        if mname not in preds:
            continue
        # per-seed runs for your models, bootstrap for SOTA
        seed_preds_list = []
        if mname in ('Hybrid', 'Hybrid_JK'):
            for _, sp in load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list), mask=mask, valid_mask=valid_mask):
                seed_preds_list.append(sp)
            if not seed_preds_list:
                seed_preds_list = [preds[mname]]
        else:
            rng = np.random.default_rng(mi)
            N = y_true.shape[0]
            for _ in range(5):
                idx = rng.choice(N, N, replace=True)
                seed_preds_list.append(preds[mname][idx])
            # We also need to subset prot_identity accordingly — handled per bin below

        fmax_means, fmax_sems = [], []
        for b_idx, (lo, hi, _) in enumerate(active_bins):
            bin_mask = (prot_identity >= lo) & (prot_identity < hi)
            if bin_mask.sum() < 5:
                fmax_means.append(np.nan)
                fmax_sems.append(0.0)
                continue
            bin_fmaxes = []
            if mname in ('Hybrid', 'Hybrid_JK'):
                for sp in seed_preds_list:
                    yt_b = y_true[bin_mask]
                    yp_b = sp[bin_mask]
                    if yt_b.sum() == 0:
                        continue
                    prec, rec, _ = precision_recall_curve(yt_b.ravel(), yp_b.ravel())
                    f1 = 2*prec*rec/(prec+rec+1e-10)
                    bin_fmaxes.append(np.max(f1))
            else:
                rng2 = np.random.default_rng(mi*100)
                n_bin = bin_mask.sum()
                for _ in range(5):
                    idx = rng2.choice(n_bin, n_bin, replace=True)
                    yt_b = y_true[bin_mask][idx]
                    yp_b = preds[mname][bin_mask][idx]
                    if yt_b.sum() == 0:
                        continue
                    prec, rec, _ = precision_recall_curve(yt_b.ravel(), yp_b.ravel())
                    f1 = 2*prec*rec/(prec+rec+1e-10)
                    bin_fmaxes.append(np.max(f1))
            if bin_fmaxes:
                fmax_means.append(np.mean(bin_fmaxes))
                fmax_sems.append(np.std(bin_fmaxes) / np.sqrt(len(bin_fmaxes)))
                ont_stats[b_idx][mname] = np.mean(bin_fmaxes)
            else:
                fmax_means.append(np.nan); fmax_sems.append(0.0)
                ont_stats[b_idx][mname] = np.nan

        offset = (mi - n_models/2 + 0.5) * bar_width
        alpha_val = 0.5 if mname == 'Hybrid_JK' else 0.9
        edge_ls = '--' if mname in ('Hybrid_JK', 'BLAST', 'DIAMOND') else '-'
        hatch = '///' if mname == 'DeepFRI_Cmap' else None
        ax.bar(x_positions + offset, fmax_means, bar_width,
               yerr=fmax_sems, capsize=3,
               color=PALETTE.get(mname, '#888888'),
               label=mname, alpha=alpha_val, edgecolor='white', linestyle=edge_ls, hatch=hatch,
               error_kw={'linewidth': 1.0, 'ecolor': '#333333'})
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([b[2] for b in active_bins])
    ax.set_xlabel('Max Sequence Identity to Training Set')
    ax.set_ylabel('Micro Fmax')
    ax.set_title('a   Performance on Difficult Proteins (Seq Identity Bins)', loc='left', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc='upper left')
    style_ax(ax)
    
    print("NOTE: >0.6 bin has elevated identity due to MMseqs2 bilateral coverage limitation (see Methods).")

    plt.tight_layout()
    save_fig('plot_A_seq_identity')
    pd.DataFrame(ont_stats).to_csv(os.path.join(OUT_DIR, 'plot_A_seq_identity_stats.csv'), index=False)


# ── PLOT C: Fmax vs IC bins ───────────────────────────────────────────────────

def plot_C_ic_bins(datasets):
    """Per-GO-term IC bins: shows which methods handle rare (high-IC) terms."""
    bins = [(0, 2, 'IC 0–2'), (2, 4, 'IC 2–4'), (4, 6, 'IC 4–6'), (6, 99, 'IC >6')]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax_idx, (ont_full, ont_short) in enumerate(ONTOLOGIES.items()):
        ax = axes[ax_idx]
        test_ds  = datasets[ont_full]['test']
        prot_list = test_ds.pdb_split_list
        goterms   = test_ds.y_labels
        y_true    = _load_y_true(ont_short, datasets)
        
        valid_mask = _load_valid_mask(ont_short)
        if valid_mask is not None:
            y_true = y_true[:, valid_mask]
            goterms = [gt for gt, v in zip(goterms, valid_mask) if v]

        ic_raw    = _load_ic(ont_short, datasets)
        if ic_raw is not None and valid_mask is not None:
            ic_raw = ic_raw[:, valid_mask]
        ic = ic_raw if ic_raw is not None else compute_ic(y_true)
        preds, mask = load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
        if mask is not None:
            y_true = y_true[mask]
            # When subsetting test proteins, recompute IC from the subset's labels directly
            ic = compute_ic(y_true)
        active_bins = [b for b in bins if ((ic >= b[0]) & (ic < b[1])).sum() > 0]
        x_positions = np.arange(len(active_bins))
        bar_width   = 0.15
        n_models    = len(MODEL_ORDER)

        for mi, mname in enumerate(MODEL_ORDER):
            if mname not in preds:
                continue
            # Build per-seed predictions list
            if mname in ('Hybrid', 'Hybrid_JK'):
                seed_pairs = [(y_true, sp) for _, sp in load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list), mask=mask, valid_mask=valid_mask)]
                if not seed_pairs:
                    seed_pairs = [(y_true, preds[mname])]
            else:
                rng = np.random.default_rng(mi * 7)
                N = y_true.shape[0]
                seed_pairs = []
                for _ in range(5):
                    idx = rng.choice(N, N, replace=True)
                    seed_pairs.append((y_true[idx], preds[mname][idx]))

            fmax_means, fmax_sems = [], []
            for b_idx, (lo, hi, _) in enumerate(active_bins):
                term_mask = (ic >= lo) & (ic < hi)
                n_terms_in_bin = term_mask.sum()
                if n_terms_in_bin < 3:
                    fmax_means.append(np.nan); fmax_sems.append(0.0); continue
                bin_fmaxes = []
                for yt, yp in seed_pairs:
                    yt_b = yt[:, term_mask]
                    yp_b = yp[:, term_mask]
                    if yt_b.sum() == 0:
                        continue
                    prec, rec, _ = precision_recall_curve(yt_b.ravel(), yp_b.ravel())
                    f1 = 2*prec*rec/(prec+rec+1e-10)
                    bin_fmaxes.append(np.max(f1))
                if bin_fmaxes:
                    fmax_means.append(np.mean(bin_fmaxes))
                    fmax_sems.append(np.std(bin_fmaxes)/np.sqrt(len(bin_fmaxes)))
                else:
                    fmax_means.append(np.nan); fmax_sems.append(0.0)

            offset = (mi - n_models/2 + 0.5) * bar_width
            alpha_val = 0.5 if mname == 'Hybrid_JK' else 0.9
            edge_ls = '--' if mname == 'Hybrid_JK' else '-'
            hatch = '///' if mname == 'DeepFRI_Cmap' else None
            ax.bar(x_positions + offset, fmax_means, bar_width,
                   yerr=fmax_sems, capsize=3,
                   color=PALETTE.get(mname, '#888888'),
                   label=mname, alpha=alpha_val, edgecolor='white', linestyle=edge_ls, hatch=hatch,
                   error_kw={'linewidth': 1.0, 'ecolor': '#333333'})

        ax.set_xticks(x_positions)
        ax.set_xticklabels([b[2] for b in active_bins], rotation=20, ha='right')
        ax.set_xlabel('GO Term IC Value')
        ax.set_ylabel('Fmax' if ax_idx == 0 else '')
        ax.set_title(f'c   {ont_short.upper()}', loc='left', fontweight='bold')
        # Dynamic y-axis: fit to data with a small margin
        ax.autoscale(axis='y')
        ax.set_ylim(bottom=0)
        if ax_idx == 0:
            ax.legend(frameon=False, loc='upper right', fontsize=7)
        style_ax(ax)

    plt.suptitle('c   Performance on Rare GO Terms (IC Bins)', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig('plot_C_ic_bins')


# ── PLOT F: Fmax vs GO depth bins ─────────────────────────────────────────────

def _compute_go_depths():
    """
    Computes GO term depths using the go-basic OBO file.
    Returns dict: GO_ID → depth (int)
    Falls back gracefully if obo not found.
    """
    import glob
    obo_file = None
    search_dirs = [
        os.path.join(PROJECT_DIR, 'preprocessing', 'data'),
        PROJECT_DIR,
        os.path.join(PROJECT_DIR, 'SOTA', 'TransFun', 'data')
    ]
    for sdir in search_dirs:
        matches = glob.glob(os.path.join(sdir, 'go-basic*.obo'))
        if matches:
            obo_file = matches[0]
            break
            
    if obo_file is None:
        print('  [SKIP plot F] go-basic.obo not found. '
              'Download from http://purl.obolibrary.org/obo/go/go-basic.obo')
        return {}

    depths = {}
    current_id = None
    current_is_a = []
    id_to_parents = {}

    with open(obo_file) as f:
        in_term = False
        for line in f:
            line = line.strip()
            if line == '[Term]':
                in_term = True
                current_id = None
                current_is_a = []
            elif line == '' and in_term:
                if current_id:
                    id_to_parents[current_id] = current_is_a
                in_term = False
            elif in_term:
                if line.startswith('id: '):
                    current_id = line[4:].strip()
                elif line.startswith('is_a: '):
                    parent = line[6:].split('!')[0].strip()
                    current_is_a.append(parent)

    # BFS from roots
    from collections import deque
    roots = {gid for gid, parents in id_to_parents.items() if not parents}
    queue = deque((r, 0) for r in roots)
    depths = {r: 0 for r in roots}
    while queue:
        nid, depth = queue.popleft()
        # find children
        for child, parents in id_to_parents.items():
            if nid in parents and child not in depths:
                depths[child] = depth + 1
                queue.append((child, depth + 1))
    return depths


def plot_F_depth_bins(datasets):
    depths = _compute_go_depths()
    if not depths:
        return

    # Bins as in DPFunc paper
    bins = [(0, 4, 'depth ≤3'), (4, 6, 'depth 4–5'),
            (6, 8, 'depth 6–7'), (8, 999, 'depth ≥8')]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax_idx, (ont_full, ont_short) in enumerate(ONTOLOGIES.items()):
        ax = axes[ax_idx]
        test_ds  = datasets[ont_full]['test']
        prot_list = test_ds.pdb_split_list
        goterms   = test_ds.y_labels
        y_true    = _load_y_true(ont_short, datasets)

        valid_mask = _load_valid_mask(ont_short)
        if valid_mask is not None:
            y_true = y_true[:, valid_mask]
            goterms = [gt for gt, v in zip(goterms, valid_mask) if v]

        ic_raw    = _load_ic(ont_short, datasets)
        if ic_raw is not None and valid_mask is not None:
            ic_raw = ic_raw[:, valid_mask]
        ic = ic_raw if ic_raw is not None else compute_ic(y_true)
        preds, mask = load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
        if mask is not None:
            y_true = y_true[mask]

        term_depths = np.array([depths.get(t, 0) for t in goterms])
        active_bins = [b for b in bins if len([i for i, t in enumerate(goterms) if b[0] <= depths.get(t, 0) < b[1]]) > 0]
        x_positions = np.arange(len(active_bins))
        bar_width   = 0.15
        n_models    = len(MODEL_ORDER)

        for mi, mname in enumerate(MODEL_ORDER):
            if mname not in preds:
                continue
            if mname in ('Hybrid', 'Hybrid_JK'):
                seed_pairs = [(y_true, sp) for _, sp in load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list), mask=mask, valid_mask=valid_mask)]
                if not seed_pairs: seed_pairs = [(y_true, preds[mname])]
            else:
                rng = np.random.default_rng(mi * 13)
                N = y_true.shape[0]
                seed_pairs = []
                for _ in range(5):
                    idx = rng.choice(N, N, replace=True)
                    seed_pairs.append((y_true[idx], preds[mname][idx]))

            fmax_means, fmax_sems = [], []
            for b_idx, (lo, hi, _) in enumerate(active_bins):
                term_mask = (term_depths >= lo) & (term_depths < hi)
                if term_mask.sum() < 3:
                    fmax_means.append(np.nan); fmax_sems.append(0.0); continue
                bin_fmaxes = []
                for yt, yp in seed_pairs:
                    yt_b = yt[:, term_mask]
                    yp_b = yp[:, term_mask]
                    if yt_b.sum() == 0: continue
                    prec, rec, _ = precision_recall_curve(yt_b.ravel(), yp_b.ravel())
                    f1 = 2*prec*rec/(prec+rec+1e-10)
                    bin_fmaxes.append(np.max(f1))
                if bin_fmaxes:
                    fmax_means.append(np.mean(bin_fmaxes))
                    fmax_sems.append(np.std(bin_fmaxes)/np.sqrt(len(bin_fmaxes)))
                else:
                    fmax_means.append(np.nan); fmax_sems.append(0.0)

            offset = (mi - n_models/2 + 0.5) * bar_width
            alpha_val = 0.5 if mname == 'Hybrid_JK' else 0.9
            edge_ls = '--' if mname == 'Hybrid_JK' else '-'
            hatch = '///' if mname == 'DeepFRI_Cmap' else None
            ax.bar(x_positions + offset, fmax_means, bar_width,
                   yerr=fmax_sems, capsize=3,
                   color=PALETTE.get(mname, '#888888'), label=mname, 
                   alpha=alpha_val, edgecolor='white', linestyle=edge_ls, hatch=hatch,
                   error_kw={'linewidth': 1.0, 'ecolor': '#333333'})

        ax.set_xticks(x_positions)
        ax.set_xticklabels([b[2] for b in active_bins], rotation=20, ha='right')
        ax.set_xlabel('GO Term Depth')
        ax.set_ylabel('Fmax' if ax_idx == 0 else '')
        ax.set_title(f'f   {ont_short.upper()}', loc='left', fontweight='bold')
        ax.autoscale(axis='y')
        ax.set_ylim(bottom=0)
        if ax_idx == 0:
            ax.legend(frameon=False, loc='upper right', fontsize=7)
        style_ax(ax)

    plt.suptitle('f   Performance on Deep GO Terms', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig('plot_F_depth_bins')


# ── PLOTS B/D/E: IC-weighted PR curves per ontology ──────────────────────────

def ic_weighted_pr(y_true, y_pred, ic):
    """
    IC-weighted precision-recall curve.
    At each threshold t:
      weighted_prec = sum_{TP} ic(j) / sum_{Pred=1} ic(j)
      weighted_rec  = sum_{TP} ic(j) / sum_{True=1} ic(j)
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    precisions, recalls = [], []
    for t in thresholds:
        pred = (y_pred >= t).astype(bool)
        # Fix: operator precedence requires explicit parentheses before multiplying by ic
        tp_ic = np.sum(((y_true == 1) & pred) * ic)
        fp_ic = np.sum(((y_true == 0) & pred) * ic)
        fn_ic = np.sum(((y_true == 1) & ~pred) * ic)
        prec = tp_ic / (tp_ic + fp_ic + 1e-12)
        rec  = tp_ic / (tp_ic + fn_ic + 1e-12)
        precisions.append(prec)
        recalls.append(rec)
    return np.array(recalls), np.array(precisions)


def plot_BDE_pr_curves(datasets):
    """IC-weighted PR curves for MF (b), CC (d), BP (e)."""
    panel_map = {
        'molecular_function':  'b',
        'cellular_component':  'd',
        'biological_process':  'e',
    }

    for ont_full, ont_short in ONTOLOGIES.items():
        panel = panel_map[ont_full]
        test_ds  = datasets[ont_full]['test']
        prot_list = test_ds.pdb_split_list
        goterms   = test_ds.y_labels
        y_true    = _load_y_true(ont_short, datasets)

        valid_mask = _load_valid_mask(ont_short)
        if valid_mask is not None:
            y_true = y_true[:, valid_mask]
            goterms = [gt for gt, v in zip(goterms, valid_mask) if v]

        ic_raw    = _load_ic(ont_short, datasets)
        if ic_raw is not None and valid_mask is not None:
            ic_raw = ic_raw[:, valid_mask]
        ic = ic_raw if ic_raw is not None else compute_ic(y_true)
        ic_flat   = np.tile(ic, (len(prot_list), 1))  # broadcast for vectorised ops
        preds, mask = load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
        if mask is not None:
            y_true = y_true[mask]
            ic_flat = ic_flat[mask]

        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        for mname in MODEL_ORDER:
            if mname not in preds:
                continue
            # Average PR curve over seeds
            if mname in ('Hybrid', 'Hybrid_JK'):
                seed_pairs = [(y_true, ic_flat, sp) for _, sp in load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list), mask=mask, valid_mask=valid_mask)]
                if not seed_pairs:
                    seed_pairs = [(y_true, ic_flat, preds[mname])]
            else:
                rng = np.random.default_rng(hash(mname) % (2**31))
                N = y_true.shape[0]
                seed_pairs = []
                for _ in range(5):
                    idx = rng.choice(N, N, replace=True)
                    seed_pairs.append((y_true[idx], ic_flat[idx], preds[mname][idx]))

            all_rec, all_prec = [], []
            for yt, yt_ic_flat, yp in seed_pairs:
                rec, prec = ic_weighted_pr(yt, yp, yt_ic_flat)
                all_rec.append(rec)
                all_prec.append(prec)
            # Interpolate to common recall grid
            common_rec = np.linspace(0, 1, 200)
            interp_precs = []
            for rec, prec in zip(all_rec, all_prec):
                sort_idx = np.argsort(rec)
                interp_precs.append(np.interp(common_rec, rec[sort_idx], prec[sort_idx]))
            mean_prec = np.mean(interp_precs, axis=0)
            std_prec  = np.std(interp_precs, axis=0)

            color = PALETTE.get(mname, '#888888')
            ls = '--' if mname == 'Hybrid_JK' else '-'
            alpha_val = 0.6 if mname == 'Hybrid_JK' else 0.9
            ax.plot(common_rec, mean_prec, color=color, linewidth=1.8, label=mname, linestyle=ls, alpha=alpha_val)
            ax.fill_between(common_rec,
                            np.clip(mean_prec - std_prec, 0, 1),
                            np.clip(mean_prec + std_prec, 0, 1),
                            alpha=0.15 if mname != 'Hybrid_JK' else 0.05, color=color)

        ax.set_xlabel('IC-weighted Recall')
        ax.set_ylabel('IC-weighted Precision')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_title(f'{panel}   IC-weighted PR — {ont_short.upper()}', loc='left', fontweight='bold')
        ax.legend(frameon=False, loc='upper right')
        style_ax(ax)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.tight_layout()
        save_fig(f'plot_{panel.upper()}_pr_curve_{ont_short}')


    # ── PLOT G: Coverage ──────────────────────────────────────────────────────────

def plot_G_coverage(datasets):
    """Number of unique GO terms each method predicts (at t=0.5)."""
    threshold = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('g   Coverage of Predicted Functions', fontsize=12, fontweight='bold')

    for ax_idx, (ont_full, ont_short) in enumerate(ONTOLOGIES.items()):
        ax = axes[ax_idx]
        test_ds  = datasets[ont_full]['test']
        prot_list = test_ds.pdb_split_list
        goterms   = test_ds.y_labels
        y_true    = _load_y_true(ont_short, datasets)
        
        valid_mask = _load_valid_mask(ont_short)
        if valid_mask is not None:
            y_true = y_true[:, valid_mask]
            goterms = [gt for gt, v in zip(goterms, valid_mask) if v]
            
        preds, mask = load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
        if mask is not None:
            y_true = y_true[mask]

        n_total_terms = len(goterms)
        model_names, coverages = [], []

        for mname in MODEL_ORDER_COVERAGE:
            if mname not in preds:
                continue
            yp = preds[mname]
            # unique terms predicted for at least one protein
            predicted_any = (yp >= threshold).any(axis=0)
            coverage = int(predicted_any.sum())
            model_names.append(mname)
            coverages.append(coverage)

        colors = [PALETTE.get(m, '#888888') for m in model_names]
        bars = ax.bar(model_names, coverages, color=colors, edgecolor='white', linewidth=0.8)
        
        for i, mname in enumerate(model_names):
            if mname == 'DeepFRI_Cmap':
                bars[i].set_hatch('///')
            if mname == 'Hybrid_JK':
                bars[i].set_alpha(0.5)
                bars[i].set_linestyle('--')
                bars[i].set_linewidth(1.2)
                bars[i].set_edgecolor(colors[i])

        # Let the bars set the y-limit, but add some headroom
        max_coverage = max(coverages) if coverages else 1
        ax.set_ylim(0, max_coverage * 1.2)

        for bar, val in zip(bars, coverages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_coverage * 0.02,
                    str(val), ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{ont_short.upper()}', loc='left')
        ax.set_ylabel('Unique GO terms predicted' if ax_idx == 0 else '')
        ax.set_xticklabels(model_names, rotation=30, ha='right')
        
        # Display total terms in the legend instead of a horizontal line
        ax.plot([], [], ' ', label=f'Total terms in dataset: {n_total_terms}')
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        
        style_ax(ax)

    plt.tight_layout()
    save_fig('plot_G_coverage')


# ── BONUS: Summary bar chart (Micro Fmax, all models, all ontologies) ─────────

def plot_summary_fmax(results_csv):
    if not os.path.exists(results_csv):
        print(f'  [SKIP summary] {results_csv} not found.')
        return
    df = pd.read_csv(results_csv)
    ont_order = ['BP', 'MF', 'CC']
    metric = 'Micro_Fmax_mean'
    metric_se = 'Micro_Fmax_se'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Overall Micro Fmax Comparison (mean ± SE, 5 seeds/bootstraps)',
                 fontsize=12, fontweight='bold')

    for ax_idx, ont in enumerate(ont_order):
        ax = axes[ax_idx]
        sub = df[df['Ontology'] == ont]
        local_order = MODEL_ORDER
        sub = sub.set_index('Model').reindex(local_order).dropna(subset=[metric])

        colors = [PALETTE.get(m, '#888888') for m in sub.index]
        vals   = sub[metric].values
        errs   = sub[metric_se].values if metric_se in sub.columns else np.zeros(len(vals))

        bars = ax.bar(sub.index, vals, color=colors, yerr=errs, capsize=4,
                      edgecolor='white', linewidth=0.8,
                      error_kw={'linewidth': 1.0, 'ecolor': '#333333'})
                      
        for i, mname in enumerate(sub.index):
            if mname == 'DeepFRI_Cmap':
                bars[i].set_hatch('///')
            if mname == 'Hybrid_JK':
                bars[i].set_alpha(0.5)
                bars[i].set_linestyle('--')
                bars[i].set_linewidth(1.2)
                bars[i].set_edgecolor(colors[i])
                
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_title(ont, fontweight='bold')
        ax.set_ylabel('Micro Fmax' if ax_idx == 0 else '')
        # Dynamic y-axis fitted to data range with margin
        if len(vals) > 0:
            lo = max(0, vals.min() - 0.05)
            hi = min(1.0, vals.max() + 0.08)
            ax.set_ylim(lo, hi)
        ax.set_xticklabels(sub.index, rotation=30, ha='right')
        style_ax(ax)

    plt.tight_layout()
    fig.text(0.5, -0.05, 
        'Note: DeepGreenGO error bars = training variance (5 seeds); '
        'SOTA error bars = bootstrap variance (5 resamples)\n'
        '*DeepFRI_Cmap MF performance is heavily elevated by high sequence-identity test proteins.',
        ha='center', fontsize=7, style='italic', color='#666666')
    save_fig('plot_summary_fmax')


# ── entry point ───────────────────────────────────────────────────────────────

args = None

def main():
    global args, MODEL_ORDER, MODEL_ORDER_COVERAGE
    parser = argparse.ArgumentParser()
    parser.add_argument('--common_subset', action='store_true')
    parser.add_argument('--supplementary', action='store_true', help="Generate supplementary figures with Hybrid_JK")
    parser.add_argument('--mode', type=str, default='dl_only', choices=['dl_only', 'baselines_only', 'all'])
    args = parser.parse_args()
    
    global OUT_DIR
    if args.mode != 'dl_only':
        OUT_DIR = os.path.join(PROJECT_DIR, f'plots_sota_comparison_{args.mode}')
    else:
        OUT_DIR = os.path.join(PROJECT_DIR, 'plots_sota_comparison')
    os.makedirs(OUT_DIR, exist_ok=True)

    
    if args.mode == 'baselines_only':
        MODEL_ORDER = ['Hybrid_JK', 'Hybrid', 'BLAST', 'DIAMOND', 'Naive']
    elif args.mode == 'all':
        MODEL_ORDER = ['Hybrid_JK', 'Hybrid', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap', 'BLAST', 'DIAMOND', 'Naive']
    else:
        if args.supplementary:
            MODEL_ORDER = MODEL_ORDER_SUPPLEMENTARY
        else:
            MODEL_ORDER = MODEL_ORDER_PERFORMANCE

    if args.supplementary:
        MODEL_ORDER_COVERAGE = ['Hybrid', 'Hybrid_JK', 'DPFunc', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']
    else:
        MODEL_ORDER_COVERAGE = ['Hybrid', 'DPFunc', 'TransFun', 'DeepFRI_Seq', 'DeepFRI_Cmap']

    print(f'Loading datasets ...')
    datasets = load_datasets()

    print('\n[Plot A] Sequence identity bins ...')
    plot_A_sequence_identity(datasets)

    print('\n[Plot C] IC bins ...')
    plot_C_ic_bins(datasets)

    print('\n[Plots B/D/E] IC-weighted PR curves ...')
    plot_BDE_pr_curves(datasets)

    print('\n[Plot F] GO depth bins ...')
    plot_F_depth_bins(datasets)

    print('\n[Plot G] Coverage ...')
    plot_G_coverage(datasets)

    print('\n[Summary] Fmax bar chart ...')
    if args.common_subset:
        plot_summary_fmax(os.path.join(RESULTS_DIR, 'evaluation_results_common_subset.csv'))
    else:
        plot_summary_fmax(os.path.join(RESULTS_DIR, 'evaluation_results.csv'))

    print(f'\n✓ All plots saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
