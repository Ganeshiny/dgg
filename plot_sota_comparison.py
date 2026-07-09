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
    'Hybrid_JK': '#EE3377',   # magenta  — your best model
    'Hybrid':    '#CC3311',   # red
    'TransFun':  '#0077BB',   # blue
    'DPFunc':    '#009988',   # teal
    'DeepFRI':   '#EE7733',   # orange
}
MODEL_ORDER = ['Hybrid_JK', 'Hybrid', 'DPFunc', 'TransFun', 'DeepFRI']

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
    path = os.path.join(OUT_DIR, f'{name}.png')
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

def _load_y_true(ont_short):
    """Load test_y_true.npy from any completed run for this ontology."""
    for runs_dir in ('tuning_runs_jk', 'tuning_runs', 'runs_jk_test', 'runs_5seeds'):
        search_root = os.path.join(PROJECT_DIR, runs_dir)
        if not os.path.isdir(search_root):
            continue
        for root, dirs, files in os.walk(search_root):
            if 'test_y_true.npy' in files:
                if os.path.basename(root).startswith(ont_short + '_'):
                    return np.load(os.path.join(root, 'test_y_true.npy'))
    return None

def _load_ic(ont_short):
    """Load pre-computed IC from saved train_labels.npy."""
    candidates = [
        os.path.join(PROJECT_DIR, f'{ont_short}_train_labels.npy'),
        os.path.join(PROJECT_DIR, f'{_FULL_ONT[ont_short]}_train_labels.npy'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return compute_ic(np.load(c))
    return None

def compute_ic(y_train):
    N = y_train.shape[0]
    counts = np.sum(y_train, axis=0)
    ic = np.zeros(counts.shape, dtype=float)
    m = counts > 0
    ic[m] = -np.log2(counts[m] / N)
    return ic


# ── prediction loaders ────────────────────────────────────────────────────────

def _txt_to_matrix(path, prot_list, goterms):
    """Load space-separated 'PROT TERM SCORE' file."""
    y = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    if not os.path.exists(path):
        print(f'  [MISSING] {path}')
        return y
    pi = {p: i for i, p in enumerate(prot_list)}
    ti = {t: i for i, t in enumerate(goterms)}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and parts[0] in pi and parts[1] in ti:
                y[pi[parts[0]], ti[parts[1]]] = float(parts[2])
    return y

def _deepfri_to_matrix(path, prot_list, goterms):
    if not os.path.exists(path):
        print(f'  [MISSING] {path}')
        return np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    with open(path) as f:
        d = json.load(f)
    chains = d['pdb_chains']
    dterms = d['goterms']
    Yhat   = np.array(d['Y_hat'], dtype=np.float32)
    y = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    pi = {p: i for i, p in enumerate(prot_list)}
    ti = {t: i for i, t in enumerate(goterms)}
    dti = {t: i for i, t in enumerate(dterms)}
    for di, dp in enumerate(chains):
        if dp not in pi:
            continue
        row = Yhat[di]
        for dt, dj in dti.items():
            if dt in ti:
                y[pi[dp], ti[dt]] = row[dj]
    return y

def load_predictions(ont_full, ont_short, prot_list, goterms):
    """Returns dict model→y_pred matrix (averaged over seeds for your models)."""
    preds = {}

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

    # TransFun
    tf_file = _find_robust([f'{ont_short}_results.txt'], 'TransFun', ['SOTA', 'SOTA_predictions'])
    if not tf_file: tf_file = os.path.join(PROJECT_DIR, 'SOTA', 'TransFun', 'data', f'{ont_short}_results.txt')
    preds['TransFun'] = _txt_to_matrix(tf_file, prot_list, goterms)

    # DeepFRI
    df_variants = [f'deepfri_seq_{ont_short.upper()}_pred_scores.json', f'deepfri_seq_{ont_short.upper()}_{ont_short.upper()}_pred_scores.json']
    df_file = _find_robust(df_variants, 'deepfri', ['baselines', 'baselines/deepfri_results'])
    if not df_file: df_file = os.path.join(PROJECT_DIR, 'baselines', 'deepfri_results', f'deepfri_seq_{ont_short.upper()}_pred_scores.json')
    preds['DeepFRI'] = _deepfri_to_matrix(df_file, prot_list, goterms)

    # DPFunc
    dpf_file = _find_robust([f'{ont_short}_results.txt'], 'DPFunc', ['SOTA_predictions', 'SOTA'])
    if not dpf_file: dpf_file = os.path.join(PROJECT_DIR, 'SOTA_predictions', 'DPFunc', f'{ont_short}_results.txt')
    preds['DPFunc'] = _txt_to_matrix(dpf_file, prot_list, goterms)

    # Your models — average over available seeds
    for mname in ('Hybrid', 'Hybrid_JK'):
        seed_preds_list = load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list))
        seed_preds = [arr for s, arr in seed_preds_list]
        if seed_preds:
            preds[mname] = np.mean(np.stack(seed_preds, axis=0), axis=0)
        else:
            preds[mname] = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
            print(f'  [MISSING] No seeds found for {mname}/{ont_short}')

    return preds

def load_per_seed_preds(ont_short, mname, goterms_len, n_prots):
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
            out.append((s, np.load(pred_path)))
    return out


# ── PLOT A: Fmax vs sequence identity bins ────────────────────────────────────

def plot_A_sequence_identity(datasets):
    """
    Requires preprocessing/data/blast_identity.csv with columns:
        prot, max_identity  (float 0-1)
    Generated by: preprocessing/calc_blast_identity.sh
    """
    blast_csv = os.path.join(PROJECT_DIR, 'preprocessing', 'data', 'blast_identity.csv')
    if not os.path.exists(blast_csv):
        print(f'  [SKIP plot A] {blast_csv} not found.')
        return

    id_df = pd.read_csv(blast_csv)   # columns: prot, max_identity
    id_map = dict(zip(id_df['prot'], id_df['max_identity']))

    bins   = [(0, 0.2, '<20%'), (0.2, 0.4, '20–40%'),
              (0.4, 0.6, '40–60%'), (0.6, 1.01, '>60%')]

    # We use BP as the representative ontology for this figure (like DPFunc paper)
    ont_full, ont_short = 'biological_process', 'bp'
    test_ds  = datasets[ont_full]['test']
    prot_list = test_ds.pdb_split_list
    goterms   = test_ds.y_labels
    y_true    = _load_y_true(ont_short)
    if y_true is None:
        print('  [SKIP plot A] No cached y_true for bp.')
        return
    ic_raw    = _load_ic(ont_short)
    ic        = ic_raw if ic_raw is not None else compute_ic(y_true)
    preds     = load_predictions(ont_full, ont_short, prot_list, goterms)

    # Assign each test protein to an identity bin
    prot_identity = np.array([id_map.get(p, 1.0) for p in prot_list])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_positions = np.arange(len(bins))
    bar_width   = 0.15
    n_models    = len(MODEL_ORDER)

    for mi, mname in enumerate(MODEL_ORDER):
        if mname not in preds:
            continue
        # per-seed runs for your models, bootstrap for SOTA
        seed_preds_list = []
        if mname in ('Hybrid', 'Hybrid_JK'):
            for _, sp in load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list)):
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
        for lo, hi, _ in bins:
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
            else:
                fmax_means.append(np.nan); fmax_sems.append(0.0)

        offset = (mi - n_models/2 + 0.5) * bar_width
        ax.bar(x_positions + offset, fmax_means, bar_width,
               yerr=fmax_sems, capsize=3,
               color=PALETTE.get(mname, '#888888'),
               label=mname, alpha=0.9,
               error_kw={'linewidth': 1.0, 'ecolor': '#333333'})

    ax.set_xticks(x_positions)
    ax.set_xticklabels([b[2] for b in bins])
    ax.set_xlabel('Max Sequence Identity to Training Set')
    ax.set_ylabel('Micro Fmax')
    ax.set_title('a   Performance on Difficult Proteins (Seq Identity Bins)', loc='left', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc='upper left')
    style_ax(ax)
    plt.tight_layout()
    save_fig('plot_A_seq_identity')


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
        y_true    = _load_y_true(ont_short)

        ic_raw    = _load_ic(ont_short); ic = ic_raw if ic_raw is not None else compute_ic(y_true)
        preds     = load_predictions(ont_full, ont_short, prot_list, goterms)

        x_positions = np.arange(len(bins))
        bar_width   = 0.15
        n_models    = len(MODEL_ORDER)

        for mi, mname in enumerate(MODEL_ORDER):
            if mname not in preds:
                continue
            # Build per-seed predictions list
            if mname in ('Hybrid', 'Hybrid_JK'):
                seed_ps = [sp for _, sp in load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list))]
                if not seed_ps:
                    seed_ps = [preds[mname]]
            else:
                rng = np.random.default_rng(mi * 7)
                N = y_true.shape[0]
                seed_ps = [preds[mname][rng.choice(N, N, replace=True)] for _ in range(5)]

            fmax_means, fmax_sems = [], []
            for lo, hi, _ in bins:
                term_mask = (ic >= lo) & (ic < hi)
                n_terms_in_bin = term_mask.sum()
                if n_terms_in_bin < 3:
                    fmax_means.append(np.nan); fmax_sems.append(0.0); continue
                bin_fmaxes = []
                for sp in seed_ps:
                    yt_b = y_true[:, term_mask]
                    yp_b = sp[:, term_mask]
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
            ax.bar(x_positions + offset, fmax_means, bar_width,
                   yerr=fmax_sems, capsize=3,
                   color=PALETTE.get(mname, '#888888'),
                   label=mname, alpha=0.9,
                   error_kw={'linewidth': 1.0, 'ecolor': '#333333'})

        ax.set_xticks(x_positions)
        ax.set_xticklabels([b[2] for b in bins], rotation=20, ha='right')
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
        y_true    = _load_y_true(ont_short)

        ic_raw    = _load_ic(ont_short); ic = ic_raw if ic_raw is not None else compute_ic(y_true)
        preds     = load_predictions(ont_full, ont_short, prot_list, goterms)

        term_depths = np.array([depths.get(t, 0) for t in goterms])
        x_positions = np.arange(len(bins))
        bar_width   = 0.15
        n_models    = len(MODEL_ORDER)

        for mi, mname in enumerate(MODEL_ORDER):
            if mname not in preds:
                continue
            if mname in ('Hybrid', 'Hybrid_JK'):
                seed_ps = [sp for _, sp in load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list))]
                if not seed_ps: seed_ps = [preds[mname]]
            else:
                rng = np.random.default_rng(mi * 13)
                N = y_true.shape[0]
                seed_ps = [preds[mname][rng.choice(N, N, replace=True)] for _ in range(5)]

            fmax_means, fmax_sems = [], []
            for lo, hi, _ in bins:
                term_mask = (term_depths >= lo) & (term_depths < hi)
                if term_mask.sum() < 3:
                    fmax_means.append(np.nan); fmax_sems.append(0.0); continue
                bin_fmaxes = []
                for sp in seed_ps:
                    yt_b = y_true[:, term_mask]
                    yp_b = sp[:, term_mask]
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
            ax.bar(x_positions + offset, fmax_means, bar_width,
                   yerr=fmax_sems, capsize=3,
                   color=PALETTE.get(mname, '#888888'), label=mname, alpha=0.9,
                   error_kw={'linewidth': 1.0, 'ecolor': '#333333'})

        ax.set_xticks(x_positions)
        ax.set_xticklabels([b[2] for b in bins], rotation=20, ha='right')
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
        y_true    = _load_y_true(ont_short)

        ic_raw    = _load_ic(ont_short); ic = ic_raw if ic_raw is not None else compute_ic(y_true)
        ic_flat   = np.tile(ic, (len(prot_list), 1))  # broadcast for vectorised ops
        preds     = load_predictions(ont_full, ont_short, prot_list, goterms)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))

        for mname in MODEL_ORDER:
            if mname not in preds:
                continue
            # Average PR curve over seeds
            if mname in ('Hybrid', 'Hybrid_JK'):
                seed_ps = [sp for _, sp in load_per_seed_preds(ont_short, mname, len(goterms), len(prot_list))]
                if not seed_ps:
                    seed_ps = [preds[mname]]
            else:
                rng = np.random.default_rng(hash(mname) % (2**31))
                N = y_true.shape[0]
                seed_ps = [preds[mname][rng.choice(N, N, replace=True)] for _ in range(5)]

            all_rec, all_prec = [], []
            for sp in seed_ps:
                rec, prec = ic_weighted_pr(y_true, sp, ic_flat)
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
            ax.plot(common_rec, mean_prec, color=color, linewidth=1.8, label=mname)
            ax.fill_between(common_rec,
                            np.clip(mean_prec - std_prec, 0, 1),
                            np.clip(mean_prec + std_prec, 0, 1),
                            alpha=0.15, color=color)

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
        preds     = load_predictions(ont_full, ont_short, prot_list, goterms)

        n_total_terms = len(goterms)
        model_names, coverages = [], []

        for mname in MODEL_ORDER:
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
        sub = sub.set_index('Model').reindex(MODEL_ORDER).dropna(subset=[metric])

        colors = [PALETTE.get(m, '#888888') for m in sub.index]
        vals   = sub[metric].values
        errs   = sub[metric_se].values if metric_se in sub.columns else np.zeros(len(vals))

        bars = ax.bar(sub.index, vals, color=colors, yerr=errs, capsize=4,
                      edgecolor='white', linewidth=0.8,
                      error_kw={'linewidth': 1.0, 'ecolor': '#333333'})
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
    save_fig('plot_summary_fmax')


# ── entry point ───────────────────────────────────────────────────────────────

def main():
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
    plot_summary_fmax(os.path.join(RESULTS_DIR, 'evaluation_results.csv'))

    print(f'\n✓ All plots saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
