"""
preprocessing/pdb_clusters/analyse_and_plot.py
===============================================
Deep-dive analysis of all PDB-cluster-based splits.

Produces:
  1.  Summary table: split sizes, achieved fracs, GO coverage per ontology.
  2.  Protein count per split per threshold (stacked bar).
  3.  Achieved split fractions vs targets.
  4.  GO term coverage heatmap (train/valid/test x ontologies x thresholds).
  5.  GO protein coverage per ontology.
  6.  GO term frequency CDF per split (for DeepGreenGO comparison).
  7.  Rare GO term risk per threshold.
  8.  Super-cluster counts per threshold.
  9.  BLAST identity distribution (test→train) for each threshold.
      (Requires blastp; if not available, this section is skipped.)
  10. DeepGreenGO vs PDB-cluster split leakage comparison.
  11. Dataset deep-dive (sequence lengths, label distributions, rank plots).
  12. GO term overlap between train and test per threshold.

Usage:
    python3 preprocessing/pdb_clusters/analyse_and_plot.py [--skip_blast]
"""

import json
import os
import pickle
import subprocess
import sys
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from common import DATA_DIR, SPLIT_ROOT, THRESHOLDS
except ImportError:
    from preprocessing.pdb_clusters.common import DATA_DIR, SPLIT_ROOT, THRESHOLDS
PLOTS_DIR = Path(__file__).resolve().parent / 'plots'
ONT_LABELS  = {
    'molecular_function': 'MF',
    'biological_process': 'BP',
    'cellular_component': 'CC',
}
SPLIT_COLORS = {'train': '#4C72B0', 'valid': '#DD8452', 'test': '#55A868'}

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_log(t: int) -> dict | None:
    p = SPLIT_ROOT / f'threshold_{t}' / 'split_log.json'
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def load_go_coverage(t: int) -> dict | None:
    p = SPLIT_ROOT / f'threshold_{t}' / 'go_coverage_full.pkl'
    if not p.exists():
        return None
    with open(p, 'rb') as fh:
        return pickle.load(fh)


def load_split_ids(t: int, split: str) -> list[str]:
    p = SPLIT_ROOT / f'threshold_{t}' / f'_{split}.txt'
    if not p.exists():
        return []
    with open(p) as fh:
        return [l.strip() for l in fh if l.strip()]


def load_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    key = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                key = line[1:].split()[0]
            elif key:
                seqs[key] = seqs.get(key, '') + line
    return seqs


def load_go_table() -> dict[str, dict[str, list[str]]]:
    pdb2go_path = DATA_DIR / 'pdb2go.tsv'
    with open(pdb2go_path) as fh:
        content = fh.read()
    sections = content.split('###')
    pdb_section = sections[7]
    lines = pdb_section.strip().split('\n')
    ONT_MAP = {1: 'molecular_function', 2: 'biological_process', 3: 'cellular_component'}
    go_table: dict[str, dict[str, list[str]]] = {}
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        chain_id = parts[0]
        go_table[chain_id] = {
            ont: (parts[col].strip().split(',') if parts[col].strip() else [])
            for col, ont in ONT_MAP.items()
        }
    return go_table


def collect_available_thresholds() -> list[int]:
    return [t for t in THRESHOLDS if load_log(t) is not None]


def spine_clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25)


# ── Figure 1: Summary table ──────────────────────────────────────────────────

def plot_summary_table(available: list[int]) -> None:
    rows = []
    for t in available:
        log = load_log(t)
        if log is None:
            continue
        row = {'threshold': t}
        for split in ['train', 'valid', 'test']:
            row[f'{split}_n'] = log['split_sizes'][split]
            row[f'{split}_frac'] = log['achieved_fracs'][split]
        row['n_clusters'] = log['n_super_clusters']
        row['unclustered'] = log['unclustered_chains']
        rows.append(row)

    if not rows:
        print('  No data for summary table.')
        return

    col_labels = [
        'Threshold',
        'Train N', 'Train %',
        'Valid N', 'Valid %',
        'Test N',  'Test %',
        'Super-clusters', 'Unclustered',
    ]
    table_data = [[
        f"{r['threshold']}%",
        f"{r['train_n']:,}", f"{r['train_frac']:.1%}",
        f"{r['valid_n']:,}", f"{r['valid_frac']:.1%}",
        f"{r['test_n']:,}",  f"{r['test_frac']:.1%}",
        f"{r['n_clusters']:,}", f"{r['unclustered']:,}",
    ] for r in rows]

    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.55 * len(rows)))
    ax.axis('off')
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#2C3E50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for ri in range(1, len(rows) + 1):
        bg = '#F5F5F5' if ri % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[(ri, j)].set_facecolor(bg)

    fig.suptitle('PDB-Cluster Split Summary (all thresholds)', fontsize=12,
                 fontweight='bold', y=1.02)
    out = PLOTS_DIR / '01_summary_table.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 2: Protein count per split per threshold ──────────────────────────

def plot_protein_counts(available: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(available))
    width = 0.25

    for i, split in enumerate(['train', 'valid', 'test']):
        counts = []
        for t in available:
            log = load_log(t)
            counts.append(log['split_sizes'][split] if log else 0)
        bars = ax.bar(x + (i - 1) * width, counts, width,
                      label=split.capitalize(), color=SPLIT_COLORS[split], alpha=0.85)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                    f'{cnt:,}', ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}%' for t in available])
    ax.set_xlabel('Sequence Identity Threshold', fontsize=11)
    ax.set_ylabel('Number of Proteins', fontsize=11)
    ax.set_title('Protein Count per Split × Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    spine_clean(ax)
    out = PLOTS_DIR / '02_protein_counts.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 3: Achieved split fractions vs targets ────────────────────────────

def plot_achieved_fracs(available: list[int]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    target = {'train': 0.80, 'valid': 0.10, 'test': 0.10}

    for ax, split in zip(axes, ['train', 'valid', 'test']):
        fracs = [load_log(t)['achieved_fracs'][split] if load_log(t) else 0
                 for t in available]
        ax.bar([f'{t}%' for t in available], fracs,
               color=SPLIT_COLORS[split], alpha=0.8)
        ax.axhline(target[split], color='red', linestyle='--', linewidth=1.5,
                   label=f'Target {target[split]:.0%}')
        ax.set_title(f'{split.capitalize()} Split', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fraction of proteins')
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=9)
        spine_clean(ax)

    fig.suptitle('Achieved vs Target Split Fractions (Bin-packed by Protein Count)',
                 fontsize=12, fontweight='bold')
    out = PLOTS_DIR / '03_achieved_fracs.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 4: GO term coverage heatmap ───────────────────────────────────────

def plot_go_coverage_heatmap(available: list[int]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    onts = list(ONT_LABELS.keys())
    splits = ['train', 'valid', 'test']

    for ax, ont in zip(axes, onts):
        matrix = np.zeros((len(available), 3))
        for ri, t in enumerate(available):
            cov = load_go_coverage(t)
            if cov is None:
                continue
            for ci, split in enumerate(splits):
                matrix[ri, ci] = cov[split][ont]['n_unique_terms']

        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Train', 'Valid', 'Test'], fontsize=10)
        ax.set_yticks(range(len(available)))
        ax.set_yticklabels([f'{t}%' for t in available], fontsize=10)
        ax.set_title(f'{ONT_LABELS[ont]} — Unique GO Terms', fontsize=11, fontweight='bold')

        for ri in range(len(available)):
            for ci in range(3):
                val = int(matrix[ri, ci])
                colour = 'white' if matrix[ri, ci] > matrix.max() * 0.65 else 'black'
                ax.text(ci, ri, f'{val:,}', ha='center', va='center', fontsize=8, color=colour)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Unique GO Terms per Split × Threshold', fontsize=13, fontweight='bold')
    out = PLOTS_DIR / '04_go_coverage_heatmap.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 5: GO protein coverage per ontology ───────────────────────────────

def plot_go_protein_coverage(available: list[int]) -> None:
    onts = list(ONT_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(available))
    width = 0.25

    for ax, ont in zip(axes, onts):
        for i, split in enumerate(['train', 'valid', 'test']):
            counts = []
            for t in available:
                cov = load_go_coverage(t)
                counts.append(cov[split][ont]['n_proteins_with_labels'] if cov else 0)
            ax.bar(x + (i - 1) * width, counts, width,
                   label=split.capitalize(), color=SPLIT_COLORS[split], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}%' for t in available], fontsize=9)
        ax.set_title(f'{ONT_LABELS[ont]} — Proteins with Labels', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of proteins', fontsize=10)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
        spine_clean(ax)

    fig.suptitle('Proteins with GO Labels per Split × Ontology × Threshold',
                 fontsize=12, fontweight='bold')
    out = PLOTS_DIR / '05_go_protein_coverage.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 6: GO term frequency distribution (CDF) ───────────────────────────

def plot_go_term_freq_cdf(available: list[int]) -> None:
    onts = list(ONT_LABELS.keys())
    n_thresh = len(available)
    fig, axes = plt.subplots(n_thresh, 3, figsize=(15, 4 * n_thresh), squeeze=False)

    for ri, t in enumerate(available):
        cov = load_go_coverage(t)
        for ci, ont in enumerate(onts):
            ax = axes[ri][ci]
            if cov is None:
                ax.set_visible(False)
                continue
            for split in ['train', 'valid', 'test']:
                term_counts = list(cov[split][ont].get('term_counts', {}).values())
                if not term_counts:
                    continue
                sorted_counts = np.sort(term_counts)
                cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
                ax.plot(sorted_counts, cdf, label=split.capitalize(),
                        color=SPLIT_COLORS[split], linewidth=1.8)
            ax.set_xscale('log')
            ax.set_xlabel('Proteins per GO term', fontsize=9)
            ax.set_ylabel('CDF', fontsize=9)
            ax.set_title(f'{t}% | {ONT_LABELS[ont]}', fontsize=10, fontweight='bold')
            ax.axvline(5, color='grey', linestyle=':', alpha=0.7, linewidth=1)
            ax.legend(fontsize=8)
            spine_clean(ax)

    fig.suptitle('GO Term Frequency CDF per Split × Threshold\n(grey dashed = 5-protein cutoff)',
                 fontsize=13, fontweight='bold')
    out = PLOTS_DIR / '06_go_freq_cdf.png'
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out, dpi=130)
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 7: Rare GO term overlap (train ∩ test) ───────────────────────────

def plot_rare_term_overlap(available: list[int]) -> None:
    onts = list(ONT_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, ont in zip(axes, onts):
        coverages = []
        for t in available:
            cov = load_go_coverage(t)
            if cov is None:
                coverages.append(np.nan)
                continue
            test_terms = set(cov['test'][ont].get('term_counts', {}).keys())
            rare_train = {term for term, cnt in cov['train'][ont].get('term_counts', {}).items()
                          if cnt <= 5}
            frac = len(test_terms & rare_train) / len(test_terms) if test_terms else 0
            coverages.append(frac)

        ax.bar([f'{t}%' for t in available], coverages, color='#C44E52', alpha=0.8)
        ax.set_xlabel('Threshold', fontsize=10)
        ax.set_ylabel('Fraction of test GO terms\nthat are rare in train (≤5 prots)', fontsize=9)
        ax.set_title(f'{ONT_LABELS[ont]}', fontsize=11, fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.05)
        spine_clean(ax)

    fig.suptitle('Rare GO Term Risk: Fraction of Test Terms That Are Rare in Train (≤5 proteins)',
                 fontsize=12, fontweight='bold')
    out = PLOTS_DIR / '07_rare_term_risk.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 8: Super-cluster size distribution ─────────────────────────────────

def plot_supercluster_sizes(available: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10

    for i, t in enumerate(available):
        log = load_log(t)
        if log is None:
            continue
        n_sc  = log['n_super_clusters']
        total = log['total_proteins']
        ax.scatter(t, n_sc, s=200, color=cmap(i / len(available)), zorder=3,
                   label=f'{t}%: {n_sc:,} clusters ({total:,} prots)')
        ax.annotate(f'{n_sc:,}', (t, n_sc), textcoords='offset points',
                    xytext=(5, 5), fontsize=9)

    ax.set_xlabel('Threshold (%)', fontsize=11)
    ax.set_ylabel('Number of Super-Clusters', fontsize=11)
    ax.set_title(
        'Number of Super-Clusters per Threshold\n'
        '(after union-find over PDB ID and cluster membership)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
    spine_clean(ax)

    out = PLOTS_DIR / '08_supercluster_counts.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 9: BLAST identity distribution ────────────────────────────────────

def run_blast_analysis(available: list[int], skip_blast: bool = False) -> None:
    import shutil
    if not shutil.which('blastp') or not shutil.which('makeblastdb'):
        print('  [SKIP] blastp/makeblastdb not found; leakage report unavailable.')
        return

    blast_results: dict[int, list[float]] = {}
    leakage_report: dict[str, dict] = {}

    for t in available:
        split_dir = SPLIT_ROOT / f'threshold_{t}'
        train_fa = split_dir / '_train_sequences.fasta'
        test_fa = split_dir / '_test_sequences.fasta'
        blast_db = split_dir / 'blast_train_db'
        blast_out = split_dir / 'blast_te_vs_tr.tsv'
        if not train_fa.exists() or not test_fa.exists():
            print(f'  [SKIP] Missing FASTA for threshold {t}%')
            continue

        if not skip_blast or not blast_out.exists():
            subprocess.run([
                'makeblastdb', '-in', str(train_fa), '-dbtype', 'prot', '-out', str(blast_db),
            ], check=True, capture_output=True)
            subprocess.run([
                'blastp', '-query', str(test_fa), '-db', str(blast_db),
                '-out', str(blast_out),
                '-outfmt', '6 qseqid sseqid pident length qlen slen',
                '-num_threads', '4', '-evalue', '1e-3', '-max_hsps', '1', '-qcov_hsp_perc', '80',
            ], check=True, capture_output=True)

        max_id: dict[str, float] = {}
        qualified_id: dict[str, float] = {}
        best_cov: dict[str, float] = {}
        with open(blast_out) as fh:
            for line in fh:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                query, pident = parts[0], float(parts[2])
                aln_len, qlen, slen = map(float, parts[3:6])
                coverage = aln_len / max(min(qlen, slen), 1.0)
                if query not in max_id or pident > max_id[query]:
                    max_id[query] = pident
                    best_cov[query] = coverage
                if coverage >= 0.80 and (query not in qualified_id or pident > qualified_id[query]):
                    qualified_id[query] = pident

        test_ids = load_split_ids(t, 'test')
        values = [max_id.get(pid, 0.0) for pid in test_ids]
        qualified_values = [qualified_id.get(pid, 0.0) for pid in test_ids]
        blast_results[t] = values
        leakage_report[str(t)] = {
            'threshold_percent': t,
            'n_test_sequences': len(test_ids),
            'n_test_queries_with_hits': len(max_id),
            'max_identity_percent': max(values) if values else 0.0,
            'mean_max_identity_percent': float(np.mean(values)) if values else 0.0,
            'fraction_at_or_above_60_percent': (
                sum(v >= 60.0 for v in values) / len(values) if values else 0.0
            ),
            'fraction_at_or_above_cluster_threshold': (
                sum(v >= t for v in values) / len(values) if values else 0.0
            ),
            'fraction_at_or_above_threshold_and_80pct_coverage': (
                sum(v >= t for v in qualified_values) / len(qualified_values) if qualified_values else 0.0
            ),
            'max_query_subject_coverage': max(best_cov.values()) if best_cov else 0.0,
        }

    if not blast_results:
        return
    (DATA_DIR / 'blast_leakage.json').write_text(json.dumps(leakage_report, indent=2) + '\n')

    fig, axes = plt.subplots(1, len(blast_results), figsize=(5 * len(blast_results), 5), squeeze=False)
    for ax, (t, values) in zip(axes[0], blast_results.items()):
        ax.hist(values, bins=50, color=plt.cm.plasma(t / 100), edgecolor='white', linewidth=0.5)
        ax.axvline(t, color='black', linestyle='--', linewidth=1.3, label=f'Cluster threshold: {t}%')
        ax.axvline(60, color='orange', linestyle=':', linewidth=1.3, label='60% reference')
        ax.set_title(f'{t}% cluster\n{sum(v >= t for v in values) / len(values):.2%} ? threshold', fontsize=10, fontweight='bold')
        ax.set_xlabel('Maximum BLAST identity to training (%)', fontsize=9)
        ax.set_ylabel('Test sequences', fontsize=9)
        ax.legend(fontsize=8)
        spine_clean(ax)
    fig.suptitle('Independent BLAST check: test ? train sequence identity', fontsize=12, fontweight='bold')
    out = PLOTS_DIR / '09_blast_identity.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 10: Comparison bar — DeepGreenGO vs PDB splits ───────────────────

def plot_leakage_verification(available: list[int]) -> None:
    report_path = DATA_DIR / 'blast_leakage.json'
    verification_path = DATA_DIR / 'split_verification.json'
    if not report_path.exists():
        print('  [SKIP] blast_leakage.json not found.')
        return
    report = json.loads(report_path.read_text())
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    thresholds = [t for t in available if str(t) in report]
    vals = [report[str(t)]['fraction_at_or_above_threshold_and_80pct_coverage'] for t in thresholds]
    overlaps = [verification.get(str(t), {}).get('components_crossing_splits', None) for t in thresholds]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([f'{t}%' for t in thresholds], vals, color=SPLIT_COLORS['test'], alpha=0.85)
    for bar, val, overlap in zip(bars, vals, overlaps):
        label = f'{val:.2%}' if overlap is None else f'{val:.2%}\ncomponents crossing={overlap}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, label,
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Fraction with identity ? threshold and ?80% aligned coverage')
    ax.set_xlabel('PDB entity-cluster threshold')
    ax.set_title('Leakage verification: cluster assignment plus independent BLAST check', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(1.1, max(vals, default=0) * 1.35 + 0.05))
    spine_clean(ax)
    out = PLOTS_DIR / '10_leakage_verification.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 11: Dataset deep-dive ─────────────────────────────────────────────

def plot_dataset_deepdive() -> None:
    """Deep-dive on the entire Viridiplantae dataset (independent of split threshold).

    Layout (3 rows × 3 cols):
      Row 0: sequence length dist | MF label-count hist | BP label-count hist
      Row 1: CC label-count hist  | MF term rank plot   | BP term rank plot
      Row 2: CC term rank plot    | (hidden)            | (hidden)
    """
    fasta_path = DATA_DIR / 'all_sequences.fasta'
    if not fasta_path.exists():
        print('  [SKIP] all_sequences.fasta not found.')
        return

    seqs = load_fasta(fasta_path)
    go_table = load_go_table()
    onts = list(ONT_LABELS.keys())   # MF, BP, CC

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # ── Row 0, Col 0: Sequence length distribution ───────────────────────────
    ax_len = axes[0][0]
    lengths = [len(v) for v in seqs.values()]
    ax_len.hist(lengths, bins=80, color='#4C72B0', edgecolor='white', linewidth=0.4)
    ax_len.axvline(np.median(lengths), color='red', linestyle='--', linewidth=1.5,
                   label=f'Median: {int(np.median(lengths)):,} aa')
    ax_len.set_xlabel('Sequence length (aa)', fontsize=10)
    ax_len.set_ylabel('Count', fontsize=10)
    ax_len.set_title('Sequence Length Distribution', fontsize=11, fontweight='bold')
    ax_len.legend(fontsize=9)
    ax_len.set_yscale('log')
    spine_clean(ax_len)

    # ── Label-count histograms: MF→(0,1), BP→(0,2), CC→(1,0) ───────────────
    label_hist_axes = [axes[0][1], axes[0][2], axes[1][0]]
    for ax, ont in zip(label_hist_axes, onts):
        label_counts = [len(go_table.get(pid, {}).get(ont, [])) for pid in seqs]
        labelled   = [c for c in label_counts if c > 0]
        unlabelled = sum(1 for c in label_counts if c == 0)
        ax.hist(labelled, bins=50, color='#55A868', edgecolor='white', linewidth=0.4)
        ax.set_xlabel('GO terms per protein', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(
            f'{ONT_LABELS[ont]}: Label Count Distribution\n'
            f'({len(labelled):,} labelled, {unlabelled:,} unlabelled)',
            fontsize=10, fontweight='bold')
        ax.set_yscale('log')
        spine_clean(ax)

    # ── Term frequency rank plots: MF→(1,1), BP→(1,2), CC→(2,0) ────────────
    rank_axes = [axes[1][1], axes[1][2], axes[2][0]]
    for ax, ont in zip(rank_axes, onts):
        term_counts: dict[str, int] = defaultdict(int)
        for pid in seqs:
            for go_term in go_table.get(pid, {}).get(ont, []):
                term_counts[go_term] += 1
        counts_sorted = sorted(term_counts.values(), reverse=True)
        ax.plot(range(1, len(counts_sorted) + 1), counts_sorted,
                color='#C44E52', linewidth=1.5)
        ax.fill_between(range(1, len(counts_sorted) + 1), counts_sorted,
                        alpha=0.15, color='#C44E52')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.axhline(5, color='grey', linestyle=':', linewidth=1.2, label='5-protein cutoff')
        rare = sum(1 for v in counts_sorted if v <= 5)
        ax.set_xlabel('GO term rank', fontsize=10)
        ax.set_ylabel('Proteins per term', fontsize=10)
        ax.set_title(
            f'{ONT_LABELS[ont]}: Term Frequency Rank Plot\n'
            f'({len(counts_sorted):,} terms, {rare:,} rare ≤5)',
            fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
        spine_clean(ax)

    # Hide unused subplots in bottom row
    for col in [1, 2]:
        axes[2][col].set_visible(False)

    fig.suptitle('Viridiplantae Dataset Deep-Dive (full dataset, pre-split)',
                 fontsize=14, fontweight='bold')
    out = PLOTS_DIR / '11_dataset_deepdive.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── Figure 12: GO term overlap between train and test per threshold ──────────

def plot_term_overlap(available: list[int]) -> None:
    onts = list(ONT_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, ont in zip(axes, onts):
        fracs = []
        for t in available:
            cov = load_go_coverage(t)
            if cov is None:
                fracs.append(np.nan)
                continue
            train_terms = set(cov['train'][ont].get('term_counts', {}).keys())
            test_terms  = set(cov['test'][ont].get('term_counts', {}).keys())
            frac = len(train_terms & test_terms) / len(test_terms) if test_terms else 0
            fracs.append(frac)

        ax.bar([f'{t}%' for t in available], fracs, color='#2C3E50', alpha=0.8)
        ax.set_xlabel('Threshold', fontsize=10)
        ax.set_ylabel('Fraction of test GO terms\npresent in train', fontsize=9)
        ax.set_title(f'{ONT_LABELS[ont]}', fontsize=11, fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.05)
        spine_clean(ax)

    fig.suptitle('GO Term Overlap: Fraction of Test Terms Present in Train',
                 fontsize=12, fontweight='bold')
    out = PLOTS_DIR / '12_go_term_overlap.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved: {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Analyse and plot PDB-cluster splits.')
    parser.add_argument('--skip_blast', action='store_true',
                        help='Skip BLAST runs, parse existing TSV files if present.')
    args = parser.parse_args()

    available = collect_available_thresholds()
    if not available:
        sys.exit('[ERROR] No split logs found. Run split_by_pdb_clusters.py --all first.')

    print(f'Available thresholds: {available}')
    print(f'Saving plots to: {PLOTS_DIR}\n')

    print('── Plot 1: Summary table ──────────────────────────────────────────')
    plot_summary_table(available)

    print('── Plot 2: Protein counts ─────────────────────────────────────────')
    plot_protein_counts(available)

    print('── Plot 3: Achieved fractions ─────────────────────────────────────')
    plot_achieved_fracs(available)

    print('── Plot 4: GO coverage heatmap ────────────────────────────────────')
    plot_go_coverage_heatmap(available)

    print('── Plot 5: GO protein coverage ────────────────────────────────────')
    plot_go_protein_coverage(available)

    print('── Plot 6: GO term frequency CDF ──────────────────────────────────')
    plot_go_term_freq_cdf(available)

    print('── Plot 7: Rare term risk ─────────────────────────────────────────')
    plot_rare_term_overlap(available)

    print('── Plot 8: Super-cluster counts ───────────────────────────────────')
    plot_supercluster_sizes(available)

    print('── Plot 9: BLAST identity distributions (?80% coverage) ───────────────────────────')
    run_blast_analysis(available, skip_blast=args.skip_blast)

    print('── Plot 10: Leakage verification ────────────────────────')
    plot_leakage_verification(available)

    print('── Plot 11: Dataset deep-dive ─────────────────────────────────────')
    plot_dataset_deepdive()

    print('── Plot 12: GO term overlap ───────────────────────────────────────')
    plot_term_overlap(available)

    print(f'\nAll plots saved to {PLOTS_DIR}')


if __name__ == '__main__':
    main()
