import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
"""
Publication-quality plotting for DeepGreenGO results.
Generates multi-metric figures with clear value annotations and professional styling.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RUNS_DIR = 'runs'
OUT_DIR = 'plots/ablations'
os.makedirs(OUT_DIR, exist_ok=True)

ONTOLOGIES = ['mf', 'bp', 'cc']
ONT_LABELS = {'mf': 'Molecular Function', 'bp': 'Biological Process', 'cc': 'Cellular Component'}
MODELS = ['MLP', 'GCN', 'GAT', 'Hybrid', 'Hybrid_JK']

# High-contrast palette (Paul Tol vibrant)
COLORS = {
    'MLP': '#EE7733',       # Orange
    'GCN': '#0077BB',       # Blue
    'GAT': '#33BBEE',       # Cyan
    'Hybrid': '#EE3377',    # Magenta
    'Hybrid_JK': '#CC3311'  # Red
}

# Matplotlib configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.edgecolor': '#CCCCCC',
    'grid.color': "#BDBDBD",
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
})

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.patheffects as pe

def annotate_bars(ax, metric_name, decimals=3):
    """Add value labels above bars."""
    is_smin = "Smin" in metric_name or "smin" in metric_name.lower()
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height) or height == 0:
            continue
        
        x_pos = patch.get_x() + patch.get_width() / 2
        
        # Special handling for Smin (lower is better, fewer decimals)
        if is_smin:
            label_text = f"{height:.1f}"
            rot = 0
        else:
            label_text = f"{height:.{decimals}f}"
            rot = 90
        
        ax.text(
            x_pos, height / 2,
            label_text,
            ha='center', va='center',
            fontsize=8, fontweight='normal',
            color='#111111',
            rotation=rot
        )


def save_fig(name, target_dir, dpi=300):
    """Save figure with consistent settings."""
    plt.savefig(
        os.path.join(target_dir, f"{name}.png"),
        bbox_inches='tight',
        dpi=dpi,
        edgecolor='none'
    )
    plt.close()
    print(f"✓ Saved: {name}.png")


def style_ax(ax, remove_spines=['top', 'right']):
    """Apply consistent styling to axes."""
    for spine in remove_spines:
        ax.spines[spine].set_visible(False)
    
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.7)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# PROTEIN-CENTRIC PLOTS (Micro-averaged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_micro_metrics(df):
    """Plot protein-centric (micro-averaged) metrics across ontologies."""
    
    metrics = [
        ('Micro_Fmax', 'Micro Fmax', 3),
        ('Micro_AUROC', 'Micro AUROC', 3),
        ('Micro_AUPRC', 'Micro AUPRC', 3),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Protein-Centric Performance (Micro-averaged)', 
                 fontsize=14, fontweight='normal', y=1.00)
    
    for idx, (metric_col, metric_label, decimals) in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data
        plot_df = df.copy()
        plot_df['Ontology_Label'] = plot_df['Ontology'].str.lower().map(ONT_LABELS)
        
        if plot_df.empty or metric_col not in plot_df.columns:
            ax.text(0.5, 0.5, f"No data for {metric_label}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create plot
        sns.barplot(
            data=plot_df,
            x='Ontology_Label',
            y=metric_col,
            hue='Model',
            hue_order=MODELS,
            order=[ONT_LABELS[o] for o in ONTOLOGIES],
            palette=COLORS,
            errorbar='sd',
            capsize=0.08,
            err_kws={'linewidth': 1.0, 'color': 'black'},
            edgecolor='white',
            linewidth=1.0,
            ax=ax
        )
        
        # Styling
        ax.set_title(metric_label, fontweight='normal', fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel(metric_label if idx == 0 else '')
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        
        style_ax(ax)
        
        if idx == 0:
            ax.legend(
                title=None,
                fontsize=8,
                frameon=False,
                loc='upper right'
            )
        else:
            ax.get_legend().remove() if ax.get_legend() else None
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
        
        # Annotations
        annotate_bars(ax, metric_label, decimals)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig('micro_metrics_combined', OUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# LABEL-CENTRIC PLOTS (Macro-averaged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_macro_metrics(df):
    """Plot label-centric (macro-averaged) metrics across ontologies."""
    
    metrics = [
        ('Macro_Fmax', 'Macro Fmax', 3),
        ('Macro_AUROC', 'Macro AUROC', 3),
        ('Macro_AUPRC', 'Macro AUPRC', 3),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Label-Centric Performance (Macro-averaged)',
                 fontsize=14, fontweight='normal', y=1.00)
    
    for idx, (metric_col, metric_label, decimals) in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data
        plot_df = df.copy()
        plot_df['Ontology_Label'] = plot_df['Ontology'].str.lower().map(ONT_LABELS)
        
        if plot_df.empty or metric_col not in plot_df.columns:
            ax.text(0.5, 0.5, f"No data for {metric_label}",
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create plot
        sns.barplot(
            data=plot_df,
            x='Ontology_Label',
            y=metric_col,
            hue='Model',
            hue_order=MODELS,
            order=[ONT_LABELS[o] for o in ONTOLOGIES],
            palette=COLORS,
            errorbar='sd',
            capsize=0.08,
            err_kws={'linewidth': 1.0, 'color': 'black'},
            edgecolor='white',
            linewidth=1.0,
            ax=ax
        )
        
        # Styling
        ax.set_title(metric_label, fontweight='normal', fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel(metric_label if idx == 0 else '')
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        
        style_ax(ax)
        
        if idx == 0:
            ax.legend(
                title=None,
                fontsize=8,
                frameon=False,
                loc='upper right'
            )
        else:
            ax.get_legend().remove() if ax.get_legend() else None
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
        
        # Annotations
        annotate_bars(ax, metric_label, decimals)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig('macro_metrics_combined', OUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC DISTANCE (SMIN)
# ─────────────────────────────────────────────────────────────────────────────

def plot_smin(df):
    """Plot Smin (semantic distance) — lower is better."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Semantic Distance (Smin)',
                 fontsize=14, fontweight='normal', y=1.00)
    
    for idx, ont in enumerate(ONTOLOGIES):
        ax = axes[idx]
        
        # Prepare data
        ont_data = df[df['Ontology'] == ont.upper()]
        
        if ont_data.empty or 'Smin' not in ont_data.columns:
            ax.text(0.5, 0.5, f"No Smin data", ha='center', va='center',
                   transform=ax.transAxes)
            continue
        
        # Create plot
        sns.barplot(
            data=ont_data,
            x='Model',
            y='Smin',
            order=MODELS,
            palette=COLORS,
            errorbar='sd',
            capsize=0.08,
            err_kws={'linewidth': 1.0, 'color': 'black'},
            edgecolor='white',
            linewidth=1.0,
            ax=ax
        )
        
        # Styling
        ax.set_title(ONT_LABELS[ont], fontweight='normal', fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('Smin' if idx == 0 else '')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        style_ax(ax)
        
        # Annotations
        annotate_bars(ax, "Smin", 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig('smin_across_ontologies', OUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# PR CURVES (Example for best model)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(runs_dir):
    """Generate PR curves for each ontology."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Precision-Recall Curves (Micro-averaged)',
                 fontsize=14, fontweight='normal', y=0.98)
    
    for idx, ont in enumerate(['mf', 'bp', 'cc']):
        ax = axes[idx]
        
        # This is a template — adapt to your actual test_y_true.npy, test_y_pred.npy files
        # For each model, compute and plot PR curve
        
        ax.set_xlabel('Recall', fontweight='normal')
        ax.set_ylabel('Precision' if idx == 0 else '', fontweight='normal')
        ax.set_title(ONT_LABELS[ont], fontweight='normal', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        
        style_ax(ax)
        ax.legend(fontsize=8, loc='lower left', framealpha=0.95, edgecolor='#CCCCCC')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig('pr_curves_combined', OUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def load_results(runs_dir='runs', best_by_fmax=True):
    """
    Load and aggregate results from run directories.
    
    Parameters:
    -----------
    runs_dir : str
        Directory containing run subdirectories with test_metrics.json
    best_by_fmax : bool
        If True, select best config per model+ontology by Macro Fmax
    
    Returns:
    --------
    pd.DataFrame
        Aggregated metrics
    """
    
    records = []
    
    for folder in os.listdir(runs_dir):
        metrics_path = os.path.join(runs_dir, folder, 'test_metrics.json')
        if not os.path.exists(metrics_path):
            continue
        
        # Parse folder name: {ont}_{model}_{loss}_lr{lr}_dp{dp}_bs{bs}_s{seed}
        parts = folder.split('_')
        if len(parts) < 3:
            continue
        
        ont = parts[0].upper()
        
        # Identify model
        model = None
        if 'Hybrid' in parts and 'JK' in parts:
            model = 'Hybrid_JK'
        elif 'Hybrid' in parts:
            model = 'Hybrid'
        elif 'GAT' in parts:
            model = 'GAT'
        elif 'GCN' in parts:
            model = 'GCN'
        elif 'MLP' in parts:
            model = 'MLP'
        else:
            continue
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except:
            continue
        
        record = {
            'Ontology': ont,
            'Model': model,
            'Folder': folder
        }
        record.update(metrics)
        records.append(record)
    
    df = pd.DataFrame(records)
    return df if not df.empty else pd.DataFrame()


def main():
    print(f"Loading results from: {RUNS_DIR}")
    df = load_results(RUNS_DIR)
    
    if df.empty:
        print("⚠ No results found. Ensure test_metrics.json files exist in run directories.")
        return
    
    print(f"✓ Loaded {len(df)} run records\n")
    
    print("Generating plots...")
    plot_micro_metrics(df)
    plot_macro_metrics(df)
    plot_smin(df)
    
    print(f"\n✓ All plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()