import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
"""
Publication-quality plotting for DeepGreenGO extended ablations (Input & Loss).
Generates multi-metric figures with clear value annotations and professional styling.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INPUT_DIR = 'runs_ablation_input'
LOSS_DIR = 'runs_ablation_loss'
OUT_DIR = 'plots/ablations'
os.makedirs(OUT_DIR, exist_ok=True)

ONTOLOGIES = ['mf', 'bp', 'cc']
ONT_LABELS = {'mf': 'Molecular Function', 'bp': 'Biological Process', 'cc': 'Cellular Component'}

# High-contrast palettes (Paul Tol vibrant-inspired)
COLORS_INPUT = {
    'Full (Seq + Struct)': '#EE3377',    # Magenta
    'Sequence Only': '#0077BB',          # Blue
    'Structure Only': '#EE7733',         # Orange
}
ORDER_INPUT = ['Full (Seq + Struct)', 'Sequence Only', 'Structure Only']

COLORS_LOSS = {
    'BCE': '#33BBEE',                 # Cyan
    'Focal (γ=1.0)': '#EE7733',       # Orange
    'Focal (γ=2.0)': '#CC3311',       # Red
    'Focal (γ=3.0)': '#009988',       # Teal
}
ORDER_LOSS = ['BCE', 'Focal (γ=1.0)', 'Focal (γ=2.0)', 'Focal (γ=3.0)']

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

def annotate_bars(ax, metric_name, decimals=3):
    """Add value labels above bars."""
    is_smin = "Smin" in metric_name or "smin" in metric_name.lower()
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height) or height == 0:
            continue
        
        x_pos = patch.get_x() + patch.get_width() / 2
        
        # Smin uses fewer decimals.
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
# PLOTTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_micro_metrics(df, hue_col, order, colors, prefix='input'):
    """Plot protein-centric (micro-averaged) metrics across ontologies."""
    
    metrics = [
        ('Micro_Fmax', 'Micro Fmax', 3),
        ('Micro_AUROC', 'Micro AUROC', 3),
        ('Micro_AUPRC', 'Micro AUPRC', 3),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f'Protein-Centric Performance ({prefix.title()} Ablation)', 
                 fontsize=14, fontweight='normal', y=1.00)
    
    for idx, (metric_col, metric_label, decimals) in enumerate(metrics):
        ax = axes[idx]
        
        plot_df = df.copy()
        plot_df['Ontology_Label'] = plot_df['Ontology'].str.lower().map(ONT_LABELS)
        
        if plot_df.empty or metric_col not in plot_df.columns:
            ax.text(0.5, 0.5, f"No data for {metric_label}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        sns.barplot(
            data=plot_df,
            x='Ontology_Label',
            y=metric_col,
            hue=hue_col,
            hue_order=order,
            order=[ONT_LABELS[o] for o in ONTOLOGIES],
            palette=colors,
            errorbar='sd',
            capsize=0.08,
            err_kws={'linewidth': 1.0, 'color': 'black'},
            edgecolor='white',
            linewidth=1.0,
            ax=ax
        )
        
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
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
        annotate_bars(ax, metric_label, decimals)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(f'{prefix}_micro_metrics', OUT_DIR)


def plot_macro_metrics(df, hue_col, order, colors, prefix='input'):
    """Plot label-centric (macro-averaged) metrics across ontologies."""
    
    metrics = [
        ('Macro_Fmax', 'Macro Fmax', 3),
        ('Macro_AUROC', 'Macro AUROC', 3),
        ('Macro_AUPRC', 'Macro AUPRC', 3),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f'Label-Centric Performance ({prefix.title()} Ablation)',
                 fontsize=14, fontweight='normal', y=1.00)
    
    for idx, (metric_col, metric_label, decimals) in enumerate(metrics):
        ax = axes[idx]
        
        plot_df = df.copy()
        plot_df['Ontology_Label'] = plot_df['Ontology'].str.lower().map(ONT_LABELS)
        
        if plot_df.empty or metric_col not in plot_df.columns:
            ax.text(0.5, 0.5, f"No data for {metric_label}",
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        sns.barplot(
            data=plot_df,
            x='Ontology_Label',
            y=metric_col,
            hue=hue_col,
            hue_order=order,
            order=[ONT_LABELS[o] for o in ONTOLOGIES],
            palette=colors,
            errorbar='sd',
            capsize=0.08,
            err_kws={'linewidth': 1.0, 'color': 'black'},
            edgecolor='white',
            linewidth=1.0,
            ax=ax
        )
        
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
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
        annotate_bars(ax, metric_label, decimals)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(f'{prefix}_macro_metrics', OUT_DIR)


def plot_smin(df, hue_col, order, colors, prefix='input'):
    """Plot Smin (semantic distance)."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f'Semantic Distance (Smin) ({prefix.title()} Ablation)',
                 fontsize=14, fontweight='normal', y=1.00)
    
    for idx, ont in enumerate(ONTOLOGIES):
        ax = axes[idx]
        
        ont_data = df[df['Ontology'] == ont.upper()]
        
        if ont_data.empty or 'Smin' not in ont_data.columns:
            ax.text(0.5, 0.5, f"No Smin data", ha='center', va='center',
                   transform=ax.transAxes)
            continue
        
        sns.barplot(
            data=ont_data,
            x=hue_col,
            y='Smin',
            order=order,
            palette=colors,
            errorbar='sd',
            capsize=0.08,
            err_kws={'linewidth': 1.0, 'color': 'black'},
            edgecolor='white',
            linewidth=1.0,
            ax=ax
        )
        
        ax.set_title(ONT_LABELS[ont], fontweight='normal', fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('Smin' if idx == 0 else '')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
        
        style_ax(ax)
        annotate_bars(ax, "Smin", 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(f'{prefix}_smin', OUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def load_input_ablations():
    records = []
    if not os.path.exists(INPUT_DIR):
        return pd.DataFrame()
        
    for folder in os.listdir(INPUT_DIR):
        metrics_path = os.path.join(INPUT_DIR, folder, 'test_metrics.json')
        if not os.path.exists(metrics_path):
            continue
        
        parts = folder.split('_')
        ont = parts[0].upper()
        
        if 'seq' in parts and 'only' in parts:
            modality = 'Sequence Only'
        elif 'struct' in parts and 'only' in parts:
            modality = 'Structure Only'
        else:
            modality = 'Full (Seq + Struct)'
            
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except:
            continue
        
        record = {
            'Ontology': ont,
            'Modality': modality,
            'Folder': folder
        }
        record.update(metrics)
        records.append(record)
    
    return pd.DataFrame(records)


def load_loss_ablations():
    records = []
    if not os.path.exists(LOSS_DIR):
        return pd.DataFrame()
        
    for folder in os.listdir(LOSS_DIR):
        metrics_path = os.path.join(LOSS_DIR, folder, 'test_metrics.json')
        if not os.path.exists(metrics_path):
            continue
        
        parts = folder.split('_')
        ont = parts[0].upper()
        
        if 'BCE' in parts:
            loss = 'BCE'
        elif 'Focal' in parts:
            gamma = parts[-1].replace('g', '')
            loss = f'Focal (γ={gamma})'
        else:
            continue
            
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        except:
            continue
        
        record = {
            'Ontology': ont,
            'Loss': loss,
            'Folder': folder
        }
        record.update(metrics)
        records.append(record)
    
    return pd.DataFrame(records)


def main():
    print("Loading Input Ablations...")
    df_input = load_input_ablations()
    if not df_input.empty:
        print(f"✓ Loaded {len(df_input)} input ablation records")
        plot_micro_metrics(df_input, hue_col='Modality', order=ORDER_INPUT, colors=COLORS_INPUT, prefix='input')
        plot_macro_metrics(df_input, hue_col='Modality', order=ORDER_INPUT, colors=COLORS_INPUT, prefix='input')
        plot_smin(df_input, hue_col='Modality', order=ORDER_INPUT, colors=COLORS_INPUT, prefix='input')
    else:
        print("⚠ No input ablation results found.")
        
    print("\nLoading Loss Ablations...")
    df_loss = load_loss_ablations()
    if not df_loss.empty:
        print(f"✓ Loaded {len(df_loss)} loss ablation records")
        plot_micro_metrics(df_loss, hue_col='Loss', order=ORDER_LOSS, colors=COLORS_LOSS, prefix='loss')
        plot_macro_metrics(df_loss, hue_col='Loss', order=ORDER_LOSS, colors=COLORS_LOSS, prefix='loss')
        plot_smin(df_loss, hue_col='Loss', order=ORDER_LOSS, colors=COLORS_LOSS, prefix='loss')
    else:
        print("⚠ No loss ablation results found.")

if __name__ == "__main__":
    main()
