"""
plot_results.py
Generate publication-quality figures for DeepGreenGO manuscript.
"""

import os
import glob
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    'BCE':   '#4C72B0',
    'Focal': '#DD8452',
}
MODEL_ORDER = ['MLP', 'GCN', 'GAT', 'Hybrid']

matplotlib.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi':     300,
    'savefig.dpi':    600,
    'savefig.bbox':   'tight',
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Ablation bar-plot with significance annotations (Placeholder)
# ─────────────────────────────────────────────────────────────────────────────
def plot_ablation_summary(agg_csv: str, out_dir: str):
    if not os.path.exists(agg_csv):
        print(f"[skip] {agg_csv} not found")
        return

    df = pd.read_csv(agg_csv)
    for ont in df['Ontology'].unique():
        sub = df[df['Ontology'] == ont].copy()
        sub['Model'] = pd.Categorical(sub['Model'], categories=MODEL_ORDER, ordered=True)
        sub = sub.sort_values('Model')

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
        fig.suptitle(f'Ablation Study — {ont.replace("_", " ").title()}', fontsize=14)

        for ax, (metric, label) in zip(axes, [('Micro_Fmax', 'Micro Fmax ↑'), ('Smin', 'Smin ↓')]):
            mean_col = f"{metric}_mean"
            std_col  = f"{metric}_std"
            if mean_col not in sub.columns: continue

            x = np.arange(len(MODEL_ORDER))
            width = 0.35
            losses = sub['Loss'].unique()

            for k, loss in enumerate(sorted(losses)):
                row = sub[sub['Loss'] == loss].set_index('Model')
                means = [row.loc[m, mean_col] if m in row.index else np.nan for m in MODEL_ORDER]
                stds  = [row.loc[m, std_col]  if m in row.index else 0 for m in MODEL_ORDER]
                bars = ax.bar(x + (k - 0.5) * width, means, width, label=loss, color=PALETTE.get(loss, None), alpha=0.88, edgecolor='white', linewidth=0.6)
                ax.errorbar(x + (k - 0.5) * width, means, yerr=stds, fmt='none', ecolor='#333', capsize=4, linewidth=1.2)

            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_ORDER)
            ax.set_xlabel("Architecture")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend(title="Loss", framealpha=0.7)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        out = os.path.join(out_dir, f'ablation_barplot_{ont}.png')
        plt.savefig(out)
        out_pdf = os.path.join(out_dir, f'ablation_barplot_{ont}.pdf')
        plt.savefig(out_pdf, format='pdf')
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Learning curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curves(runs_dir: str, out_dir: str):
    log_files = glob.glob(os.path.join(runs_dir, "*", "training_log.csv"))
    if not log_files: return

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.get_cmap('tab20', len(log_files))
    for i, log in enumerate(sorted(log_files)):
        run_name = os.path.basename(os.path.dirname(log))
        try: df = pd.read_csv(log)
        except Exception: continue
        if 'Valid_Micro_Fmax' not in df.columns: continue
        ax.plot(df['Epoch'], df['Valid_Micro_Fmax'], label=run_name, color=cmap(i), alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Micro Fmax")
    ax.set_title("Learning Curves — Validation Performance")
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Put legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.6)

    out = os.path.join(out_dir, 'learning_curves.png')
    plt.savefig(out)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-cluster violin-plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_performance(cluster_csv: str, out_dir: str):
    if not os.path.exists(cluster_csv): return

    df = pd.read_csv(cluster_csv)
    CAT_ORDER = ["Singleton (1)", "Small (2–4)", "Medium (5–19)", "Large (≥20)"]

    def categorize(size):
        if size == 1: return "Singleton (1)"
        if size < 5:  return "Small (2–4)"
        if size < 20: return "Medium (5–19)"
        return "Large (≥20)"

    df['Cluster_Size'] = df['Size'].apply(categorize)
    df['Cluster_Size'] = pd.Categorical(df['Cluster_Size'], categories=CAT_ORDER, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Model Generalisation by Homology Cluster Size', fontsize=14)

    for ax, (col, label) in zip(axes, [('Micro_Fmax', 'Micro Fmax ↑'), ('Smin', 'Smin ↓')]):
        if col not in df.columns: continue
        sns.violinplot(data=df, x='Cluster_Size', y=col, order=CAT_ORDER, palette='Blues', ax=ax, inner="quartile")
        ax.set_xlabel("Cluster Size Category")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(out_dir, 'cluster_violinplot.png')
    plt.savefig(out)
    out_pdf = os.path.join(out_dir, 'cluster_violinplot.pdf')
    plt.savefig(out_pdf, format='pdf')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metric heat-map
# ─────────────────────────────────────────────────────────────────────────────
def plot_metric_heatmap(agg_csv: str, out_dir: str):
    if not os.path.exists(agg_csv): return

    df = pd.read_csv(agg_csv)
    MEAN_METRICS = ['Micro_Fmax_mean', 'Macro_Fmax_mean', 'Micro_AUROC_mean', 'Macro_AUROC_mean', 'Micro_AUPRC_mean', 'Macro_AUPRC_mean', 'Smin_mean']

    for ont in df['Ontology'].unique():
        sub = df[df['Ontology'] == ont].copy()
        sub['Config'] = sub['Model'] + '\n' + sub['Loss']

        available = [c for c in MEAN_METRICS if c in sub.columns]
        if not available: continue

        pivot = sub.set_index('Config')[available]
        pivot.columns = [c.replace('_mean', '') for c in pivot.columns]

        fig, ax = plt.subplots(figsize=(len(pivot.columns) * 1.4 + 1, len(pivot) * 0.6 + 1.5))
        sns.heatmap(pivot.astype(float), annot=True, fmt='.3f', cmap='YlGnBu', linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.7})
        ax.set_title(f'All Metrics — {ont.replace("_", " ").title()}')
        ax.set_xlabel('')
        ax.set_ylabel('Config (Model / Loss)')

        plt.tight_layout()
        out = os.path.join(out_dir, f'metric_heatmap_{ont}.png')
        plt.savefig(out)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5.  PR and ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_pr_roc_curves(runs_dir: str, out_dir: str):
    runs = glob.glob(os.path.join(runs_dir, "*"))
    
    # We will plot one curve per ontology for the best performing model (or all models)
    # For simplicity, we just look at any run that has test_y_true.npy and test_y_pred.npy
    for run in runs:
        if not os.path.isdir(run): continue
        run_name = os.path.basename(run)
        y_true_path = os.path.join(run, 'test_y_true.npy')
        y_pred_path = os.path.join(run, 'test_y_pred.npy')
        
        if not (os.path.exists(y_true_path) and os.path.exists(y_pred_path)):
            continue
            
        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
        
        # Flatten for micro-average
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        
        # Plot PR Curve
        precision, recall, pr_thresh = precision_recall_curve(y_true_flat, y_pred_flat)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, color='#2ca02c', lw=2, label=f'Micro-average (AUC = {pr_auc:.3f})')
        
        # Calculate F1 scores to find max F1 (Fmax) and mark it
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        ax.plot(recall[best_idx], precision[best_idx], marker='o', markersize=8, color='red', label=f'Max F1: {f1_scores[best_idx]:.3f}')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve ({run_name})')
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'pr_curve_{run_name}.png'))
        plt.savefig(os.path.join(out_dir, f'pr_curve_{run_name}.pdf'))
        plt.close()
        
        # Plot ROC Curve
        fpr, tpr, roc_thresh = roc_curve(y_true_flat, y_pred_flat)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'Micro-average (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve ({run_name})')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'roc_curve_{run_name}.png'))
        plt.savefig(os.path.join(out_dir, f'roc_curve_{run_name}.pdf'))
        plt.close()
        print(f"  Saved PR/ROC for {run_name}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Radar Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar_summary(agg_csv: str, out_dir: str):
    if not os.path.exists(agg_csv): return
    df = pd.read_csv(agg_csv)
    
    metrics = ['Micro_Fmax_mean', 'Macro_Fmax_mean', 'Micro_AUROC_mean', 'Micro_AUPRC_mean']
    
    for ont in df['Ontology'].unique():
        sub = df[df['Ontology'] == ont].copy()
        sub['Config'] = sub['Model'] + ' (' + sub['Loss'] + ')'
        
        # Normalize metrics to 0-1 range for radar plot if needed, but these are mostly 0-1 anyway
        available = [c for c in metrics if c in sub.columns]
        if len(available) < 3: continue
        
        angles = np.linspace(0, 2 * np.pi, len(available), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        for idx, row in sub.iterrows():
            values = row[available].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=row['Config'])
            ax.fill(angles, values, alpha=0.1)
            
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), [m.replace('_mean', '').replace('_', ' ') for m in available])
        ax.set_title(f'Performance Radar — {ont.replace("_", " ").title()}', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'radar_plot_{ont}.png'))
        plt.savefig(os.path.join(out_dir, f'radar_plot_{ont}.pdf'))
        plt.close()

if __name__ == "__main__":
    out_dir = "plots/"
    os.makedirs(out_dir, exist_ok=True)

    print("Generating publication-quality figures...")
    plot_ablation_summary("runs/aggregated_results.csv", out_dir)
    plot_learning_curves("runs/", out_dir)
    plot_cluster_performance("runs/cluster_performance.csv", out_dir)
    plot_metric_heatmap("runs/aggregated_results.csv", out_dir)
    plot_pr_roc_curves("runs/", out_dir)
    plot_radar_summary("runs/aggregated_results.csv", out_dir)
    print(f"\nAll figures saved to {out_dir}")
