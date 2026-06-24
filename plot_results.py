"""
plot_results.py
Generate publication-quality figures for DeepGreenGO manuscript.

Figures produced:
  1.  ablation_barplot_<ontology>.png  – Model × Loss comparison (mean±std bars)
  2.  learning_curves.png              – Val Micro-Fmax over epochs per run
  3.  cluster_boxplot.png              – Per-cluster Fmax by cluster-size category
  4.  metric_heatmap_<ontology>.png    – Heat-map of all metrics per config
  5.  pr_curves_<ontology>.png         – Aggregated PR-curve placeholder (if raw preds available)
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
    'figure.dpi':     150,
    'savefig.dpi':    300,
    'savefig.bbox':   'tight',
})

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Ablation bar-plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_ablation_summary(agg_csv: str, out_dir: str):
    if not os.path.exists(agg_csv):
        print(f"[skip] {agg_csv} not found — run aggregate_results.py first")
        return

    df = pd.read_csv(agg_csv)

    for ont in df['Ontology'].unique():
        sub = df[df['Ontology'] == ont].copy()

        # Keep models in fixed order
        sub['Model'] = pd.Categorical(sub['Model'], categories=MODEL_ORDER, ordered=True)
        sub = sub.sort_values('Model')

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
        fig.suptitle(f'Ablation Study — {ont.replace("_", " ").title()}', fontsize=14)

        for ax, (metric, label) in zip(axes, [
            ('Micro_Fmax', 'Micro Fmax ↑'),
            ('Smin',       'Smin ↓'),
        ]):
            mean_col = f"{metric}_mean"
            std_col  = f"{metric}_std"

            if mean_col not in sub.columns:
                continue

            x = np.arange(len(MODEL_ORDER))
            width = 0.35
            losses = sub['Loss'].unique()

            for k, loss in enumerate(sorted(losses)):
                row = sub[sub['Loss'] == loss].set_index('Model')
                means = [row.loc[m, mean_col] if m in row.index else np.nan
                         for m in MODEL_ORDER]
                stds  = [row.loc[m, std_col]  if m in row.index else 0
                         for m in MODEL_ORDER]
                bars = ax.bar(x + (k - 0.5) * width, means, width,
                              label=loss, color=PALETTE.get(loss, None),
                              alpha=0.88, edgecolor='white', linewidth=0.6)
                ax.errorbar(x + (k - 0.5) * width, means, yerr=stds,
                            fmt='none', ecolor='#333', capsize=4, linewidth=1.2)

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
        plt.close()
        print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Learning curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curves(runs_dir: str, out_dir: str):
    log_files = glob.glob(os.path.join(runs_dir, "*", "training_log.csv"))
    if not log_files:
        print(f"[skip] No training_log.csv files in {runs_dir}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    cmap = plt.cm.get_cmap('tab20', len(log_files))
    for i, log in enumerate(sorted(log_files)):
        run_name = os.path.basename(os.path.dirname(log))
        try:
            df = pd.read_csv(log)
        except Exception:
            continue

        if 'Valid_Micro_Fmax' not in df.columns:
            continue

        ax.plot(df['Epoch'], df['Valid_Micro_Fmax'],
                label=run_name, color=cmap(i), alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Micro Fmax")
    ax.set_title("Learning Curves — Validation Performance")
    ax.grid(linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7, ncol=3, loc='lower right', framealpha=0.6)

    out = os.path.join(out_dir, 'learning_curves.png')
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-cluster box-plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_cluster_performance(cluster_csv: str, out_dir: str):
    if not os.path.exists(cluster_csv):
        print(f"[skip] {cluster_csv} not found — run per_cluster_eval.py first")
        return

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

    for ax, (col, label) in zip(axes, [
        ('Micro_Fmax',  'Micro Fmax ↑'),
        ('Smin',        'Smin ↓'),
    ]):
        if col not in df.columns:
            continue
        sns.boxplot(data=df, x='Cluster_Size', y=col, order=CAT_ORDER,
                    palette='Blues', ax=ax, linewidth=1.2)
        ax.set_xlabel("Cluster Size Category")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(out_dir, 'cluster_boxplot.png')
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metric heat-map
# ─────────────────────────────────────────────────────────────────────────────
def plot_metric_heatmap(agg_csv: str, out_dir: str):
    if not os.path.exists(agg_csv):
        return

    df = pd.read_csv(agg_csv)
    MEAN_METRICS = ['Micro_Fmax_mean', 'Macro_Fmax_mean',
                    'Micro_AUROC_mean', 'Macro_AUROC_mean',
                    'Micro_AUPRC_mean', 'Macro_AUPRC_mean',
                    'Smin_mean']

    for ont in df['Ontology'].unique():
        sub = df[df['Ontology'] == ont].copy()
        sub['Config'] = sub['Model'] + '\n' + sub['Loss']

        # Only keep columns that exist
        available = [c for c in MEAN_METRICS if c in sub.columns]
        if not available:
            continue

        pivot = sub.set_index('Config')[available]
        pivot.columns = [c.replace('_mean', '') for c in pivot.columns]

        fig, ax = plt.subplots(figsize=(len(pivot.columns) * 1.4 + 1,
                                       len(pivot) * 0.6 + 1.5))
        sns.heatmap(pivot.astype(float), annot=True, fmt='.3f', cmap='YlGnBu',
                    linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.7})
        ax.set_title(f'All Metrics — {ont.replace("_", " ").title()}')
        ax.set_xlabel('')
        ax.set_ylabel('Config (Model / Loss)')

        plt.tight_layout()
        out = os.path.join(out_dir, f'metric_heatmap_{ont}.png')
        plt.savefig(out)
        plt.close()
        print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = "plots/"
    os.makedirs(out_dir, exist_ok=True)

    print("Generating figures...")
    plot_ablation_summary("runs/aggregated_results.csv", out_dir)
    plot_learning_curves("runs/", out_dir)
    plot_cluster_performance("runs/cluster_performance.csv", out_dir)
    plot_metric_heatmap("runs/aggregated_results.csv", out_dir)
    print(f"\nAll figures saved to {out_dir}")
