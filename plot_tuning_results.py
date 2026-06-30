import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_tuning_summary(csv_path="tuning_runs/tuning_results_summary.csv", out_path="tuning_runs/tuning_visualization.pdf"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Filter only completed runs
    df = df[df['Status'] == 'Completed']
    if df.empty:
        print("No completed runs found to plot.")
        return

    ontologies = ['biological_process', 'molecular_function', 'cellular_component']
    titles = ['Biological Process', 'Molecular Function', 'Cellular Component']
    
    # Set up publication-quality plot settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 300,
        'font.family': 'sans-serif'
    })
    
    sns.set_style("whitegrid")
    
    # Create 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    
    for i, (ont, title) in enumerate(zip(ontologies, titles)):
        ax = axes[i]
        ont_df = df[df['Ontology'] == ont].copy()
        
        if ont_df.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=14)
            ax.set_title(title)
            continue
            
        # Create a readable label for each combination
        ont_df['Params'] = ont_df.apply(lambda row: f"LR:{row['LR']} | DP:{row['Dropout']} | BS:{int(row['BatchSize'])}", axis=1)
        
        # Sort descending and take top 10 for readability
        ont_df = ont_df.sort_values('Val_Macro_Fmax', ascending=False).head(10)
        
        sns.barplot(
            data=ont_df,
            x='Val_Macro_Fmax',
            y='Params',
            ax=ax,
            palette='viridis',
            edgecolor='k'
        )
        
        ax.set_title(title, pad=15, fontweight='bold')
        ax.set_xlabel('Validation Macro-Fmax')
        ax.set_ylabel('')
        
        # Dynamically set X-axis limits to zoom in on the differences
        min_val = ont_df['Val_Macro_Fmax'].min()
        max_val = ont_df['Val_Macro_Fmax'].max()
        padding = (max_val - min_val) * 0.2
        if padding == 0: padding = 0.05
        ax.set_xlim(max(0, min_val - padding), max_val + padding)
        
        # Add value labels on the bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=5, fontsize=10)
            
    plt.tight_layout()
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
    print(f"Saved highly polished publication plot to {out_path} (and .png)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot tuning results from CSV")
    parser.add_argument("--csv_path", type=str, default="tuning_runs/tuning_results_summary.csv", help="Path to tuning summary CSV")
    parser.add_argument("--out_path", type=str, default="tuning_runs/tuning_visualization.pdf", help="Path to output plot PDF")
    args = parser.parse_args()
    
    plot_tuning_summary(csv_path=args.csv_path, out_path=args.out_path)
