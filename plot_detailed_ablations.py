import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics for Nature-quality plots
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

OUTPUT_DIR = 'ablation_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_runs(base_dir):
    data = []
    if not os.path.exists(base_dir):
        return data
    for folder in os.listdir(base_dir):
        metrics_path = os.path.join(base_dir, folder, 'test_metrics.json')
        if not os.path.exists(metrics_path):
            continue
            
        # Parse based on folder naming conventions
        # Loss Ablation format: {ont}_{model}_{loss}_lr..._g{gamma}
        # Input Ablation format: {ont}_{model}_{loss}_lr..._s{seed}_{modality}
        
        parts = folder.split('_')
        ont = parts[0].upper()
        
        # Loss and Gamma
        loss = "BCE" if "_BCE_" in folder else "Focal"
        gamma = "N/A"
        if loss == "Focal":
            for p in parts:
                if p.startswith('g') and '.' in p:
                    gamma = p[1:]
        
        # Modality
        modality = "full"
        if "seq_only" in folder: modality = "Sequence Only"
        elif "struct_only" in folder: modality = "Structure Only"
        elif "full" in folder: modality = "Seq + Struct (Full)"
        
        # Seed
        seed = None
        for p in parts:
            if p.startswith('s') and p[1:].isdigit():
                seed = int(p[1:])
                
        with open(metrics_path, 'r') as f:
            try: m = json.load(f)
            except: continue
            
        data.append({
            'Ontology': ont,
            'Loss': loss,
            'Gamma': gamma,
            'Modality': modality,
            'Seed': seed,
            'Macro_Fmax': m.get('Macro_Fmax'),
            'Macro_AUROC': m.get('Macro_AUROC'),
            'Macro_AUPRC': m.get('Macro_AUPRC'),
            'Smin': m.get('Smin')
        })
    return pd.DataFrame(data)

def save_fig(name):
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"), bbox_inches='tight', format='pdf', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), bbox_inches='tight', format='png', dpi=300)
    plt.close()

# -------------------------------------------------------------------------
# 1. Input Ablation Plots
# -------------------------------------------------------------------------
df_input = parse_runs('runs_ablation_input')
if not df_input.empty and len(df_input['Modality'].unique()) > 1:
    print("Generating Input Ablation Plots...")
    g = sns.catplot(
        data=df_input, x='Modality', y='Macro_Fmax', col='Ontology',
        kind='bar', palette='viridis', errorbar='sd',
        height=5, aspect=1.0, capsize=.1, err_kws={'linewidth': 1.5}
    )
    g.set(ylim=(0.5, 1.0))
    g.set_axis_labels("", "Macro Fmax")
    g.set_titles("{col_name} Ontology")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Input Modality Ablation (Hybrid_JK)")
    save_fig("ablation_input_fmax")
else:
    print("Input ablation data incomplete or missing. Skipping input plots.")

# -------------------------------------------------------------------------
# 2. Loss Ablation Plots (Gamma values)
# -------------------------------------------------------------------------
df_loss = parse_runs('runs_ablation_loss')
if not df_loss.empty:
    print("Generating Loss Ablation Plots...")
    # Create a unified 'Configuration' column for plotting
    df_loss['Config'] = df_loss.apply(lambda row: 'BCE' if row['Loss'] == 'BCE' else f"Focal (γ={row['Gamma']})", axis=1)
    
    # Sort order: BCE, Focal(1.0), Focal(2.0), Focal(3.0)...
    order = sorted(df_loss['Config'].unique(), key=lambda x: (0, '') if x == 'BCE' else (1, x))
    
    g = sns.catplot(
        data=df_loss, x='Config', y='Macro_Fmax', col='Ontology',
        kind='bar', order=order, palette='Set2', errorbar='sd',
        height=5, aspect=1.0, capsize=.1, err_kws={'linewidth': 1.5}
    )
    g.set(ylim=(0.5, 1.0))
    g.set_axis_labels("", "Macro Fmax")
    g.set_titles("{col_name} Ontology")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Loss Function Ablation: BCE vs Focal Gammas (Hybrid_JK)")
    save_fig("ablation_loss_fmax")
    
    # Also plot AUPRC which is highly sensitive to Focal
    g_prc = sns.catplot(
        data=df_loss, x='Config', y='Macro_AUPRC', col='Ontology',
        kind='bar', order=order, palette='Set2', errorbar='sd',
        height=5, aspect=1.0, capsize=.1, err_kws={'linewidth': 1.5}
    )
    g_prc.set_axis_labels("", "Macro AUPRC")
    g_prc.set_titles("{col_name} Ontology")
    for ax in g_prc.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
    plt.subplots_adjust(top=0.85)
    g_prc.fig.suptitle("Loss Function Ablation: Impact on AUPRC")
    save_fig("ablation_loss_auprc")
else:
    print("Loss ablation data incomplete or missing. Skipping loss plots.")

print("Done plotting ablations!")
