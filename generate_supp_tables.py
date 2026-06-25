"""
generate_supp_tables.py
Generates comprehensive supplementary tables (CSV and LaTeX format)
for hyperparameter configs, model details, and dataset statistics.
"""

import os
import glob
import json
import pandas as pd

def gather_configs(runs_dir: str):
    configs = []
    for conf_file in glob.glob(os.path.join(runs_dir, "*", "config.json")):
        with open(conf_file, 'r') as f:
            c = json.load(f)
            # Add run name
            c['Run'] = os.path.basename(os.path.dirname(conf_file))
            configs.append(c)
    return configs

def generate_hyperparam_table(configs, out_dir):
    if not configs:
        print("[skip] No config.json files found.")
        return
        
    df = pd.DataFrame(configs)
    
    # Select columns of interest for the supplementary table
    cols = ['Run', 'model', 'loss', 'ontology', 'lr', 'batch_size', 'epochs', 
            'dropout', 'focal_gamma', 'hidden_sizes', 'num_heads', 'seed', 'accumulation_steps', 'patience']
            
    # Only keep columns that exist in the parsed configs
    existing_cols = [c for c in cols if c in df.columns]
    supp_df = df[existing_cols]
    
    # Save CSV
    csv_path = os.path.join(out_dir, "supp_table_hyperparameters.csv")
    supp_df.to_csv(csv_path, index=False)
    
    # Save LaTeX fragment
    tex_path = os.path.join(out_dir, "supp_table_hyperparameters.tex")
    with open(tex_path, 'w') as f:
        f.write(supp_df.to_latex(index=False, caption="Hyperparameter configurations for all runs.", label="tab:hyperparams"))
        
    print(f"Generated hyperparameter supplementary tables in {out_dir}")

def generate_hardware_table(configs, out_dir):
    if not configs: return
    
    hw_info = []
    for c in configs:
        if 'hardware' in c:
            hw_info.append({'Run': c['Run'], **c['hardware']})
            
    if not hw_info: return
    
    df = pd.DataFrame(hw_info)
    df.to_csv(os.path.join(out_dir, "supp_table_hardware.csv"), index=False)
    
    with open(os.path.join(out_dir, "supp_table_hardware.tex"), 'w') as f:
        f.write(df.to_latex(index=False, caption="Hardware specifications per run.", label="tab:hardware"))
        
    print(f"Generated hardware supplementary tables in {out_dir}")

if __name__ == "__main__":
    runs_dir = "runs/"
    out_dir = "supp_material/"
    os.makedirs(out_dir, exist_ok=True)
    
    configs = gather_configs(runs_dir)
    generate_hyperparam_table(configs, out_dir)
    generate_hardware_table(configs, out_dir)
    
    # You can extend this to load datasets.pkl and print sequence length statistics if needed.
    print("Done generating supplementary tables.")
