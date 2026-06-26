"""
aggregate_tuning.py
Scans tuning_runs/ for completed hyperparameter tuning runs and prints/saves
the results sorted by Valid_Macro_Fmax for each ontology.
"""
import os
import json
import pandas as pd
import glob


def aggregate_tuning(runs_dir: str, output_file: str):
    results = []

    for config_file in glob.glob(os.path.join(runs_dir, "*", "config.json")):
        run_dir = os.path.dirname(config_file)
        metric_file = os.path.join(run_dir, "valid_metrics.json")

        # Config always exists if we are in this loop, but metrics might not if it failed
        with open(config_file) as f:
            config = json.load(f)
            
        row = {
            'Ontology': config.get('ontology', 'Unknown'),
            'Model':    config.get('model', 'Unknown'),
            'LR':       config.get('lr', 'Unknown'),
            'Dropout':  config.get('dropout', 'Unknown'),
            'BatchSize': config.get('batch_size', 'Unknown'),
            'Seed':     config.get('seed', 'Unknown'),
        }

        if os.path.exists(metric_file):
            with open(metric_file) as f:
                metrics = json.load(f)
            row.update({
                'Val_Micro_Fmax': metrics.get('Micro_Fmax', float('nan')),
                'Val_Macro_Fmax': metrics.get('Macro_Fmax', float('nan')),
                'Val_Smin':       metrics.get('Smin', float('nan')),
            })
            row['Status'] = 'Completed'
        else:
            row.update({
                'Val_Micro_Fmax': float('nan'),
                'Val_Macro_Fmax': float('nan'),
                'Val_Smin':       float('nan'),
            })
            row['Status'] = 'Failed/Incomplete'

        results.append(row)

    if not results:
        print(f"No tuning runs found in {runs_dir}")
        return

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nAll tuning results saved to {output_file}")

    # Pretty-print summary table
    for ont in df['Ontology'].unique():
        sub = df[(df['Ontology'] == ont) & (df['Status'] == 'Completed')]
        if sub.empty:
            print(f"\nNo completed runs for {ont}.")
            continue
            
        sub = sub.sort_values(by='Val_Macro_Fmax', ascending=False)
        
        print(f"\n{'='*75}")
        print(f"  Tuning Results: {ont} (Best configurations at the top)")
        print(f"{'='*75}")
        
        display_cols = ['LR', 'Dropout', 'BatchSize', 'Val_Macro_Fmax', 'Val_Micro_Fmax', 'Val_Smin']
        print(sub[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    aggregate_tuning("tuning_runs/", "tuning_runs/tuning_results_summary.csv")
