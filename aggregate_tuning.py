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

    for log_file in glob.glob(os.path.join(runs_dir, "*", "training_log.csv")):
        run_dir = os.path.dirname(log_file)
        dir_name = os.path.basename(run_dir)
        
        # Example dir_name: bp_Hybrid_JK_Focal_lr0.0001_dp0.2_bs16_s42
        parts = dir_name.split('_')
        
        # Handle ontology prefix mapping
        ont_map = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
        ontology = ont_map.get(parts[0], 'Unknown')
        
        # Parse params from dir name
        lr = 'Unknown'
        dp = 'Unknown'
        bs = 'Unknown'
        seed = 'Unknown'
        for part in parts:
            if part.startswith('lr'): lr = part[2:]
            elif part.startswith('dp'): dp = part[2:]
            elif part.startswith('bs'): bs = part[2:]
            elif part.startswith('s') and part[1:].isdigit(): seed = part[1:]
            
        row = {
            'Ontology': ontology,
            'Model':    'Hybrid_JK' if 'JK' in dir_name else 'Hybrid',
            'LR':       lr,
            'Dropout':  dp,
            'BatchSize': bs,
            'Seed':     seed,
        }

        # Get best metrics from training_log.csv
        try:
            log_df = pd.read_csv(log_file)
            if not log_df.empty and 'Valid_Macro_Fmax' in log_df.columns:
                best_idx = log_df['Valid_Macro_Fmax'].idxmax()
                best_row = log_df.loc[best_idx]
                row.update({
                    'Val_Micro_Fmax': best_row['Valid_Micro_Fmax'],
                    'Val_Macro_Fmax': best_row['Valid_Macro_Fmax'],
                    'Val_Smin':       best_row['Valid_Smin'],
                })
                row['Status'] = 'Completed'
            else:
                row.update({
                    'Val_Micro_Fmax': float('nan'),
                    'Val_Macro_Fmax': float('nan'),
                    'Val_Smin':       float('nan'),
                })
                row['Status'] = 'Failed/Incomplete'
        except Exception as e:
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
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate tuning results")
    parser.add_argument("--runs_dir", type=str, default="tuning_runs/", help="Directory containing tuning runs")
    parser.add_argument("--output_file", type=str, default="tuning_runs/tuning_results_summary.csv", help="Path to output CSV")
    args = parser.parse_args()
    aggregate_tuning(args.runs_dir, args.output_file)
