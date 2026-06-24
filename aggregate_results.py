"""
aggregate_results.py
Scans runs/ for completed experiments and prints mean±std tables.
"""
import os
import json
import pandas as pd
import glob
import numpy as np


METRIC_COLS = [
    'Micro_Fmax', 'Macro_Fmax',
    'Micro_AUROC', 'Macro_AUROC',
    'Micro_AUPRC', 'Macro_AUPRC',
    'Smin',
]


def aggregate_results(runs_dir: str, output_file: str):
    results = []

    for metric_file in glob.glob(os.path.join(runs_dir, "*", "test_metrics.json")):
        run_dir    = os.path.dirname(metric_file)
        config_file = os.path.join(run_dir, "config.json")

        if not os.path.exists(config_file):
            continue

        with open(config_file)  as f: config  = json.load(f)
        with open(metric_file)  as f: metrics = json.load(f)

        row = {
            'Ontology': config.get('ontology', 'biological_process'),
            'Model':    config.get('model',    'Unknown'),
            'Loss':     config.get('loss',     'Unknown'),
            'Seed':     config.get('seed',     0),
        }
        row.update({k: metrics.get(k, float('nan')) for k in METRIC_COLS})
        results.append(row)

    if not results:
        print("No completed runs found in", runs_dir)
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Group by configuration and compute mean ± std across seeds
    group_cols = ['Ontology', 'Model', 'Loss']
    agg_funcs  = {col: ['mean', 'std'] for col in METRIC_COLS}
    grouped = df.groupby(group_cols).agg(agg_funcs)

    # Flatten the multi-level column index properly
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    grouped.to_csv(output_file, index=False)
    print(f"\nAggregated results saved to {output_file}")

    # Pretty-print summary table
    for ont in grouped['Ontology'].unique():
        sub = grouped[grouped['Ontology'] == ont]
        print(f"\n{'='*70}")
        print(f"  Ontology: {ont}  ({len(df[df['Ontology']==ont])} seeds × configs)")
        print(f"{'='*70}")

        display_cols = ['Model', 'Loss'] + [
            f"{m}_{s}" for m in ['Micro_Fmax', 'Macro_Fmax', 'Smin']
            for s in ['mean', 'std']
        ]
        print(sub[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return grouped


if __name__ == "__main__":
    aggregate_results("runs/", "runs/aggregated_results.csv")
