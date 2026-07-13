import pandas as pd
import argparse
import os

def get_best(ontology, summary_file="tuning_runs/tuning_results_summary.csv"):
    if not os.path.exists(summary_file):
        return None
    try:
        df = pd.read_csv(summary_file)
        # Filter for the correct ontology and completed runs
        df = df[(df['Ontology'] == ontology) & (df['Status'] == 'Completed')]
        if df.empty:
            return None
        # Sort by Val_Macro_Fmax descending and get the top row
        best = df.sort_values('Val_Macro_Fmax', ascending=False).iloc[0]
        return best['LR'], best['Dropout'], int(best['BatchSize'])
    except Exception as e:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get best hyperparameters from tuning summary.")
    parser.add_argument('--ontology', required=True, help="Ontology name (e.g. biological_process)")
    parser.add_argument('--summary_file', default="tuning_runs/tuning_results_summary.csv", help="Path to tuning summary CSV")
    args = parser.parse_args()

    best = get_best(args.ontology, args.summary_file)
    if best:
        # Output as bash-evaluable variables
        print(f"export LR={best[0]} DROPOUT={best[1]} BATCH_SIZE={best[2]}")
    else:
        # Print nothing or export empty variables so bash knows it failed
        pass
