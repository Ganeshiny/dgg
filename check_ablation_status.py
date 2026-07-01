import os
import glob
import json
import pandas as pd

def check_status():
    run_dirs = glob.glob("runs/*")
    
    completed = 0
    in_progress = 0
    failed_or_empty = 0
    
    results = []
    
    for rd in run_dirs:
        if not os.path.isdir(rd):
            continue
            
        dir_name = os.path.basename(rd)
        
        # Extract ontology
        if dir_name.startswith('bp_'):
            ont = 'biological_process'
        elif dir_name.startswith('mf_'):
            ont = 'molecular_function'
        elif dir_name.startswith('cc_'):
            ont = 'cellular_component'
        else:
            ont = 'unknown'
            
        parts = dir_name.split('_')
        
        # Parse model
        model = "Unknown"
        if "Hybrid_JK" in dir_name:
            model = "Hybrid_JK"
        elif "Hybrid" in dir_name:
            model = "Hybrid"
        elif "MLP" in dir_name:
            model = "MLP"
        elif "GCN" in dir_name:
            model = "GCN"
        elif "GAT" in dir_name:
            model = "GAT"
            
        is_completed = os.path.exists(os.path.join(rd, 'test_metrics.json'))
        
        if is_completed:
            completed += 1
            status = "Completed"
        else:
            if os.path.exists(os.path.join(rd, 'training_log.csv')):
                in_progress += 1
                status = "In Progress"
            else:
                failed_or_empty += 1
                status = "Empty"
                
        # Read log for max Fmax
        log_file = os.path.join(rd, 'training_log.csv')
        max_fmax = 0.0
        epochs = 0
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                if 'Valid_Macro_Fmax' in df.columns and not df.empty:
                    max_fmax = df['Valid_Macro_Fmax'].max()
                    epochs = len(df)
            except:
                pass
                
        results.append({
            'Ontology': ont,
            'Model': model,
            'Status': status,
            'Epochs': epochs,
            'Max_Fmax': max_fmax
        })
        
    print(f"Total Runs Found: {len(results)}")
    print(f"Completed: {completed}")
    print(f"In Progress: {in_progress}")
    print(f"Empty/Failed: {failed_or_empty}")
    print("-" * 50)
    
    df = pd.DataFrame(results)
    if not df.empty:
        summary = df.groupby(['Ontology', 'Status']).size().reset_index(name='Count')
        print("\nStatus by Ontology:")
        print(summary.to_string(index=False))
        
        print("\nMax Fmax by Model/Ontology so far:")
        best = df.groupby(['Ontology', 'Model'])['Max_Fmax'].max().reset_index()
        # pivot for better viewing
        pivot = best.pivot(index='Model', columns='Ontology', values='Max_Fmax').fillna(0.0)
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    check_status()
