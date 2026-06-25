import os
import json
import csv
import math
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from collections import defaultdict

# --- Evaluation Metrics ---
def get_micro_fmax(y_true, y_pred_probs):
    thresholds = np.arange(0.01, 1.0, 0.01)
    fmax = 0.0
    for t in thresholds:
        preds = (y_pred_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f = 2 * precision * recall / (precision + recall)
            if f > fmax:
                fmax = f
    return fmax

def compute_ic(y_train):
    N = y_train.shape[0]
    counts = np.sum(y_train, axis=0) + 1
    ic = -np.log2(counts / N)
    return ic

def get_smin(y_true, y_pred_probs, ic):
    thresholds = np.arange(0.01, 1.0, 0.01)
    s_min = float('inf')
    N = y_true.shape[0]
    for t in thresholds:
        preds = (y_pred_probs >= t).astype(int)
        fn_mask = (y_true == 1) & (preds == 0)
        ru = np.sum(fn_mask * ic) / N
        fp_mask = (y_true == 0) & (preds == 1)
        mi = np.sum(fp_mask * ic) / N
        s = math.sqrt(ru**2 + mi**2)
        if s < s_min:
            s_min = s
    return s_min

# --- Parsing Predictions ---
def parse_json_predictions(json_path, test_prots, go_terms, is_list_format=False):
    """
    Parses BLAST/DIAMOND (list of GOs) or Naive (dict of GO:score).
    """
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    num_prots = len(test_prots)
    num_terms = len(go_terms)
    term2idx = {g: i for i, g in enumerate(go_terms)}
    
    y_pred = np.zeros((num_prots, num_terms), dtype=np.float32)
    
    for i, prot in enumerate(test_prots):
        if prot in data:
            # We assume the JSON is structured {prot: {ontology: ...}} but these jsons might just be {prot: ...} 
            # We will handle it flexibly.
            # Actually our baselines saved it as {prot: {ont: [terms]}}
            # We need to find the correct ontology key. We'll search all sub-keys and map terms.
            for ont_key, preds in data[prot].items():
                if is_list_format:
                    for t in preds:
                        if t in term2idx:
                            y_pred[i, term2idx[t]] = 1.0
                else:
                    for t, score in preds.items():
                        if t in term2idx:
                            y_pred[i, term2idx[t]] = score
    return y_pred

def parse_deepfri_csv(csv_path, test_prots, go_terms):
    if not os.path.exists(csv_path):
        return None
    
    term2idx = {g: i for i, g in enumerate(go_terms)}
    y_pred = np.zeros((len(test_prots), len(go_terms)), dtype=np.float32)
    
    prot2idx = {p: i for i, p in enumerate(test_prots)}
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3: continue
            prot_raw = row[0]
            prot = prot_raw.replace('.npz', '')
            go_id = row[1]
            score = float(row[2])
            
            if prot in prot2idx and go_id in term2idx:
                y_pred[prot2idx[prot], term2idx[go_id]] = score
    return y_pred

# --- Bootstrapping ---
def bootstrap_metrics(y_true, y_pred, ic, n_resamples=1000):
    fmax_list = []
    smin_list = []
    N = y_true.shape[0]
    
    np.random.seed(42)
    for _ in tqdm(range(n_resamples), desc="Bootstrapping", leave=False):
        idx = np.random.choice(N, N, replace=True)
        sample_y_true = y_true[idx]
        sample_y_pred = y_pred[idx]
        
        # Guard against zero positive samples in bootstrap
        if np.sum(sample_y_true) == 0:
            continue
            
        fmax_list.append(get_micro_fmax(sample_y_true, sample_y_pred))
        smin_list.append(get_smin(sample_y_true, sample_y_pred, ic))
        
    return {
        "Fmax_Mean": np.mean(fmax_list), "Fmax_Std": np.std(fmax_list),
        "Smin_Mean": np.mean(smin_list), "Smin_Std": np.std(smin_list)
    }

def main():
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_pkl = os.path.join(proj_dir, 'preprocessing', 'data', 'split_files', 'datasets.pkl')
    out_csv = os.path.join(proj_dir, 'baselines', 'baseline_metrics.csv')
    
    with open(data_pkl, 'rb') as f:
        datasets = pickle.load(f)
        
    ontologies = {'molecular_function': 'mf', 'biological_process': 'bp', 'cellular_component': 'cc'}
    
    results = []
    
    for full_ont, short_ont in ontologies.items():
        print(f"--- Evaluating {full_ont.upper()} ---")
        train_ds = datasets[full_ont]['train']
        test_ds = datasets[full_ont]['test']
        
        # Get y_true and ic
        # train_ds contains graph data. We need the raw labels to compute IC.
        y_train = np.vstack([data.y.numpy() for data in train_ds])
        y_true = np.vstack([data.y.numpy() for data in test_ds])
        
        ic = compute_ic(y_train)
        
        test_prots = test_ds.pdb_split_list
        go_terms = test_ds.y_labels
        
        # Baselines definition
        models = {
            "Naive": parse_json_predictions(
                os.path.join(proj_dir, 'baselines', 'naive_frequency', 'naive_predictions.json'),
                test_prots, go_terms, is_list_format=False),
            "BLAST": parse_json_predictions(
                os.path.join(proj_dir, 'baselines', 'blast_diamond', 'blast_predictions.json'),
                test_prots, go_terms, is_list_format=True),
            "DIAMOND": parse_json_predictions(
                os.path.join(proj_dir, 'baselines', 'blast_diamond', 'diamond_predictions.json'),
                test_prots, go_terms, is_list_format=True),
            "DeepFRI_Seq": parse_deepfri_csv(
                os.path.join(proj_dir, 'baselines', 'deepfri_results', f'deepfri_seq_{short_ont.upper()}_predictions.csv'),
                test_prots, go_terms),
            "DeepFRI_Cmap": parse_deepfri_csv(
                os.path.join(proj_dir, 'baselines', 'deepfri_results', f'deepfri_cmap_{short_ont.upper()}_predictions.csv'),
                test_prots, go_terms)
        }
        
        for model_name, y_pred in models.items():
            if y_pred is None:
                print(f"Skipping {model_name} (predictions not found)")
                continue
                
            print(f"Bootstrapping {model_name}...")
            # Set number of resamples (100 is fast enough for baseline CI computation)
            mets = bootstrap_metrics(y_true, y_pred, ic, n_resamples=100)
            
            results.append({
                "Ontology": short_ont.upper(),
                "Model": model_name,
                "Micro_Fmax_Mean": mets['Fmax_Mean'],
                "Micro_Fmax_Std": mets['Fmax_Std'],
                "Smin_Mean": mets['Smin_Mean'],
                "Smin_Std": mets['Smin_Std']
            })
            
    # Write output
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"\nAll baseline evaluation completed! Metrics saved to {out_csv}")

if __name__ == "__main__":
    main()
