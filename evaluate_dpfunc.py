import os
import sys
import pickle
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals import evaluate_all, compute_ic
from preprocessing.create_batch_dataset import PDB_Dataset, _get_protbert

# Mock _get_protbert to avoid loading weights for evaluation
import preprocessing.create_batch_dataset
preprocessing.create_batch_dataset._get_protbert = lambda: (None, None, None)

def parse_dpfunc_results(result_file, goterms, prot_list):
    """Parse DPFunc results into a numpy matrix aligned with prot_list and goterms."""
    # Initialize zero probability matrix
    y_pred = np.zeros((len(prot_list), len(goterms)), dtype=np.float32)
    
    if not os.path.exists(result_file):
        print(f"File not found: {result_file}")
        return y_pred
        
    prot_idx = {prot: i for i, prot in enumerate(prot_list)}
    term_idx = {term: i for i, term in enumerate(goterms)}
    
    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                prot = parts[0]
                term = parts[1]
                score = float(parts[2])
                
                if prot in prot_idx and term in term_idx:
                    y_pred[prot_idx[prot], term_idx[term]] = score
                    
    return y_pred

def main():
    dataset_pkl = 'preprocessing/data/split_files/datasets.pkl'
    print(f"Loading datasets from {dataset_pkl}...")
    
    # We must remap numpy._core to numpy.core
    class RenameUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "numpy._core.multiarray":
                module = "numpy.core.multiarray"
            return super().find_class(module, name)
            
    with open(dataset_pkl, 'rb') as f:
        datasets = RenameUnpickler(f).load()
        
    ONTOLOGIES = ['biological_process', 'molecular_function', 'cellular_component']
    FILE_MAPPING = {
        'cellular_component': 'SOTA_predictions/DPFunc/cc_results.txt',
        'molecular_function': 'SOTA_predictions/DPFunc/mf_results.txt',
        'biological_process': 'SOTA_predictions/DPFunc/bp_results.txt'
    }
    
    results = {}
    
    for ont in ONTOLOGIES:
        print(f"\nEvaluating DPFunc on {ont}...")
        ds_test = datasets[ont]['test']
        ds_train = datasets[ont]['train']
        
        # Get goterms for the ontology
        goterms = ds_test.y_labels
        prot_list = ds_test.pdb_split_list
        
        # Load true arrays from an existing baseline run
        ont_prefix = {'biological_process': 'bp', 'molecular_function': 'mf', 'cellular_component': 'cc'}[ont]
        run_dir = f"tuning_runs_jk/{ont_prefix}_Hybrid_JK_Focal_lr0.0001_dp0.2_bs16_s42"
        y_true = np.load(os.path.join(run_dir, 'test_y_true.npy'))
        
        # Build IC
        all_train_labels = np.load(f"{ont}_train_labels.npy")
        ic = compute_ic(all_train_labels)
        
        # Parse predictions
        res_file = FILE_MAPPING[ont]
        y_pred = parse_dpfunc_results(res_file, goterms, prot_list)
        
        # Evaluate
        metrics = evaluate_all(y_true, y_pred, ic)
        results[ont] = metrics
        
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    print("\nFinal Results Summary:")
    for ont in ONTOLOGIES:
        print(f"--- {ont} ---")
        for k, v in results[ont].items():
            print(f"  {k}: {v:.4f}")

if __name__ == '__main__':
    main()
