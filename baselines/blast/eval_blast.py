import os
import sys
import json
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from preprocessing.create_batch_dataset import PDB_Dataset
from evals import evaluate_all, compute_ic

def main():
    test_list_file = os.path.join(project_root, 'preprocessing/data/split_files/_test.txt')
    train_list_file = os.path.join(project_root, 'preprocessing/data/split_files/_train.txt')
    annot_file = os.path.join(project_root, 'preprocessing/data/pdb2go.tsv')
    pred_file = os.path.join(os.path.dirname(__file__), 'blast_predictions.json')

    print("[EVAL] Loading annotations...")
    prot2annot, goterms, gonames, prot_list = PDB_Dataset.annot_file_reader(annot_file)

    print("[EVAL] Loading splits and predictions...")
    with open(test_list_file) as fh:
        test_prots = [l.strip() for l in fh if l.strip()]
    with open(train_list_file) as fh:
        train_prots = [l.strip() for l in fh if l.strip()]

    with open(pred_file) as fh:
        predictions = json.load(fh)

    onts = ['molecular_function', 'biological_process', 'cellular_component']
    short_onts = {'molecular_function': 'mf', 'biological_process': 'bp', 'cellular_component': 'cc'}

    print("=========================================")
    print("      BLAST Baseline Evaluation          ")
    print("=========================================\n")

    for ont in onts:
        short_ont = short_onts[ont]
        print(f"--- Evaluating {ont} ---")
        
        # Calculate IC from training set
        y_train = np.array([prot2annot[p][ont] for p in train_prots if p in prot2annot])
        ic = compute_ic(y_train)

        # Build true and pred matrices for test set
        y_true = []
        y_pred = []

        num_classes = len(goterms[ont])

        for p in test_prots:
            if p in prot2annot:
                y_true.append(prot2annot[p][ont])
                
                # Build pred vector (1.0 if BLAST predicted it, 0.0 otherwise)
                pred_vec = np.zeros(num_classes, dtype=np.float32)
                if p in predictions:
                    pred_terms = predictions[p].get(short_ont, [])
                    indices = [goterms[ont].index(t) for t in pred_terms if t in goterms[ont]]
                    pred_vec[indices] = 1.0
                y_pred.append(pred_vec)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = evaluate_all(y_true, y_pred, ic)
        
        print(f"Micro_Fmax:  {metrics['Micro_Fmax']:.4f}")
        print(f"Macro_Fmax:  {metrics['Macro_Fmax']:.4f}")
        print(f"Macro_AUROC: {metrics['Macro_AUROC']:.4f}")
        print(f"Micro_AUROC: {metrics['Micro_AUROC']:.4f}")
        print(f"Macro_AUPRC: {metrics['Macro_AUPRC']:.4f}")
        print(f"Micro_AUPRC: {metrics['Micro_AUPRC']:.4f}")
        print(f"Smin:        {metrics['Smin']:.4f}")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()
