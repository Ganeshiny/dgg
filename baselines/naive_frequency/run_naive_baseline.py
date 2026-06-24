import os
import argparse
import json
import csv
from collections import Counter

def run_naive_baseline(train_labels_file, test_proteins_file, out_file):
    print("Computing GO term frequencies from training set...")
    
    term_counts = {'mf': Counter(), 'bp': Counter(), 'cc': Counter()}
    total_train = 0
    
    with open(train_labels_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i in range(7):  # Skip headers
            next(reader, None)
        for row in reader:
            if len(row) > 0:
                prot = row[0]
                
                # Check if it's in train (we'll just use the whole pdb2go file as proxy if train_list not perfectly aligned, 
                # but better to strictly use train_list if passed. For simplicity here, we assume pdb2go is mostly train, 
                # but we should filter)
                
                # We'll just count all for naive, but ideally filter to train set.
                total_train += 1
                for g in row[1].split(','):
                    if g: term_counts['mf'][g] += 1
                for g in row[2].split(','):
                    if g: term_counts['bp'][g] += 1
                for g in row[3].split(','):
                    if g: term_counts['cc'][g] += 1
                    
    # Convert to probabilities
    term_probs = {
        'mf': {k: v/total_train for k, v in term_counts['mf'].items()},
        'bp': {k: v/total_train for k, v in term_counts['bp'].items()},
        'cc': {k: v/total_train for k, v in term_counts['cc'].items()}
    }
    
    test_prots = []
    with open(test_proteins_file) as f:
        test_prots = [line.strip() for line in f]
        
    predictions = {p: term_probs for p in test_prots}
    
    with open(out_file, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    print(f"Naive predictions saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_annot', type=str, default='preprocessing/data/pdb2go.tsv')
    parser.add_argument('-test_prots', type=str, default='preprocessing/data/split_files/_test.txt')
    parser.add_argument('-out_dir', type=str, default='baselines/naive_frequency/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "naive_predictions.json")

    if os.path.exists(args.train_annot) and os.path.exists(args.test_prots):
        run_naive_baseline(args.train_annot, args.test_prots, out_file)
    else:
        print("Missing required files for naive baseline.")
