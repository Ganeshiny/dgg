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
        # pdb2go.tsv has 13 header rows:
        #   rows 1-2:  GO-terms (MF) header + term list
        #   rows 3-4:  GO-names (MF)
        #   rows 5-6:  GO-terms (BP)
        #   rows 7-8:  GO-names (BP)
        #   rows 9-10: GO-terms (CC)
        #   rows 11-12: GO-names (CC)
        #   row 13:    column header (### PDB-chain, GO-terms...)
        for _ in range(13):
            next(reader, None)
        for row in reader:
            # Each data row must have: prot_id, mf_terms, bp_terms, cc_terms
            if len(row) < 4:
                continue
            total_train += 1
            for g in row[1].split(','):
                if g:
                    term_counts['mf'][g] += 1
            for g in row[2].split(','):
                if g:
                    term_counts['bp'][g] += 1
            for g in row[3].split(','):
                if g:
                    term_counts['cc'][g] += 1

    if total_train == 0:
        print("[ERROR] No training proteins found — check pdb2go.tsv format.")
        return

    # Convert to probabilities
    term_probs = {
        'mf': {k: v / total_train for k, v in term_counts['mf'].items()},
        'bp': {k: v / total_train for k, v in term_counts['bp'].items()},
        'cc': {k: v / total_train for k, v in term_counts['cc'].items()}
    }

    print(f"  MF terms: {len(term_probs['mf'])}  |  BP terms: {len(term_probs['bp'])}  |  CC terms: {len(term_probs['cc'])}")

    test_prots = []
    with open(test_proteins_file) as f:
        test_prots = [line.strip() for line in f if line.strip()]

    predictions = {p: term_probs for p in test_prots}

    with open(out_file, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"Naive predictions saved to {out_file}  ({len(test_prots)} proteins)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_annot', type=str, default='preprocessing/data/pdb2go.tsv')
    parser.add_argument('-test_prots',  type=str, default='preprocessing/data/split_files/_test.txt')
    parser.add_argument('-out_dir',     type=str, default='baselines/naive_frequency/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "naive_predictions.json")

    if os.path.exists(args.train_annot) and os.path.exists(args.test_prots):
        run_naive_baseline(args.train_annot, args.test_prots, out_file)
    else:
        print("Missing required files for naive baseline.")
