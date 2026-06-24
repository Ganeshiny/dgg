"""
baselines/diamond/run_diamond_baseline.py

DIAMOND nearest-neighbour annotation transfer baseline.
Requires DIAMOND installed and on PATH.
"""
import os
import sys
import subprocess
import argparse
import json
import csv


def build_diamond_db(train_fasta: str, db_name: str):
    print(f"[DIAMOND] Building database from {train_fasta}…")
    subprocess.run([
        "diamond", "makedb",
        "--in", train_fasta,
        "-d",   db_name,
    ], check=True)


def run_diamond(query_fasta: str, db_name: str, out_file: str):
    print(f"[DIAMOND] Running blastp against {db_name}…")
    subprocess.run([
        "diamond", "blastp",
        "-q",  query_fasta,
        "-d",  db_name,
        "-o",  out_file,
        "-f",  "6", "qseqid", "sseqid", "pident", "bitscore", "evalue",
        "-e",  "10",
        "--max-target-seqs", "1",
        "--more-sensitive",
        "-p", str(os.cpu_count() or 4),
    ], check=True)


def load_train_annotations(train_list_file: str, annot_tsv: str) -> dict:
    with open(train_list_file) as fh:
        train_set = {line.strip() for line in fh if line.strip()}

    annots = {}
    with open(annot_tsv) as fh:
        reader = csv.reader(fh, delimiter='\t')
        for _ in range(6):
            next(reader, None)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            prot = row[0]
            if prot in train_set:
                annots[prot] = {
                    'mf': [g for g in row[1].split(',') if g],
                    'bp': [g for g in row[2].split(',') if g],
                    'cc': [g for g in row[3].split(',') if g],
                }
    print(f"[DIAMOND] Loaded annotations for {len(annots)} training proteins")
    return annots


def transfer_annotations(result_tsv: str, train_annots: dict, test_proteins: list) -> dict:
    predictions = {p: {'mf': [], 'bp': [], 'cc': []} for p in test_proteins}
    seen = set()
    with open(result_tsv) as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            qseqid, sseqid = parts[0], parts[1]
            if qseqid in predictions and qseqid not in seen:
                seen.add(qseqid)
                if sseqid in train_annots:
                    predictions[qseqid] = train_annots[sseqid]
    covered = sum(1 for v in predictions.values() if any(v.values()))
    print(f"[DIAMOND] Annotated {covered}/{len(test_proteins)} test proteins via top-1 hit")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_fasta', default='preprocessing/data/split_files/_train_sequences.fasta')
    parser.add_argument('-test_fasta',  default='preprocessing/data/split_files/_test_sequences.fasta')
    parser.add_argument('-train_list',  default='preprocessing/data/split_files/_train.txt')
    parser.add_argument('-test_list',   default='preprocessing/data/split_files/_test.txt')
    parser.add_argument('-train_annot', default='preprocessing/data/pdb2go.tsv')
    parser.add_argument('-out_dir',     default='baselines/diamond/')
    args = parser.parse_args()

    for f in [args.train_fasta, args.test_fasta, args.train_annot]:
        if not os.path.exists(f):
            sys.exit(f"[ERROR] Required file not found: {f}\n"
                     "  Run preprocessing/cluster_and_split.py first.")

    os.makedirs(args.out_dir, exist_ok=True)
    db_path   = os.path.join(args.out_dir, "train_db")
    result_tsv = os.path.join(args.out_dir, "diamond_results.tsv")
    pred_file  = os.path.join(args.out_dir, "diamond_predictions.json")

    try:
        build_diamond_db(args.train_fasta, db_path)
        run_diamond(args.test_fasta, db_path, result_tsv)
    except FileNotFoundError:
        sys.exit("[ERROR] DIAMOND not found. Install with: conda install -c bioconda diamond")

    with open(args.test_list) as fh:
        test_prots = [l.strip() for l in fh if l.strip()]

    train_annots = load_train_annotations(args.train_list, args.train_annot)
    preds = transfer_annotations(result_tsv, train_annots, test_prots)

    with open(pred_file, 'w') as fh:
        json.dump(preds, fh, indent=2)
    print(f"[DIAMOND] Predictions saved → {pred_file}")


if __name__ == "__main__":
    main()
