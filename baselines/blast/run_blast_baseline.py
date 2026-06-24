"""
baselines/blast/run_blast_baseline.py

BLAST nearest-neighbour annotation transfer baseline.
Requires BLAST+ (makeblastdb, blastp) installed and on PATH.
"""
import os
import sys
import subprocess
import argparse
import json
import csv


def build_blast_db(train_fasta: str, db_name: str):
    print(f"[BLAST] Building database from {train_fasta}…")
    subprocess.run([
        "makeblastdb",
        "-in",     train_fasta,
        "-dbtype", "prot",
        "-out",    db_name,
    ], check=True)


def run_blastp(query_fasta: str, db_name: str, out_file: str, evalue: float = 10.0):
    print(f"[BLAST] Running BLASTp against {db_name}…")
    subprocess.run([
        "blastp",
        "-query",           query_fasta,
        "-db",              db_name,
        "-out",             out_file,
        "-outfmt",          "6 qseqid sseqid pident bitscore evalue",
        "-evalue",          str(evalue),
        "-max_target_seqs", "1",
        "-num_threads",     str(os.cpu_count() or 4),
    ], check=True)


def load_train_annotations(train_list_file: str, annot_tsv: str) -> dict:
    """Return {prot_id: {'mf': [...], 'bp': [...], 'cc': [...]}}"""
    with open(train_list_file) as fh:
        train_set = {line.strip() for line in fh if line.strip()}

    annots = {}
    with open(annot_tsv) as fh:
        reader = csv.reader(fh, delimiter='\t')
        # Skip 6 header rows (3 ontologies × 2 rows each)
        for _ in range(6):
            next(reader, None)
        next(reader, None)  # column header row
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
    print(f"[BLAST] Loaded annotations for {len(annots)} training proteins")
    return annots


def transfer_annotations(blast_out: str, train_annots: dict, test_proteins: list) -> dict:
    """Nearest-neighbour label transfer: top-1 BLAST hit annotation."""
    predictions = {p: {'mf': [], 'bp': [], 'cc': []} for p in test_proteins}
    seen = set()
    with open(blast_out) as fh:
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
    print(f"[BLAST] Annotated {covered}/{len(test_proteins)} test proteins via top-1 hit")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_fasta', default='preprocessing/data/split_files/_train_sequences.fasta')
    parser.add_argument('-test_fasta',  default='preprocessing/data/split_files/_test_sequences.fasta')
    parser.add_argument('-train_list',  default='preprocessing/data/split_files/_train.txt')
    parser.add_argument('-test_list',   default='preprocessing/data/split_files/_test.txt')
    parser.add_argument('-train_annot', default='preprocessing/data/pdb2go.tsv')
    parser.add_argument('-out_dir',     default='baselines/blast/')
    args = parser.parse_args()

    for f in [args.train_fasta, args.test_fasta, args.train_annot]:
        if not os.path.exists(f):
            sys.exit(f"[ERROR] Required file not found: {f}\n"
                     "  Run preprocessing/cluster_and_split.py first.")

    os.makedirs(args.out_dir, exist_ok=True)
    db_path    = os.path.join(args.out_dir, "train_db")
    blast_out  = os.path.join(args.out_dir, "blast_results.tsv")
    pred_file  = os.path.join(args.out_dir, "blast_predictions.json")

    try:
        build_blast_db(args.train_fasta, db_path)
        run_blastp(args.test_fasta, db_path, blast_out)
    except FileNotFoundError:
        sys.exit("[ERROR] BLAST+ not found. Install with: conda install -c bioconda blast")

    with open(args.test_list) as fh:
        test_prots = [l.strip() for l in fh if l.strip()]

    train_annots = load_train_annotations(args.train_list, args.train_annot)
    preds = transfer_annotations(blast_out, train_annots, test_prots)

    with open(pred_file, 'w') as fh:
        json.dump(preds, fh, indent=2)
    print(f"[BLAST] Predictions saved → {pred_file}")


if __name__ == "__main__":
    main()
