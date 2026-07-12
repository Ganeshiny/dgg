#!/usr/bin/env python3
"""
preprocessing/calc_blast_identity.py
=====================================
Computes the maximum sequence identity of each TEST protein against
the TRAINING set using BLAST (via subprocess).

Output: preprocessing/data/blast_identity.csv   columns: prot, max_identity

Requires:
  - blastp installed (available on ARC: module load blast+)
  - preprocessing/data/split_files/datasets.pkl
  - A FASTA file of training sequences.  If not present, we extract them
    from the .pt files in the processed data folder.

Usage:
    python preprocessing/calc_blast_identity.py
    python preprocessing/calc_blast_identity.py --skip_blast   # use cached TSV only
"""

import os
import sys
import subprocess
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DATASET_PKL  = PROJECT_DIR / 'preprocessing' / 'data' / 'split_files' / 'datasets.pkl'
OUT_FASTA_TR = PROJECT_DIR / 'preprocessing' / 'data' / 'train_sequences.fasta'
OUT_FASTA_TE = PROJECT_DIR / 'preprocessing' / 'data' / 'test_sequences.fasta'
BLAST_DB     = PROJECT_DIR / 'preprocessing' / 'data' / 'blast_train_db'
BLAST_OUT    = PROJECT_DIR / 'preprocessing' / 'data' / 'blast_identity_raw.tsv'
OUT_CSV      = PROJECT_DIR / 'preprocessing' / 'data' / 'blast_identity.csv'

def load_datasets():
    class _Up(pickle.Unpickler):
        def find_class(self, m, n):
            if m == 'numpy._core.multiarray':
                m = 'numpy.core.multiarray'
            return super().find_class(m, n)
    import __main__
    from preprocessing.create_batch_dataset import PDB_Dataset
    __main__.PDB_Dataset = PDB_Dataset
    with open(DATASET_PKL, 'rb') as f:
        return _Up(f).load()

def extract_sequences_from_npz(prot_ids, npz_dir):
    """
    Extracts amino acid sequences from original .npz files.
    """
    seqs = {}
    for prot in prot_ids:
        npz_path = os.path.join(npz_dir, f'{prot}.npz')
        if not os.path.exists(npz_path):
            continue
        try:
            cmap = np.load(npz_path, allow_pickle=True)
            raw_seqres = cmap['seqres']
            seq = str(raw_seqres.item()) if raw_seqres.ndim == 0 else str(raw_seqres)
            seqs[prot] = seq
        except Exception:
            pass
    return seqs

def write_fasta(prot_seqs, out_path):
    with open(out_path, 'w') as f:
        for prot, seq in prot_seqs.items():
            f.write(f'>{prot}\n{seq}\n')
    print(f'  Written {len(prot_seqs)} sequences to {out_path}')

def run_blast(db_path, query_fasta, out_tsv, num_threads=8):
    """Run blastp and write result TSV."""
    # Format: qseqid sseqid pident length
    cmd = [
        'blastp',
        '-query',   str(query_fasta),
        '-db',      str(db_path),
        '-out',     str(out_tsv),
        '-outfmt',  '6 qseqid sseqid pident length',
        '-num_threads', str(num_threads),
        '-evalue',  '1e-3',
        '-max_hsps', '1',
    ]
    print(f'  Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)

def parse_blast_output(tsv_path):
    """Returns dict: query_prot → max_identity (0–1)."""
    df = pd.read_csv(tsv_path, sep='\t', header=None,
                     names=['qseqid', 'sseqid', 'pident', 'length'])
    max_id = df.groupby('qseqid')['pident'].max() / 100.0
    return max_id.to_dict()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_blast', action='store_true',
                        help='Skip BLAST run, parse existing TSV only')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()

    print('Loading datasets ...')
    datasets = load_datasets()

    # We only need one ontology to get the protein lists — they are the same
    train_ds = datasets['biological_process']['train']
    test_ds  = datasets['biological_process']['test']
    train_prots = train_ds.pdb_split_list
    test_prots  = test_ds.pdb_split_list
    print(f'  Train: {len(train_prots)} proteins,  Test: {len(test_prots)} proteins')

    npz_dir = train_ds.npz_dir
    if not os.path.exists(npz_dir):
        npz_dir = PROJECT_DIR / 'preprocessing' / 'data' / 'structure_files' / 'tmp_cmap_files'

    if not args.skip_blast:
        # Extract sequences
        print('Extracting sequences from .npz files ...')
        train_seqs = extract_sequences_from_npz(train_prots, npz_dir)
        test_seqs  = extract_sequences_from_npz(test_prots, npz_dir)

        if not train_seqs:
            print('WARNING: Could not extract sequences from .npz files.')
            print('Please provide train_sequences.fasta manually and re-run with --skip_blast.')
            sys.exit(1)

        write_fasta(train_seqs, OUT_FASTA_TR)
        write_fasta(test_seqs,  OUT_FASTA_TE)

        # Make BLAST database
        print('Building BLAST database from training sequences ...')
        cmd_makedb = [
            'makeblastdb',
            '-in',     str(OUT_FASTA_TR),
            '-dbtype', 'prot',
            '-out',    str(BLAST_DB),
        ]
        subprocess.run(cmd_makedb, check=True)

        # Run BLAST
        print('Running blastp ...')
        run_blast(BLAST_DB, OUT_FASTA_TE, BLAST_OUT, num_threads=args.threads)

    if not BLAST_OUT.exists():
        print(f'ERROR: {BLAST_OUT} not found. Run without --skip_blast first.')
        sys.exit(1)

    print('Parsing BLAST output ...')
    id_map = parse_blast_output(BLAST_OUT)

    # Build output CSV — assign max_identity=1.0 (same as training) if missing
    rows = []
    for p in test_prots:
        rows.append({'prot': p, 'max_identity': id_map.get(p, 0.0)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'✓ Saved {OUT_CSV}  ({len(df)} proteins)')
    print(df['max_identity'].describe())


if __name__ == '__main__':
    main()
