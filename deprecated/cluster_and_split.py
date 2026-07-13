"""
cluster_and_split.py
Clusters all sequences with MMseqs2 at 30% identity, then splits the
*clusters* (not individual sequences) into train / valid / test sets.
This guarantees zero sequence-homology leakage across splits.

Also exports per-split FASTA files needed by BLAST / DIAMOND baselines.

Usage:
    python3 preprocessing/cluster_and_split.py \
        -fasta  preprocessing/data/all_sequences.fasta \
        -prefix preprocessing/data/ \
        -seq_id 0.30 \
        -cov    0.80 \
        -seed   42
"""

import os
import sys
import argparse
import subprocess
import pickle
import numpy as np


def write_prot_list(protein_list, filename):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w') as fh:
        for p in protein_list:
            fh.write(f"{p}\n")


def write_fasta_subset(seqs: dict, subset: list, out_path: str):
    """Write a subset of sequences to a FASTA file."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as fh:
        for prot in subset:
            if prot in seqs:
                fh.write(f">{prot}\n{seqs[prot]}\n")


def read_fasta(path: str) -> dict:
    seqs = {}
    current_key = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                current_key = line[1:].split()[0]  # take only the ID field
            elif current_key:
                seqs[current_key] = seqs.get(current_key, '') + line
    return seqs


def run_clustering_and_split(
    fasta_in: str,
    prefix: str,
    seq_id: float = 0.3,
    cov: float = 0.8,
    seed: int = 42,
    train_frac: float = 0.80,
    valid_frac: float = 0.10,
):
    """
    Run MMseqs2 easy-cluster then split clusters into train/valid/test.
    Returns (prot2cluster, train_list, valid_list, test_list).
    """
    split_dir  = os.path.join(prefix, "split_files")
    tmp_dir    = os.path.join(prefix, "mmseqs_tmp")
    res_prefix = os.path.join(prefix, "clusterRes")

    os.makedirs(split_dir, exist_ok=True)

    # ── Run MMseqs2 ────────────────────────────────────────────────────────
    cluster_tsv = res_prefix + "_cluster.tsv"
    if os.path.exists(cluster_tsv):
        print(f"[INFO] Cluster file already exists — skipping MMseqs2: {cluster_tsv}")
    else:
        print(f"[INFO] Running MMseqs2 easy-cluster (id={seq_id}, cov={cov})…")
        try:
            subprocess.run([
                "mmseqs", "easy-cluster",
                fasta_in, res_prefix, tmp_dir,
                "--min-seq-id", str(seq_id),
                "-c", str(cov),
                "--cov-mode", "0",
                "-v", "1",
            ], check=True)
        except FileNotFoundError:
            sys.exit(
                "[ERROR] mmseqs not found. Install with:\n"
                "  conda install -c conda-forge -c bioconda mmseqs2"
            )

    if not os.path.exists(cluster_tsv):
        sys.exit(f"[ERROR] Expected cluster file not found: {cluster_tsv}")

    # ── Parse cluster file ─────────────────────────────────────────────────
    prot2cluster  = {}   # member → representative
    cluster2prots = {}   # representative → [members]

    with open(cluster_tsv) as fh:
        for line in fh:
            rep, member = line.strip().split('\t')
            prot2cluster[member] = rep
            cluster2prots.setdefault(rep, []).append(member)

    n_clusters = len(cluster2prots)
    n_prots    = len(prot2cluster)
    print(f"[INFO] Clusters: {n_clusters}  |  Total proteins: {n_prots}")

    # Save cluster mapping for per-cluster evaluation
    mapping_path = os.path.join(split_dir, 'cluster_mapping.pkl')
    with open(mapping_path, 'wb') as fh:
        pickle.dump(prot2cluster, fh)
    print(f"[INFO] Cluster mapping saved → {mapping_path}")

    # ── Cluster-level split ────────────────────────────────────────────────
    cluster_ids = list(cluster2prots.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(cluster_ids)

    test_target  = int(n_prots * (1.0 - train_frac - valid_frac))
    valid_target = int(n_prots * valid_frac)

    train_list, valid_list, test_list = [], [], []

    for cid in cluster_ids:
        members = cluster2prots[cid]
        if len(test_list) < test_target:
            test_list.extend(members)
        elif len(valid_list) < valid_target:
            valid_list.extend(members)
        else:
            train_list.extend(members)

    print(f"[INFO] Split → Train: {len(train_list)} | "
          f"Valid: {len(valid_list)} | Test: {len(test_list)}")

    # ── Write split list files ─────────────────────────────────────────────
    write_prot_list(train_list, os.path.join(split_dir, '_train.txt'))
    write_prot_list(valid_list, os.path.join(split_dir, '_valid.txt'))
    write_prot_list(test_list,  os.path.join(split_dir, '_test.txt'))
    print(f"[INFO] Split lists saved to {split_dir}/")

    # ── Write per-split FASTA files (for BLAST / DIAMOND baselines) ────────
    print("[INFO] Writing per-split FASTA files for sequence-based baselines…")
    seqs = read_fasta(fasta_in)

    write_fasta_subset(seqs, train_list, os.path.join(split_dir, '_train_sequences.fasta'))
    write_fasta_subset(seqs, valid_list, os.path.join(split_dir, '_valid_sequences.fasta'))
    write_fasta_subset(seqs, test_list,  os.path.join(split_dir, '_test_sequences.fasta'))
    print("[INFO] Per-split FASTA files written.")

    return prot2cluster, train_list, valid_list, test_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="MMseqs2-based clustering and cluster-aware dataset splitting."
    )
    parser.add_argument('-fasta',   type=str, default='preprocessing/data/all_sequences.fasta',
                        help="Input FASTA file.")
    parser.add_argument('-prefix',  type=str, default='preprocessing/data/',
                        help="Output prefix directory.")
    parser.add_argument('-seq_id',  type=float, default=0.30,
                        help="Minimum sequence identity for clustering.")
    parser.add_argument('-cov',     type=float, default=0.80,
                        help="Minimum coverage for clustering.")
    parser.add_argument('-seed',    type=int,   default=42,
                        help="Random seed for reproducible splitting.")
    args = parser.parse_args()

    run_clustering_and_split(
        fasta_in=args.fasta,
        prefix=args.prefix,
        seq_id=args.seq_id,
        cov=args.cov,
        seed=args.seed,
    )
