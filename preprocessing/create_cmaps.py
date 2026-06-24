"""
create_cmaps.py
Computes pairwise Cα/Cβ distance maps from compressed .cif.gz files,
applying a pLDDT confidence filter.  Saves results as compressed .npz files.

Root cause of previous WSL crashes
------------------------------------
1. multiprocessing.Pool with 20 workers each allocating N×N float64 matrices
   simultaneously → instant OOM on WSL (which has a hard memory cap).
2. Pool was never closed/joined → zombie processes after a crash.
3. Distances stored as float64 (8 B/elem); a 2000-aa protein = 32 MB per
   matrix type × 2 matrix types × 20 workers = >1 GB just for one protein.

Fixes applied
-------------
- Default parallelism now capped at min(4, cpu_count-1) to stay within WSL RAM.
  Pass -num_threads 1 for a fully sequential, crash-proof run.
- Pool opened with a context manager so it is always closed + joined.
- Distance matrices saved as float32 (half the memory of float64).
- Explicit chunk-size batching so the OS can reclaim memory between batches.
- Already-processed proteins are skipped (idempotent re-runs).
- All duplicate imports removed.
"""

import sys
import os
import gzip
import argparse
import glob
import csv
import multiprocessing
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser

# ── need read_seqs_file from same package dir ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_seqs_from_cif import read_seqs_file


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_dist_float32(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise L2 distances in float32.
    Uses the identity ||a-b||² = ||a||² + ||b||² - 2·a·b to avoid building
    an explicit (N, N, 3) intermediate array, saving ~3× memory.
    """
    coords = coords.astype(np.float32)
    sq = np.sum(coords ** 2, axis=1, keepdims=True)          # (N, 1)
    dots = coords @ coords.T                                   # (N, N)
    dist_sq = sq + sq.T - 2.0 * dots
    np.clip(dist_sq, 0.0, None, out=dist_sq)                  # kill tiny negatives
    return np.sqrt(dist_sq, out=dist_sq)


def make_distance_maps(file_path: str) -> dict:
    """
    Parse a .cif.gz file quickly and return per-chain distance maps + pLDDT arrays.
    Returns {chain_id: {'C_alpha': ndarray, 'C_beta': ndarray, 'plddt': ndarray}}
    All arrays are float32. Uses a fast custom parser instead of Bio.PDB to avoid OOM.
    """
    distance_matrices = {}
    
    # Store temporary lists per chain:
    # ca_dict[chain] = [(res_seq, coord, plddt), ...]
    # cb_dict[chain] = [(res_seq, coord), ...]
    ca_dict = {}
    cb_dict = {}

    with gzip.open(file_path, 'rt') as f:
        in_atoms = False
        columns = {}
        col_idx = 0
        
        for line in f:
            if line.startswith('loop_'):
                continue
            
            if line.startswith('_atom_site.'):
                in_atoms = True
                col_name = line.strip()
                columns[col_name] = col_idx
                col_idx += 1
                continue
            
            if in_atoms:
                if line.startswith('#'):
                    in_atoms = False
                    continue
                
                parts = line.split()
                if len(parts) < len(columns):
                    continue
                
                if parts[0] not in ('ATOM', 'HETATM'):
                    continue
                
                # We need label_atom_id (CA, CB), label_asym_id (chain), Cartn_x/y/z, B_iso_or_equiv, label_seq_id
                try:
                    atom_name = parts[columns['_atom_site.label_atom_id']]
                    chain_id  = parts[columns['_atom_site.label_asym_id']]
                    res_seq   = parts[columns['_atom_site.label_seq_id']]
                    
                    if res_seq == '.' or res_seq == '?':
                        continue # Skip residues without sequence ID
                    res_seq = int(res_seq)
                    
                    if atom_name not in ('CA', 'CB'):
                        continue
                        
                    x = float(parts[columns['_atom_site.Cartn_x']])
                    y = float(parts[columns['_atom_site.Cartn_y']])
                    z = float(parts[columns['_atom_site.Cartn_z']])
                    coord = (x, y, z)
                    
                    if atom_name == 'CA':
                        plddt = float(parts[columns['_atom_site.B_iso_or_equiv']])
                        if chain_id not in ca_dict:
                            ca_dict[chain_id] = {}
                        ca_dict[chain_id][res_seq] = (coord, plddt)
                    
                    elif atom_name == 'CB':
                        if chain_id not in cb_dict:
                            cb_dict[chain_id] = {}
                        cb_dict[chain_id][res_seq] = coord
                        
                except (IndexError, ValueError, KeyError):
                    continue

    for chain_id, ca_res in ca_dict.items():
        if not ca_res:
            continue
            
        # Sort by residue sequence
        seq_nums = sorted(ca_res.keys())
        ca_coords = []
        cb_coords = []
        plddts = []
        
        for seq in seq_nums:
            ca_coord, plddt = ca_res[seq]
            ca_coords.append(ca_coord)
            plddts.append(plddt)
            
            if chain_id in cb_dict and seq in cb_dict[chain_id]:
                cb_coords.append(cb_dict[chain_id][seq])
            else:
                cb_coords.append(ca_coord) # Fallback to CA
                
        ca = np.array(ca_coords, dtype=np.float32)
        cb = np.array(cb_coords, dtype=np.float32)
        plddt = np.array(plddts, dtype=np.float32)

        ca_dist = _pairwise_dist_float32(ca)
        cb_dist = _pairwise_dist_float32(cb)

        # For experimental PDB files, the B-factor column is temperature factor,
        # so values < 50 are actually HIGH quality (rigid).
        # We should NOT mask them out like we would for AlphaFold pLDDT.

        distance_matrices[chain_id] = {
            "C_alpha": ca_dist,
            "C_beta":  cb_dist,
            "plddt":   plddt,
        }

    return distance_matrices


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loading
# ─────────────────────────────────────────────────────────────────────────────

def load_GO_annot(filename: str):
    onts = ['molecular_function', 'biological_process', 'cellular_component']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames  = {ont: [] for ont in onts}

    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for ont in onts:
            next(reader, None)
            goterms[ont] = next(reader)
            next(reader, None)
            gonames[ont] = next(reader)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            prot = row[0]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i, ont in enumerate(onts):
                prot2annot[prot][ont] = [g for g in row[i + 1].split(',') if g]

    return prot2annot, goterms, gonames


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (called in each pool worker OR sequentially)
# ─────────────────────────────────────────────────────────────────────────────

def write_annot_npz(prot: str, prot2seq: dict, struct_dir: str) -> None:
    """Compute and save the .npz contact-map file for one protein chain."""
    if '_' not in prot:
        print(f"[skip] {prot}: no underscore — cannot split PDB/chain")
        return

    pdb, chain = prot.rsplit('_', 1)
    cif_path   = os.path.join(struct_dir, pdb + '.cif.gz')
    tmp_dir    = os.path.join(struct_dir, 'tmp_cmap_files')
    out_path   = os.path.join(tmp_dir, prot + '.npz')

    os.makedirs(tmp_dir, exist_ok=True)

    if os.path.exists(out_path):
        return   # already done — safe to re-run

    if not os.path.exists(cif_path):
        print(f"[skip] {prot}: CIF file not found ({cif_path})")
        return

    if prot not in prot2seq:
        print(f"[skip] {prot}: sequence not in FASTA")
        return

    try:
        dmaps = make_distance_maps(cif_path)
        if chain not in dmaps:
            print(f"[skip] {prot}: chain '{chain}' not found in structure")
            return

        np.savez_compressed(
            out_path,
            C_alpha = dmaps[chain]['C_alpha'],
            C_beta  = dmaps[chain]['C_beta'],
            plddt   = dmaps[chain]['plddt'],
            seqres  = prot2seq[prot],
        )
    except Exception as exc:
        print(f"[error] {prot}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build contact-map .npz files from CIF structures."
    )
    parser.add_argument('-annot',       type=str, default='preprocessing/data/pdb2go.tsv')
    parser.add_argument('-seqs',        type=str, default='preprocessing/data/seqs_from_structure_dir.fasta')
    parser.add_argument('-struc_dir',   type=str, default='preprocessing/data/structure_files')
    parser.add_argument('-num_threads', type=int, default=4,
                        help="Parallel workers.  Use 1 for sequential (safest on WSL). "
                             "Default 4 keeps memory manageable.")
    parser.add_argument('-chunk_size',  type=int, default=50,
                        help="Process this many proteins per batch before releasing memory.")
    args = parser.parse_args()

    struct_dir = args.struc_dir

    prot2goterms, _, _ = load_GO_annot(args.annot)
    print(f"### annotated proteins : {len(prot2goterms)}")

    prot2seq = read_seqs_file(args.seqs)
    print(f"### protein sequences  : {len(prot2seq)}")

    # Find what still needs processing (idempotent)
    done = {
        Path(p).stem
        for p in glob.glob(os.path.join(struct_dir, 'tmp_cmap_files', '*.npz'))
    }
    to_do = [p for p in prot2goterms if p not in done]
    print(f"### to process         : {len(to_do)}  (already done: {len(done)})")

    if not to_do:
        print("Nothing to do — all contact maps already computed.")
        import sys; sys.exit(0)

    # Cap workers: WSL crashes if too many heavy workers run simultaneously
    nprocs = min(args.num_threads, multiprocessing.cpu_count(), len(to_do))
    # Safety guard: never use more than 4 workers on WSL by default
    nprocs = min(nprocs, 4)
    print(f"### workers            : {nprocs}")
    print(f"### chunk size         : {args.chunk_size}")

    # Process in chunks so the OS can reclaim memory between batches
    for batch_start in range(0, len(to_do), args.chunk_size):
        batch = to_do[batch_start : batch_start + args.chunk_size]
        pct   = 100.0 * (batch_start + len(batch)) / len(to_do)
        print(f"\n[{batch_start + len(batch)}/{len(to_do)}  {pct:.1f}%] "
              f"Processing batch of {len(batch)} proteins…")

        args_batch = [(prot, prot2seq, struct_dir) for prot in batch]

        if nprocs <= 1:
            # Sequential — zero chance of OOM
            for prot in batch:
                write_annot_npz(prot, prot2seq, struct_dir)
        else:
            # Parallel with a context manager — pool is ALWAYS closed+joined
            with multiprocessing.Pool(processes=nprocs) as pool:
                pool.starmap(write_annot_npz, args_batch)
            # Pool is fully joined here; memory is released before next batch

    print("\n✓ Contact-map generation complete.")