"""
preprocessing/pdb_clusters/split_by_pdb_clusters.py
====================================================
Splits Viridiplantae chain-level protein IDs into train / valid / test
using PDB's weekly DIAMOND sequence clusters, with the following guarantees:

  1. UNION-FIND over chain-level IDs before assigning to splits, so that every
     chain belonging to the same PDB entry that spans multiple entity clusters
     is transitively merged into one super-cluster.  Uses entity_map.json built
     by build_entity_map.py.

  2. BIN-PACKING by protein *count* (not cluster count) using a greedy shuffle
     + fill approach, with a fixed seed.  Achieved vs target ratios are logged.

  3. UNCLUSTERED IDs (not in any RCSB cluster file) are treated as singletons,
     logged, and included in the split (assigned to train by default unless
     they fill valid/test).

  4. Does NOT touch existing split files or data.  Outputs go to a dedicated
     subdirectory: preprocessing/data/pdb_split_<threshold>/

  5. Runs for thresholds in {30, 40, 50, 70, 90, 95}.  100 is excluded as it
     provides no meaningful separation.

Usage (single threshold):
    python3 preprocessing/pdb_clusters/split_by_pdb_clusters.py --threshold 30

Usage (all thresholds):
    python3 preprocessing/pdb_clusters/split_by_pdb_clusters.py --all
"""

import json
import os
import pickle
import sys
import argparse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = PROJECT_DIR / 'preprocessing' / 'data'
ENTITY_MAP  = DATA_DIR / 'entity_map.json'
CSV_IDS     = PROJECT_DIR / 'preprocessing' / 'viridiplantae_pdb_ids_2024-06-25.csv'
FASTA_IN    = DATA_DIR / 'all_sequences.fasta'

THRESHOLDS  = [30, 40, 50, 70, 90, 95]
CLUSTER_URL = 'https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{t}.txt'

TRAIN_FRAC = 0.80
VALID_FRAC = 0.10
SEED       = 42


# ── Union-Find ──────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        """Return root -> [members]."""
        g: dict[str, list[str]] = defaultdict(list)
        for x in self._parent:
            g[self.find(x)].append(x)
        return dict(g)


# ── FASTA reader ─────────────────────────────────────────────────────────────

def read_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    key = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                key = line[1:].split()[0]
            elif key:
                seqs[key] = seqs.get(key, '') + line
    return seqs


def write_fasta_subset(seqs: dict, ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        for pid in ids:
            if pid in seqs:
                fh.write(f'>{pid}\n{seqs[pid]}\n')


# ── Cluster file download ─────────────────────────────────────────────────────

def download_cluster_file(threshold: int) -> list[list[str]]:
    """Download the cluster file and return list-of-lists of entity IDs."""
    url = CLUSTER_URL.format(t=threshold)
    print(f'  Downloading: {url}')
    clusters: list[list[str]] = []
    with urllib.request.urlopen(url) as resp:
        for line in resp:
            members = line.decode('utf-8').strip().split()
            if members:
                clusters.append(members)
    print(f'  Downloaded {len(clusters):,} clusters.')
    return clusters


# ── Main split function ───────────────────────────────────────────────────────

def run_split(
    threshold: int,
    all_chain_ids: list[str],
    entity_map: dict[str, dict[str, str]],
    seqs: dict[str, str],
    go_table: dict[str, dict[str, list[str]]],   # chain_id -> {ont: [go_terms]}
) -> dict:
    """
    Build a train/valid/test split using PDB clusters at `threshold`% identity.
    Returns a dict with split lists and diagnostic stats.
    """
    out_dir = DATA_DIR / f'pdb_split_{threshold}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Map chain IDs -> entity IDs ─────────────────────────────────
    chain_to_entity: dict[str, str | None] = {}   # chain_id -> 'PDBID_entitynum' or None
    missing_entity: list[str] = []

    for chain_id in all_chain_ids:
        pdb_id  = chain_id.split('_')[0].upper()
        auth_ch = chain_id[len(pdb_id) + 1:]
        pdb_map = entity_map.get(pdb_id, {})
        ent_num = pdb_map.get(auth_ch)
        if ent_num is not None:
            chain_to_entity[chain_id] = f'{pdb_id}_{ent_num}'
        else:
            chain_to_entity[chain_id] = None
            missing_entity.append(chain_id)

    print(f'  Chain IDs with entity mapping: {len(all_chain_ids) - len(missing_entity):,}')
    print(f'  Chain IDs without entity mapping: {len(missing_entity):,}')
    if missing_entity[:5]:
        print(f'    e.g. {missing_entity[:5]}')

    # ── Step 2: Download and index PDB clusters ──────────────────────────────
    clusters = download_cluster_file(threshold)

    # Build entity_id -> cluster_idx
    entity_to_cluster: dict[str, int] = {}
    for cidx, members in enumerate(clusters):
        for m in members:
            entity_to_cluster[m] = cidx

    # ── Step 3: Union-Find — merge all chain IDs that must stay together ─────
    # Logic:
    #   a) All chains of the same PDB ID share at least the PDB-entry constraint.
    #   b) Additionally merge via the PDB cluster membership (by entity ID).
    uf = UnionFind()

    # (a) Pre-seed: all chains of the same PDB ID union together
    pdb_to_chains: dict[str, list[str]] = defaultdict(list)
    for cid in all_chain_ids:
        pdb_to_chains[cid.split('_')[0].upper()].append(cid)

    for pdb_id, chains in pdb_to_chains.items():
        for c in chains[1:]:
            uf.union(chains[0], c)

    # (b) Merge via cluster membership: if two chain IDs map to the same entity
    #     cluster, they must be in the same super-cluster.
    cluster_to_first_chain: dict[int, str] = {}   # cluster_idx -> representative chain
    unclustered: list[str] = []

    for cid in all_chain_ids:
        entity_id = chain_to_entity.get(cid)
        if entity_id is None:
            unclustered.append(cid)
            continue
        cidx = entity_to_cluster.get(entity_id)
        if cidx is None:
            unclustered.append(cid)
            continue
        if cidx in cluster_to_first_chain:
            uf.union(cluster_to_first_chain[cidx], cid)
        else:
            cluster_to_first_chain[cidx] = cid

    print(f'  Chains with no cluster assignment (treated as singletons): {len(unclustered):,}')

    # ── Step 4: Extract super-cluster groups ─────────────────────────────────
    groups = uf.groups()   # root -> [chain_ids]
    super_clusters = list(groups.values())
    print(f'  Super-clusters after union-find: {len(super_clusters):,}')
    sizes = sorted([len(g) for g in super_clusters], reverse=True)
    print(f'  Largest super-cluster: {sizes[0]:,} chains')
    print(f'  Median super-cluster size: {sizes[len(sizes)//2]:,}')
    print(f'  Singletons: {sizes.count(1):,}')

    # ── Step 5: Bin-pack by protein count ────────────────────────────────────
    total_prots = sum(len(g) for g in super_clusters)
    train_target = int(total_prots * TRAIN_FRAC)
    valid_target = int(total_prots * VALID_FRAC)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(super_clusters)).tolist()

    train_list: list[str] = []
    valid_list: list[str] = []
    test_list:  list[str] = []

    for idx in order:
        g = super_clusters[idx]
        if len(test_list) < (total_prots - train_target - valid_target):
            test_list.extend(g)
        elif len(valid_list) < valid_target:
            valid_list.extend(g)
        else:
            train_list.extend(g)

    ach_tr = len(train_list) / total_prots
    ach_va = len(valid_list) / total_prots
    ach_te = len(test_list)  / total_prots

    print(f'  Split sizes (target  {TRAIN_FRAC:.0%}/{VALID_FRAC:.0%}/{1-TRAIN_FRAC-VALID_FRAC:.0%}):')
    print(f'    Train: {len(train_list):>6,}  ({ach_tr:.1%})')
    print(f'    Valid: {len(valid_list):>6,}  ({ach_va:.1%})')
    print(f'    Test:  {len(test_list):>6,}  ({ach_te:.1%})')

    # ── Step 6: GO label coverage per split ──────────────────────────────────
    def count_go_coverage(ids: list[str], ont: str) -> dict:
        term_counts: dict[str, int] = defaultdict(int)
        proteins_with_any = 0
        for cid in ids:
            terms = go_table.get(cid, {}).get(ont, [])
            if terms:
                proteins_with_any += 1
            for t in terms:
                term_counts[t] += 1
        return {
            'n_proteins_with_labels': proteins_with_any,
            'n_unique_terms': len(term_counts),
            'term_counts': dict(term_counts),
        }

    onts = ['molecular_function', 'biological_process', 'cellular_component']
    go_coverage: dict[str, dict] = {}
    for split_name, split_ids in [('train', train_list), ('valid', valid_list), ('test', test_list)]:
        go_coverage[split_name] = {ont: count_go_coverage(split_ids, ont) for ont in onts}

    # ── Step 7: Write outputs ─────────────────────────────────────────────────
    def write_list(ids: list[str], name: str) -> None:
        p = out_dir / f'_{name}.txt'
        with open(p, 'w') as fh:
            fh.write('\n'.join(ids) + '\n')
        print(f'  Wrote {p.name}: {len(ids):,} IDs')

    write_list(train_list, 'train')
    write_list(valid_list, 'valid')
    write_list(test_list,  'test')

    # FASTA subsets
    write_fasta_subset(seqs, train_list, out_dir / '_train_sequences.fasta')
    write_fasta_subset(seqs, valid_list, out_dir / '_valid_sequences.fasta')
    write_fasta_subset(seqs, test_list,  out_dir / '_test_sequences.fasta')
    print(f'  Wrote per-split FASTA files.')

    # Log file
    log_path = out_dir / 'split_log.json'
    log = {
        'threshold': threshold,
        'seed': SEED,
        'train_frac_target': TRAIN_FRAC,
        'valid_frac_target': VALID_FRAC,
        'total_proteins': total_prots,
        'n_super_clusters': len(super_clusters),
        'achieved_fracs': {'train': round(ach_tr, 4), 'valid': round(ach_va, 4), 'test': round(ach_te, 4)},
        'split_sizes': {'train': len(train_list), 'valid': len(valid_list), 'test': len(test_list)},
        'unclustered_chains': len(unclustered),
        'chains_without_entity_map': len(missing_entity),
        'go_coverage': {
            split: {ont: {
                'n_proteins_with_labels': v['n_proteins_with_labels'],
                'n_unique_terms': v['n_unique_terms'],
            } for ont, v in cov.items()}
            for split, cov in go_coverage.items()
        },
    }
    with open(log_path, 'w') as fh:
        json.dump(log, fh, indent=2)
    print(f'  Wrote split log → {log_path}')

    # Pickle with full GO coverage (including per-term counts) for plots
    cov_pkl = out_dir / 'go_coverage_full.pkl'
    with open(cov_pkl, 'wb') as fh:
        pickle.dump(go_coverage, fh)

    return {
        'threshold': threshold,
        'train': train_list,
        'valid': valid_list,
        'test':  test_list,
        'go_coverage': go_coverage,
        'log': log,
    }


# ── GO table loader ──────────────────────────────────────────────────────────

def load_go_table() -> dict[str, dict[str, list[str]]]:
    """Load pdb2go.tsv into {chain_id: {ont: [terms]}}."""
    pdb2go_path = DATA_DIR / 'pdb2go.tsv'
    print(f'Loading GO table from {pdb2go_path} …')
    with open(pdb2go_path) as fh:
        content = fh.read()

    sections = content.split('###')
    # Section 7: PDB-chain to GO terms
    pdb_section = sections[7]
    lines = pdb_section.strip().split('\n')
    # Header: PDB-chain \t GO-terms (mf) \t GO-terms (bp) \t GO-terms (cc)
    go_table: dict[str, dict[str, list[str]]] = {}
    ONT_MAP = {
        1: 'molecular_function',
        2: 'biological_process',
        3: 'cellular_component',
    }
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        chain_id = parts[0]
        go_table[chain_id] = {}
        for col, ont in ONT_MAP.items():
            raw = parts[col].strip()
            if raw:
                go_table[chain_id][ont] = raw.split(',')
            else:
                go_table[chain_id][ont] = []
    print(f'  Loaded GO annotations for {len(go_table):,} chains.')
    return go_table


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Split Viridiplantae data using PDB clusters.')
    parser.add_argument('--threshold', type=int, choices=THRESHOLDS,
                        help='Single identity threshold to run.')
    parser.add_argument('--all', action='store_true',
                        help='Run all thresholds: ' + str(THRESHOLDS))
    args = parser.parse_args()

    if not args.all and args.threshold is None:
        parser.error('Specify --threshold N or --all')

    # Validate prerequisite files
    for p in [ENTITY_MAP, CSV_IDS, FASTA_IN]:
        if not p.exists():
            sys.exit(f'[ERROR] Required file not found: {p}\n'
                     f'Run build_entity_map.py first if entity_map.json is missing.')

    # Load shared data once
    print('Loading entity map …')
    with open(ENTITY_MAP) as fh:
        entity_map: dict[str, dict[str, str]] = json.load(fh)
    print(f'  Entity map covers {len(entity_map):,} PDB IDs.')

    print('Loading Viridiplantae chain IDs from split files …')
    all_chain_ids: list[str] = []
    for split_file in ['_train.txt', '_valid.txt', '_test.txt']:
        p = DATA_DIR / 'split_files' / split_file
        with open(p) as fh:
            all_chain_ids.extend(line.strip() for line in fh if line.strip())
    print(f'  Total chain IDs: {len(all_chain_ids):,}')

    print('Loading GO table …')
    go_table = load_go_table()

    print('Loading FASTA sequences …')
    seqs = read_fasta(FASTA_IN)
    print(f'  Loaded {len(seqs):,} sequences.')

    thresholds = THRESHOLDS if args.all else [args.threshold]
    for t in thresholds:
        print(f'\n{"="*60}')
        print(f'  Running split at {t}% sequence identity')
        print(f'{"="*60}')
        run_split(t, all_chain_ids, entity_map, seqs, go_table)

    print('\nDone.')


if __name__ == '__main__':
    main()
