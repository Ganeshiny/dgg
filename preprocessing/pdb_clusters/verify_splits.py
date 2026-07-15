#!/usr/bin/env python3
"""Verify split disjointness, cluster-component isolation, and exact sequence leakage."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from common import DATA_DIR, PDB_CLUSTER_DIR, SPLIT_ROOT, THRESHOLDS, read_fasta, read_id_file

class UnionFind:
    def __init__(self, items): self.parent = {item: item for item in items}
    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.parent[rb] = ra

def main():
    fasta = read_fasta(DATA_DIR / 'all_sequences.fasta')
    entity_map = json.loads((DATA_DIR / 'entity_map.json').read_text())
    report = {}
    for threshold in THRESHOLDS:
        root = SPLIT_ROOT / f'threshold_{threshold}'
        split = {name: set(read_id_file(root / f'_{name}.txt')) for name in ('train','valid','test')}
        if split['train'] & split['valid'] or split['train'] & split['test'] or split['valid'] & split['test']:
            raise SystemExit(f'{threshold}%: split overlap')
        if set().union(*split.values()) != set(fasta):
            raise SystemExit(f'{threshold}%: split universe differs from all_sequences.fasta')
        labels = {item: name for name, ids in split.items() for item in ids}
        uf = UnionFind(fasta)
        by_pdb = defaultdict(list)
        for chain in fasta:
            pdb, auth_chain = chain.split('_', 1)
            by_pdb[pdb].append(chain)
            entity = entity_map.get(pdb, {}).get(auth_chain)
            if entity is not None:
                labels_entity = f'{pdb}_{entity}'
                labels.setdefault(chain, labels[chain])
        for chains in by_pdb.values():
            for chain in chains[1:]: uf.union(chains[0], chain)
        entity_to_chain = {}
        for chain in fasta:
            pdb, auth_chain = chain.split('_', 1)
            entity = entity_map.get(pdb, {}).get(auth_chain)
            if entity is not None: entity_to_chain.setdefault(f'{pdb}_{entity}', chain)
        override = __import__('os').environ.get(f'DGG_CLUSTER_FILE_{threshold}')
        cluster_file = Path(override).expanduser().resolve() if override else PDB_CLUSTER_DIR / f'clusters-by-entity-{threshold}.txt'
        for line in cluster_file.read_text().splitlines():
            members = [entity_to_chain[e] for e in line.split() if e in entity_to_chain]
            for chain in members[1:]: uf.union(members[0], chain)
        components = defaultdict(set)
        for chain in fasta: components[uf.find(chain)].add(labels[chain])
        crossing = sum(len(groups) > 1 for groups in components.values())
        sequence_sets = {name: {fasta[item] for item in ids} for name, ids in split.items()}
        exact_duplicates = sum(
            len(sequence_sets[left] & sequence_sets[right])
            for left, right in (('train', 'valid'), ('train', 'test'), ('valid', 'test'))
        )
        report[str(threshold)] = {
            'n_train': len(split['train']), 'n_valid': len(split['valid']), 'n_test': len(split['test']),
            'components': len(components), 'components_crossing_splits': crossing,
            'exact_sequence_duplicates_train_test': exact_duplicates,
            'cluster_file': str(cluster_file),
        }
        if crossing or exact_duplicates:
            raise SystemExit(f'{threshold}%: leakage detected: components={crossing}, exact_sequences={exact_duplicates}')
    (DATA_DIR / 'split_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()