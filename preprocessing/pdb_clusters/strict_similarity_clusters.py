#!/usr/bin/env python3
"""Build a strict MMseqs2 similarity cluster file for split_by_pdb_clusters."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from common import DATA_DIR, read_fasta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--threshold', type=int, default=30)
    ap.add_argument('--coverage', type=float, default=0.80)
    ap.add_argument('--threads', type=int, default=4)
    args=ap.parse_args()
    fasta=DATA_DIR/'all_sequences.fasta'; root=DATA_DIR/'strict_mmseqs'; root.mkdir(parents=True,exist_ok=True)
    db=root/'seqdb'; clu=root/f'cluster_{args.threshold}'; tmp=root/'tmp'; tsv=root/f'cluster_{args.threshold}.tsv'; out=root/f'clusters-by-entity-{args.threshold}.txt'
    subprocess.run(['mmseqs','createdb',str(fasta),str(db)],check=True)
    subprocess.run(['mmseqs','cluster',str(db),str(clu),str(tmp),'--min-seq-id',str(args.threshold/100),'-c',str(args.coverage),'--cov-mode','0','--cluster-mode','1','--threads',str(args.threads)],check=True)
    subprocess.run(['mmseqs','createtsv',str(db),str(db),str(clu),str(tsv)],check=True)
    groups={}
    for line in tsv.read_text().splitlines():
        rep,member=line.split('	')[:2]; groups.setdefault(rep,[]).append(member)
    entity_map=json.loads((DATA_DIR/'entity_map.json').read_text())
    lines=[]; skipped=0
    for members in groups.values():
        entities=[]
        for chain in members:
            pdb,auth=chain.split('_',1); ent=entity_map.get(pdb,{}).get(auth)
            if ent is not None: entities.append(f'{pdb}_{ent}')
            else: skipped+=1
        if entities: lines.append(' '.join(sorted(set(entities))))
    out.write_text(''.join(line+'\n' for line in lines))
    (root/f'manifest_{args.threshold}.json').write_text(json.dumps({'threshold':args.threshold,'coverage':args.coverage,'groups':len(groups),'clusters_written':len(lines),'chains_without_entity':skipped},indent=2)+'\n')
    print(f'Wrote strict cluster file: {out} ({len(lines)} groups)')
    print(f'DGG_CLUSTER_FILE_{args.threshold}={out}')
if __name__=='__main__': main()
