#!/usr/bin/env python3
"""Add information-content Smin to a SPROF-GO metrics report."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from evaluate_sprof_go import unpack_dataset
from pickle_compat import load_pickle_compat

def main():
    p=argparse.ArgumentParser();p.add_argument("--evaluation-dir",type=Path,required=True);p.add_argument("--tuning-root",type=Path,required=True);a=p.parse_args()
    report_path=a.evaluation_dir/"metrics.json";report=json.loads(report_path.read_text())
    for ont in report["ontologies"]:
        array_path=a.evaluation_dir/"prediction_arrays"/f"{ont}.npz"
        with np.load(array_path,allow_pickle=False) as payload:
            stored_terms=payload["terms"].tolist();y=payload["truth"].astype(bool);score=payload["scores"]
        with (a.tuning_root/"datasets"/f"{ont}_train.pkl").open("rb") as f: _,terms,ytrain=unpack_dataset(load_pickle_compat(f))
        if stored_terms != terms: raise ValueError(f"Term order mismatch for {ont}")
        ic=-np.log2((ytrain.sum(0)+1)/(len(ytrain)+1));best=float("inf");best_t=0.0
        for t in np.linspace(0,1,101):
            pred=score>t if t==0 else score>=t;ru=np.sum((y & ~pred)*ic,axis=1).mean();mi=np.sum((~y & pred)*ic,axis=1).mean();s=float(np.hypot(ru,mi))
            if s<best:best,best_t=s,float(t)
        report["ontologies"][ont]["smin"]=best;report["ontologies"][ont]["smin_threshold"]=best_t
    report_path.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
