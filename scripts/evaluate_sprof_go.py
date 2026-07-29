#!/usr/bin/env python3
"""Align official SPROF-GO scores to ARC vocabularies and compute held-out metrics."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pickle_compat import load_pickle_compat

MAP = {"MF":"molecular_function", "BP":"biological_process", "CC":"cellular_component"}

def parse_sprof(path):
    lines = Path(path).read_text().splitlines()
    pos, terms = 1, {}
    for task in MAP:
        while pos < len(lines) and lines[pos].strip() != f"{task}:": pos += 1
        terms[task] = [x.strip() for x in lines[pos+1].split(";")]
        # Some official SPROF-GO releases omit the blank separator after the
        # final CC vocabulary block. Locate the first protein block instead of
        # assuming a fixed four-line offset after CC.
        if task != "CC":
            pos += 4
    while pos < len(lines):
        if (
            lines[pos].strip()
            and pos + 1 < len(lines)
            and lines[pos + 1].strip() == "MF:"
        ):
            break
        pos += 1
    scores = {}
    while pos < len(lines):
        if not lines[pos].strip(): pos += 1; continue
        pid = lines[pos].strip(); pos += 1; scores[pid] = {}
        for task in MAP:
            if lines[pos].strip() != f"{task}:": raise ValueError(f"Malformed output near {pid}/{task}")
            vals = np.asarray([float(x) for x in lines[pos+1].split(";")], dtype=np.float32)
            if len(vals) != len(terms[task]): raise ValueError(f"Length mismatch for {pid}/{task}")
            scores[pid][task] = vals; pos += 2
    return terms, scores

def unpack_dataset(obj):
    terms = list(getattr(obj, "terms", obj.get("terms") if isinstance(obj, dict) else []))
    attrs = getattr(obj, "__dict__", {})
    ids = next((attrs.get(k) for k in ("protein_ids","ids","proteins") if attrs.get(k) is not None), None)
    labels = next((attrs.get(k) for k in ("labels","y","targets") if attrs.get(k) is not None), None)
    if isinstance(obj, dict):
        ids = ids if ids is not None else next((obj.get(k) for k in ("protein_ids","ids","proteins") if obj.get(k) is not None), None)
        labels = labels if labels is not None else next((obj.get(k) for k in ("labels","y","targets") if obj.get(k) is not None), None)
    if hasattr(obj, "columns"):
        idcol = next((k for k in ("protein_id","protein","id") if k in obj.columns), None)
        labcol = next((k for k in ("labels","label","annotations","gos") if k in obj.columns), None)
        ids, labels = obj[idcol].tolist(), obj[labcol].tolist()
    if ids is None or labels is None or not terms: raise TypeError(f"Unsupported test dataset {type(obj).__name__}: {attrs.keys()}")
    y = np.zeros((len(ids), len(terms)), dtype=np.uint8)
    term_idx = {t:i for i,t in enumerate(terms)}
    for i, lab in enumerate(labels):
        arr = np.asarray(lab)
        if arr.ndim == 1 and len(arr) == len(terms) and np.issubdtype(arr.dtype, np.number): y[i] = arr > 0
        else:
            for item in lab:
                j = int(item) if isinstance(item, (int,np.integer)) else term_idx.get(str(item), -1)
                if 0 <= j < len(terms): y[i,j] = 1
    return [str(x) for x in ids], terms, y

def metrics(y, p):
    thresholds = np.linspace(0, 1, 101); best_micro = best_macro = 0.0
    for t in thresholds:
        q = p > t if t == 0 else p >= t; tp = (q & (y==1)).sum(); fp = (q & (y==0)).sum(); fn = ((~q) & (y==1)).sum()
        pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
        best_micro = max(best_micro, 2*pr*rc/(pr+rc) if pr+rc else 0)
        spp = (q & (y==1)).sum(1); pp=q.sum(1); ap=(y==1).sum(1)
        mp=np.mean(np.divide(spp,pp,out=np.zeros_like(spp,dtype=float),where=pp>0)); mr=np.mean(np.divide(spp,ap,out=np.zeros_like(spp,dtype=float),where=ap>0))
        best_macro=max(best_macro,2*mp*mr/(mp+mr) if mp+mr else 0)
    valid = (y.sum(0)>0) & (y.sum(0)<len(y))
    return {"micro_fmax":best_micro,"macro_fmax":best_macro,
      "micro_aupr":float(average_precision_score(y.ravel(),p.ravel())),
      "macro_aupr":float(average_precision_score(y[:,valid],p[:,valid],average="macro")) if valid.any() else None,
      "micro_auroc":float(roc_auc_score(y.ravel(),p.ravel())),
      "macro_auroc":float(roc_auc_score(y[:,valid],p[:,valid],average="macro")) if valid.any() else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",type=Path,required=True); ap.add_argument("--tuning-root",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); a=ap.parse_args()
    src_terms, src_scores=parse_sprof(a.predictions); a.output_dir.mkdir(parents=True,exist_ok=True); report={"model":"SPROF-GO","alignment_policy":"target-vocabulary; missing SPROF terms assigned score 0","ontologies":{}}
    with (a.output_dir/"predictions_long.csv").open("w",newline="") as fh:
      wr=csv.writer(fh); wr.writerow(["ontology","protein_id","go_id","score","truth"])
      for task,ont in MAP.items():
        with (a.tuning_root/"datasets"/f"{ont}_test.pkl").open("rb") as f: ids,terms,y=unpack_dataset(load_pickle_compat(f))
        src_idx={t:i for i,t in enumerate(src_terms[task])}; p=np.zeros(y.shape,dtype=np.float32); missing_proteins=[]
        for i,pid in enumerate(ids):
          if pid not in src_scores: missing_proteins.append(pid); continue
          for j,term in enumerate(terms):
            k=src_idx.get(term); p[i,j]=src_scores[pid][task][k] if k is not None else 0
          for j,term in enumerate(terms): wr.writerow([ont,pid,term,float(p[i,j]),int(y[i,j])])
        m=metrics(y,p); m.update({"proteins":len(ids),"missing_proteins":len(missing_proteins),"target_terms":len(terms),"common_terms":sum(t in src_idx for t in terms),"sprof_terms":len(src_idx)})
        report["ontologies"][ont]=m
    (a.output_dir/"metrics.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
