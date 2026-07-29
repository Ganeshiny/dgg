#!/usr/bin/env python3
"""Publication plots comparing DeepGreenGO with SPROF-GO only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ONTS=[("molecular_function","MF"),("biological_process","BP"),("cellular_component","CC")]
METRICS=[("macro_fmax","Macro Fmax"),("micro_fmax","Micro Fmax"),("macro_aupr","Macro AUPR"),("micro_aupr","Micro AUPR"),("macro_auroc","Macro AUROC"),("micro_auroc","Micro AUROC")]
COLORS={"DeepGreenGO":"#1f77b4","SPROF-GO":"#e76f51"}

def find_onts(obj):
    if "ontologies" in obj:return obj["ontologies"]
    for key in ("results","metrics","test"):
        if isinstance(obj.get(key),dict) and any(k in obj[key] for k,_ in ONTS):return obj[key]
    return obj

def main():
    p=argparse.ArgumentParser();p.add_argument("--deepgreengo",type=Path,required=True);p.add_argument("--sprof",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
    data={"DeepGreenGO":find_onts(json.loads(a.deepgreengo.read_text())),"SPROF-GO":find_onts(json.loads(a.sprof.read_text()))};a.output_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharey=False);x=np.arange(3);width=.36
    for ax,(metric,label) in zip(axes.flat,METRICS):
        for j,model in enumerate(data):
            vals=[data[model].get(ont,{}).get(metric,np.nan) for ont,_ in ONTS]
            ax.bar(x+(j-.5)*width,vals,width,label=model,color=COLORS[model])
        ax.set_title(label);ax.set_xticks(x,[short for _,short in ONTS]);ax.set_ylim(0,1);ax.grid(axis="y",alpha=.25)
    axes[0,0].legend(frameon=False);fig.suptitle("Held-out test performance: DeepGreenGO vs SPROF-GO");fig.tight_layout()
    for ext in ("png","pdf","svg"):fig.savefig(a.output_dir/f"deepgreengo_vs_sprof_go.{ext}",dpi=400,bbox_inches="tight")
    plt.close(fig)
if __name__=="__main__":main()
