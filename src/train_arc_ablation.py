#!/usr/bin/env python3
"""Train one ARC model/input ablation on train/valid and report held-out test metrics."""
from __future__ import annotations
import argparse, json, pickle, random
from pathlib import Path
import numpy as np
import torch
from src.arc_dataset import ArcGraphDataset, make_dataloader
from src.model import MLPModel, GCNModel, GATModel, HybridGNN, HybridGNN_JK
from src.evals import compute_ic, evaluate_all
from src.tune_hybrid import weighted_loss

ONTOLOGIES=("molecular_function","biological_process","cellular_component")
MODELS=("MLP","GCN","GAT","Hybrid","Hybrid_JK")
MODALITIES=("full","seq_only","struct_only")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True); p.add_argument("--graph-root",type=Path,required=True)
    p.add_argument("--ontology",choices=ONTOLOGIES,required=True); p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--input-modality",choices=MODALITIES,default="full"); p.add_argument("--split-label",default="nominal_30_identity_80_coverage")
    p.add_argument("--config-json",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=1103); p.add_argument("--epochs",type=int,default=100); p.add_argument("--workers",type=int,default=0)
    return p.parse_args()

def load_cfg(path,ontology):
    obj=json.loads(path.read_text()); return obj.get(ontology,obj)

def load_ds(root,graph,ontology,split):
    with (root/f"{ontology}_{split}.pkl").open("rb") as h: ds=pickle.load(h)
    if not isinstance(ds,ArcGraphDataset) or ds.split!=split: raise RuntimeError(f"Unexpected dataset for {split}: {root}")
    ds.graph_dir=str(graph.resolve()); return ds

def transform(batch,mode):
    if mode=="struct_only": batch.x=torch.zeros_like(batch.x)
    elif mode=="seq_only":
        n=batch.x.size(0); batch.edge_index=torch.arange(n,device=batch.x.device).repeat(2,1)
    return batch

def build(name,inp,hidden,out,drop):
    hs=[hidden,hidden]
    if name=="MLP": return MLPModel(inp,hs,out)
    if name=="GCN": return GCNModel(inp,hs,out)
    if name=="GAT": return GATModel(inp,hs,out,num_attention_heads=4)
    if name=="Hybrid_JK": return HybridGNN_JK(inp,hs,out,num_attention_heads=4,dropout=drop)
    return HybridGNN(inp,hs,out,num_attention_heads=4,dropout=drop)

def evaluate(model,loader,device,ic,mode):
    ys=[]; ps=[]; model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch=transform(batch.to(device),mode); ys.append(batch.y.cpu().numpy()); ps.append(torch.sigmoid(model(batch.x,batch.edge_index,batch.batch)).cpu().numpy())
    y=np.vstack(ys); p=np.vstack(ps); return evaluate_all(y,p,ic)

def main():
    a=parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); cfg=load_cfg(a.config_json,a.ontology)
    train=load_ds(a.dataset_dir,a.graph_root,a.ontology,"train"); valid=load_ds(a.dataset_dir,a.graph_root,a.ontology,"valid"); test=load_ds(a.dataset_dir,a.graph_root,a.ontology,"test")
    bs=int(cfg.get("batch_size",32)); hidden=int(cfg.get("hidden_dim",512)); drop=float(cfg.get("dropout",0.2)); patience=int(cfg.get("patience",8)); gamma=cfg.get("focal_gamma")
    tl=make_dataloader(train,bs,True,a.workers); vl=make_dataloader(valid,bs,False,a.workers); el=make_dataloader(test,bs,False,a.workers)
    pos=torch.tensor((len(train)-train.labels.sum(0)+1)/(train.labels.sum(0)+1),dtype=torch.float32,device=device); ic=compute_ic(train.labels)
    model=build(a.model,int(train[0].x.shape[1]),hidden,train.num_classes,drop).to(device); opt=torch.optim.AdamW(model.parameters(),lr=float(cfg.get("learning_rate",1e-4)),weight_decay=float(cfg.get("weight_decay",1e-6)))
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True); run_cfg={**cfg,"ontology":a.ontology,"model":a.model,"input_modality":a.input_modality,"split_label":a.split_label,"seed":a.seed}
    (out/"config.json").write_text(json.dumps(run_cfg,indent=2)+"\n"); best=-1.; stale=0; best_metrics=None
    for epoch in range(1,a.epochs+1):
        model.train(); losses=[]
        for batch in tl:
            batch=transform(batch.to(device),a.input_modality); opt.zero_grad(set_to_none=True); z=model(batch.x,batch.edge_index,batch.batch); loss=weighted_loss(z,batch.y.float(),pos,str(cfg.get("loss","BCE")),gamma); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get("gradient_clip",1.0))); opt.step(); losses.append(float(loss.detach().cpu()))
        met=evaluate(model,vl,device,ic,a.input_modality); score=float(met["Macro_Fmax"]); print(f"epoch={epoch} loss={np.mean(losses):.6f} validation_macro_fmax={score:.6f} validation_micro_fmax={float(met['Micro_Fmax']):.6f}",flush=True)
        if score>best+1e-8:
            best=score; stale=0; best_metrics=met; torch.save({"model_state_dict":model.state_dict(),"config":run_cfg,"metrics":met},out/"best_checkpoint.pt"); (out/"validation_metrics.json").write_text(json.dumps(met,indent=2)+"\n")
        else: stale+=1
        if stale>=patience: break
    if best_metrics is None: raise RuntimeError("No checkpoint produced")
    ck=torch.load(out/"best_checkpoint.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model_state_dict"]); test_metrics=evaluate(model,el,device,ic,a.input_modality); (out/"test_metrics.json").write_text(json.dumps(test_metrics,indent=2)+"\n"); print(json.dumps({"ontology":a.ontology,"model":a.model,"input_modality":a.input_modality,"split_label":a.split_label,"test":test_metrics},sort_keys=True),flush=True)

if __name__=="__main__": main()
