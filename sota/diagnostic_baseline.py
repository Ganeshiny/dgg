import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import plot_sota_comparison as psc

def diagnostic_run():
    print("Capturing baseline diagnostics...")
    
    # Fake args for the module to load datasets
    class DummyArgs:
        mode = 'all'
        common_subset = False
        supplementary = False
    psc.args = DummyArgs()
    
    datasets = psc.load_datasets()
    
    diagnostic = {
        "raw_data": {
            "ontologies": list(psc.ONTOLOGIES.keys()),
        },
        "bin_populations": {},
        "y_limits_raw": {},
        "palette": psc.PALETTE.copy(),
    }
    
    # 1. IC Bin Populations
    print("Checking IC bin populations...")
    ic_counts = {}
    for ont_full, ont_short in psc.ONTOLOGIES.items():
        y_true = psc._load_y_true(ont_short, datasets)
        valid_mask = psc._load_valid_mask(ont_short)
        if valid_mask is not None and y_true is not None:
            y_true = y_true[:, valid_mask]
        
        ic_raw = psc._load_ic(ont_short, datasets)
        if ic_raw is not None and valid_mask is not None:
            ic_raw = ic_raw[valid_mask]
        
        ic = ic_raw if ic_raw is not None else psc.compute_ic(y_true)
        
        test_ds = datasets[ont_full]['test']
        prot_list = test_ds.pdb_split_list
        goterms = test_ds.y_labels
        
        preds, mask = psc.load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
        
        if mask is not None:
            if ic is not None and len(ic) == len(mask):
                ic = ic[mask]
        
        bins = [(0, 2, '0–2'), (2, 4, '2–4'), (4, 6, '4–6'), (6, 15, '>6')]
        active_bins = [b for b in bins if ((ic >= b[0]) & (ic < b[1])).sum() > 0]
        
        ic_counts[ont_short] = {}
        for b in bins:
            ic_counts[ont_short][b[2]] = int(((ic >= b[0]) & (ic < b[1])).sum()) if ic is not None else 0
            
    diagnostic["bin_populations"]["ic_values"] = ic_counts

    # 2. Depth Bin Populations
    print("Checking Depth bin populations...")
    depth_counts = {}
    for ont_full, ont_short in psc.ONTOLOGIES.items():
        y_true = psc._load_y_true(ont_short, datasets)
        valid_mask = psc._load_valid_mask(ont_short)
        if valid_mask is not None and y_true is not None:
            y_true = y_true[:, valid_mask]
        
        go_depth_map = psc._compute_go_depths()
        
        goterms = datasets[ont_full]['test'].y_labels
        if valid_mask is not None:
            goterms = [gt for gt, v in zip(goterms, valid_mask) if v]
            
        term_depths = np.array([go_depth_map.get(gt, 0) for gt in goterms])
        
        test_ds = datasets[ont_full]['test']
        prot_list = test_ds.pdb_split_list
        preds, mask = psc.load_predictions(ont_full, ont_short, prot_list, goterms, valid_mask=valid_mask)
        
        if mask is not None:
            # depth applies to GO terms (columns), mask applies to proteins (rows). 
            pass 
            
        bins = [(0, 4, '0–3'), (4, 6, '4–5'), (6, 8, '6–7'), (8, 20, '≥8')]
        depth_counts[ont_short] = {}
        for b in bins:
            depth_counts[ont_short][b[2]] = int(((term_depths >= b[0]) & (term_depths < b[1])).sum())
            
    diagnostic["bin_populations"]["depth_bins"] = depth_counts
    
    # 3. Y-limits Raw (Summary Fmax)
    print("Checking Y-limits Raw...")
    results_csv = os.path.join(psc.PROJECT_DIR, 'runs_5seeds', 'evaluation_results_all.csv')
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        metric = 'Micro_Fmax_mean'
        for ont in ['BP', 'MF', 'CC']:
            sub = df[df['Ontology'] == ont]
            sub = sub.set_index('Model').reindex(psc.MODEL_ORDER).dropna(subset=[metric])
            vals = sub[metric].values
            if len(vals) > 0:
                lo = float(max(0, vals.min() - 0.05))
                hi = float(min(1.0, vals.max() + 0.08))
                diagnostic["y_limits_raw"][f"summary_fmax_{ont.lower()}"] = (lo, hi)
                
    # 4. DeepFRI Color Collision
    seq_color = psc.PALETTE.get("DeepFRI_Seq", psc.PALETTE.get("DeepFRI", "#888888"))
    cmap_color = psc.PALETTE.get("DeepFRI_Cmap", psc.PALETTE.get("DeepFRI", "#888888"))
    diagnostic["colors"] = {
        "DeepFRI_Seq": seq_color,
        "DeepFRI_Cmap": cmap_color,
        "Collision": seq_color == cmap_color
    }

    Path("diagnostic_baseline.json").write_text(json.dumps(diagnostic, indent=2))
    print("✅ Diagnostic baseline written to diagnostic_baseline.json")

if __name__ == "__main__":
    diagnostic_run()
