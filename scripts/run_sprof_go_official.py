#!/usr/bin/env python3
"""Run the unmodified official SPROF-GO code without editing its hard-coded ProtT5 path."""
from __future__ import annotations
import argparse, importlib, sys
from pathlib import Path

from check_sprof_go_environment import require_safe_torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sprof-root", type=Path, required=True)
    p.add_argument("--prott5-root", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--outpath", type=Path, required=True)
    p.add_argument("--feat-bs", type=int, default=2)
    p.add_argument("--pred-bs", type=int, default=4)
    p.add_argument("--save-feat", action="store_true")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    import torch

    try:
        require_safe_torch(torch.__version__)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    script_dir = (a.sprof_root / "script").resolve()
    required = [script_dir / "predict.py", script_dir / "diamond", a.prott5_root]
    missing = [str(x) for x in required if not x.exists()]
    if missing: raise SystemExit("Missing SPROF-GO components: " + ", ".join(missing))
    a.outpath.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(script_dir))
    predict = importlib.import_module("predict")
    predict.ProtTrans_path = str(a.prott5_root.resolve())
    seq_info = predict.process_fasta(str(a.fasta.resolve()), str(a.outpath.resolve()) + "/")
    if not isinstance(seq_info, list): raise SystemExit(f"SPROF-GO rejected FASTA (code {seq_info})")
    run_id = a.fasta.stem.replace(" ", "_")
    predict.main(run_id, seq_info, str(a.outpath.resolve()) + "/", 20, a.feat_bs, a.pred_bs, a.save_feat, not a.cpu)

if __name__ == "__main__": main()
