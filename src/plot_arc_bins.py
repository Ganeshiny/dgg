#!/usr/bin/env python3
"""Plot test performance across sequence-homology and IC bins."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ONTOLOGIES = ["molecular_function", "biological_process", "cellular_component"]
LABELS = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}

def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=240, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bin-csv", type=Path, required=True); ap.add_argument("--output-dir", type=Path, default=Path("plots/arc_tuning_cafa/bin_evaluation")); args = ap.parse_args()
    df = pd.read_csv(args.bin_csv); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True); df.to_csv(out / "bin_metrics.csv", index=False)
    # Average over seeds, retaining model and input modality as separate lines.
    for bin_type in sorted(df["bin_type"].dropna().unique()):
        sub = df[df["bin_type"] == bin_type]
        for metric in ["Micro_Fmax", "Macro_Fmax", "Micro_AUPRC", "Macro_AUPRC", "Micro_AUROC", "Macro_AUROC", "Smin"]:
            if metric not in sub: continue
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
            for ax, ont in zip(axes, ONTOLOGIES):
                s = sub[sub["ontology"] == ont]
                for (model, modality), g in s.groupby(["model", "input_modality"]):
                    means = g.groupby("bin")[metric].mean()
                    ax.plot(means.index, means.values, marker="o", linewidth=1.8, label=f"{model}/{modality}")
                ax.set_title(LABELS[ont]); ax.set_xlabel(f"{bin_type} bin"); ax.tick_params(axis="x", rotation=35); ax.grid(alpha=.2)
            axes[0].set_ylabel(metric.replace("_", " ")); axes[-1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
            save(fig, out / f"{bin_type}_{metric.lower()}.png")
    print(f"Wrote bin plots to {out}")

if __name__ == "__main__": main()
