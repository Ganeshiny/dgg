#!/usr/bin/env python3
"""Create publication plots from ARC ablation result JSON files or logs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ONTOLOGIES = ["molecular_function", "biological_process", "cellular_component"]
LABELS = {"molecular_function": "MF", "biological_process": "BP", "cellular_component": "CC"}
MODELS = ["MLP", "GCN", "GAT", "Hybrid", "Hybrid_JK"]
MODES = ["full", "seq_only", "struct_only"]

def read_results(root: Path, logs: Path):
    rows = []
    for p in root.rglob("test_metrics.json"):
        rel = p.relative_to(root).parts
        if len(rel) >= 5:
            rows.append({"ontology": rel[0], "model": rel[1], "input": rel[2], "seed": rel[3], **json.loads(p.read_text())})
    # Each completed ARC array task writes one final JSON object to its .out log.
    # Use it when the result folders themselves were not downloaded.
    keys = {(r["ontology"], r["model"], r["input"], str(r["seed"])) for r in rows}
    for p in sorted(logs.glob("arc_ablation_*.out")):
        for line in reversed(p.read_text(errors="ignore").splitlines()):
            try: item = json.loads(line)
            except json.JSONDecodeError: continue
            if not {"input_modality", "model", "ontology", "test"}.issubset(item): continue
            seed = p.stem.rsplit("_", 1)[-1]
            key = (item["ontology"], item["model"], item["input_modality"], seed)
            if key not in keys:
                rows.append({"ontology": item["ontology"], "model": item["model"], "input": item["input_modality"], "seed": seed, **item["test"]})
                keys.add(key)
            break
    return pd.DataFrame(rows)

def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=240, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)

def make_plots(df, out):
    out.mkdir(parents=True, exist_ok=True); df.to_csv(out / "ablation_test_metrics.csv", index=False)
    full = df[df["input"] == "full"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ont in zip(axes, ONTOLOGIES):
        sub = full[full["ontology"] == ont]; x = np.arange(len(MODELS)); width = .16
        for j, metric in enumerate(["Micro_Fmax", "Macro_Fmax", "Micro_AUPRC", "Macro_AUPRC", "Smin"]):
            vals = [sub.loc[sub["model"] == m, metric].mean() if not sub.loc[sub["model"] == m, metric].empty else np.nan for m in MODELS]
            ax.bar(x + (j - 2) * width, vals, width, label=metric.replace("_", " "))
        ax.set_title(f"{LABELS[ont]} — full input"); ax.set_xticks(x, MODELS, rotation=35, ha="right"); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Mean test metric across seeds"); axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left"); save(fig, out / "architecture_ablation_full_input.png")

    proposed = df[df["model"].isin(["Hybrid", "Hybrid_JK"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True); x = np.arange(len(MODES))
    for ax, ont in zip(axes, ONTOLOGIES):
        sub = proposed[proposed["ontology"] == ont]
        for model in ["Hybrid", "Hybrid_JK"]:
            vals = [sub[(sub["model"] == model) & (sub["input"] == mode)]["Micro_Fmax"].mean() for mode in MODES]
            ax.plot(x, vals, marker="o", linewidth=2, label=model)
        ax.set_title(LABELS[ont]); ax.set_xticks(x, ["Full", "Sequence", "Structure"]); ax.grid(alpha=.2)
    axes[0].set_ylabel("Micro-Fmax"); axes[-1].legend(); save(fig, out / "hybrid_input_ablation_micro_fmax.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for ax, ont in zip(axes, ONTOLOGIES):
        sub = df[df["ontology"] == ont]; table = sub.pivot_table(index="model", columns="input", values="Micro_Fmax", aggfunc="mean").reindex(index=MODELS, columns=MODES)
        image = ax.imshow(table.values, cmap="viridis", aspect="auto", vmin=0, vmax=1); ax.set_title(LABELS[ont]); ax.set_xticks(range(3), ["Full", "Seq", "Struct"]); ax.set_yticks(range(5), MODELS)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                if pd.notna(table.iloc[i, j]): ax.text(j, i, f"{table.iloc[i,j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    save(fig, out / "micro_fmax_model_input_heatmap.png")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ablations-root", type=Path, default=Path("arc_tuning_cafa/ablations/nominal_30_identity_80_coverage")); ap.add_argument("--logs-dir", type=Path, default=Path("logs")); ap.add_argument("--output-dir", type=Path, default=Path("plots/arc_tuning_cafa/ablations")); args = ap.parse_args()
    df = read_results(args.ablations_root.resolve(), args.logs_dir.resolve())
    if df.empty: raise SystemExit("No ablation result JSON or logs found")
    make_plots(df, args.output_dir.resolve()); print(f"Loaded {len(df)} completed ablation runs"); print(f"Wrote plots to {args.output_dir.resolve()}")

if __name__ == "__main__": main()
