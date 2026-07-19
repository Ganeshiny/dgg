# DeepGreenGO

A multilabel Gene Ontology (GO) term prediction model for Viridiplantae (green plant) proteins. Combines ProtBERT sequence embeddings with structure-derived contact-map graphs (GCN / GAT / Hybrid / Hybrid-JK / MLP architectures), evaluated CAFA-style (Fmax, Smin, AUPRC, AUROC) across the three GO sub-ontologies.

<img width="2100" height="1500" alt="Methodology" src="https://github.com/user-attachments/assets/47f64b6c-4ca7-4d0c-9ff2-e730ab4f4128" />

---

## Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate deepgreengo
```

> **PyTorch Geometric extras**: after activating the env, install the C++ extension wheels matching your PyTorch + CUDA version from https://data.pyg.org/whl/:
> ```bash
> pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
>     -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
> ```

### Option B — pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -r requirements.txt
```

### External tools

```bash
conda install -c conda-forge -c bioconda mmseqs2  # homology clustering
conda install -c bioconda blast diamond            # baseline predictors (optional)
```

### HuggingFace token

Required to load ProtBERT without rate limits. On an HPC cluster, put it in your profile rather than a job script:

```bash
echo 'export HF_TOKEN="your_hf_token_here"' >> ~/.bashrc
source ~/.bashrc
```

---

## Repository layout

```
deep-green-GO/
├── src/              model, training, evaluation, and figure-generation code
├── scripts/          pipeline orchestration & analysis CLIs (trial selection, bin evaluation, aggregation)
├── preprocessing/     sequence/structure extraction, homology-aware splitting, graph dataset construction
├── "arc slurms/"     SLURM job scripts for the ARC HPC pipeline
├── baselines/        BLAST / DIAMOND / naive-frequency / DeepFRI baselines
├── checks/, cmap_checks/   QA scripts for contact maps and splits
├── .agents/AGENTS.md  ARC cluster usage rules and agent conventions
├── logs/             SLURM stdout/stderr (tracked)
├── plots/            generated figures (not committed)
└── arc_tuning*/       tuning and ablation run outputs (not committed)
```

`sota/`, `bash/`, and the legacy scripts under `plots/` (`plot_model_ablations.py`, `plot_results.py`, etc.) belong to an earlier, superseded pipeline version, preserved on the `archive/pre-arc-mmseqs2-pipeline` branch. The active pipeline is the ARC-based one below.

---

## Running the pipeline

The active pipeline runs on the University of Calgary ARC HPC cluster, using a homology-controlled split (nominal 30% identity / 80% coverage — **not** leakage-free; see `preprocessing/ARC_HYBRID_TUNING.md` for the residual-similarity audit). Full stage-by-stage detail lives there rather than here.

```bash
sbatch "arc slurms/run_hybrid_tuning.slurm"    # 40-trial hyperparameter search + 5-seed confirmation
sbatch "arc slurms/run_arc_ablations.slurm"    # architecture x input-modality ablation (225 runs)
sbatch "arc slurms/run_arc_bin_eval.slurm"     # homology / information-content bin evaluation
```

See `.agents/AGENTS.md` for cluster rules (paths, GPU requirements, never use the test set during tuning).

---

## Train a single model

```bash
python src/train.py \
    --model Hybrid --loss Focal --seed 42 \
    --ontology biological_process --epochs 200
```

## Run inference

```bash
python src/predictions.py \
    -struc_dir examples/structure_files \
    -model_path runs/bp_Hybrid_Focal_s42/best_model.pth \
    -output examples/my_predictions.csv
```

## Generate figures

```bash
python src/plot_arc_ablations.py
python src/plot_arc_bins.py --bin-csv arc_tuning_cafa/ablations/nominal_30_identity_80_coverage/bin_evaluation/bin_metrics.csv
python src/plot_arc_tuning.py --tuning-root arc_tuning_cafa
```
