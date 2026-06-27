# DeepGreenGO

A multilabel protein function prediction model for Viridiplantae (green plants) using Graph Neural Networks with ProtBERT embeddings.

<img width="2100" height="1500" alt="Methodology" src="https://github.com/user-attachments/assets/47f64b6c-4ca7-4d0c-9ff2-e730ab4f4128" />

---

## Environment Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate deepgreengo
```

> **Note on PyTorch Geometric extras**: After activating the env, install the C++ extension wheels matching your exact PyTorch + CUDA version from https://data.pyg.org/whl/:
> ```bash
> # Example for torch 2.1.0 + CUDA 12.1:
> pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
>     -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
> ```

### Option B — pip

```bash
# 1. Install PyTorch first (choose CUDA version at https://pytorch.org):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install PyTorch Geometric:
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 3. Install remaining dependencies:
pip install -r requirements.txt
```

### External tools (via conda)

```bash
conda install -c conda-forge -c bioconda mmseqs2  # Homology clustering
conda install -c bioconda blast                   # BLAST baseline (optional)
conda install -c bioconda diamond                 # DIAMOND baseline (optional)
```

---

## Data Preparation

Place your downloaded Viridiplantae PDB structures (`.cif.gz`) in:
```
preprocessing/data/structure_files/
```

You also need the SIFTS annotation file and GO OBO file in `preprocessing/data/`.

---

## Run the Full Pipeline

Before running, set your Hugging Face token as an environment variable to prevent rate limits and unauthenticated download errors for ProtBERT. 

If you are running on an HPC cluster (like SLURM), **do not hardcode your token into job scripts**. Instead, add it to your user profile:
```bash
echo 'export HF_TOKEN="your_hf_token_here"' >> ~/.bashrc
source ~/.bashrc
```

Then you can run the pipeline:

```bash
bash run_all.sh
```

The script will:
1. Extract sequences and build GO annotations from CIF files
2. Cluster sequences at 30% identity (MMseqs2) and split into Train/Valid/Test
3. Compute pLDDT-filtered contact maps and build PyG graph datasets
4. Run BLAST / DIAMOND / Naive baselines
5. Train all model ablations (MLP / GCN / GAT / Hybrid × BCE / Focal, 3 seeds, 3 ontologies)
6. Run per-cluster generalisation evaluation
7. Aggregate results and generate figures

### Skip flags

```bash
bash run_all.sh --skip-preprocess   # Preprocessing already done
bash run_all.sh --skip-ablations    # Only run preprocessing + baselines
bash run_all.sh --skip-plots        # Skip figure generation
```

### Environment overrides

```bash
EPOCHS=50 BATCH_SIZE=16 MAIN_MODEL=GAT MAIN_LOSS=BCE bash run_all.sh
```

### Hyperparameter Tuning
To run an automated grid search for learning rate, dropout, and batch size across all 3 ontologies:
```bash
bash run_tuning_64.sh
python3 aggregate_tuning.py
```
*Note: On HPC clusters (e.g., UCalgary ARC), use the provided SLURM scripts and submit as an array job to train ontologies in parallel:*
```bash
sbatch --array=0-2 arc_submit_64.slurm
```

---

## Train a Single Model

```bash
python3 train.py \
    --model Hybrid \
    --loss  Focal  \
    --seed  42     \
    --ontology biological_process \
    --epochs 200
```

---

## Run Inference

```bash
python3 predictions.py \
    -struc_dir  examples/structure_files \
    -model_path runs/bp_Hybrid_Focal_s42/best_model.pth \
    -output     examples/my_predictions.csv
```

---

## Project Structure

```
deep-green-GO/
├── preprocessing/
│   ├── extract_seqs_from_cif.py  # Sequence extraction + GO annotation
│   ├── cluster_and_split.py      # MMseqs2 clustering + cluster-aware split
│   ├── create_cmaps.py           # pLDDT-filtered contact maps
│   └── create_batch_dataset.py   # PyG graph dataset builder (ProtBERT)
├── baselines/
│   ├── blast/                    # BLASTp nearest-neighbour baseline
│   ├── diamond/                  # DIAMOND nearest-neighbour baseline
│   ├── naive_frequency/          # GO term frequency prior baseline
│   └── deepfri_comparison/       # Comparison notes vs DeepFRI
├── model.py                      # GCN / GAT / Hybrid / MLP architectures
├── train.py                      # Training script with early stopping
├── evals.py                      # Micro/Macro Fmax, Smin, AUROC, AUPRC
├── focal_loss.py                 # Focal loss implementation
├── per_cluster_eval.py           # Per homology-cluster generalisation eval
├── aggregate_results.py          # Aggregate runs into mean±std tables
├── aggregate_tuning.py           # Summarize grid search tuning results
├── plot_results.py               # Publication-quality figure generation
├── predictions.py                # Inference on new structures
├── run_all.sh                    # ONE-CLICK full pipeline
├── run_ablations.sh              # Ablation sweep helper
├── run_tuning_64.sh              # Batch 64 hyperparameter sensitivity helper
├── arc_submit_64.slurm           # ARC HPC SLURM submission script
├── generate_supp_tables.py       # LaTeX config table generator
├── environment.yml               # Conda environment
└── requirements.txt              # pip requirements
```
