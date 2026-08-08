# DeepGreenGO

A multilabel Gene Ontology (GO) term prediction model for Viridiplantae (green plant) proteins. Combines ProtBERT sequence embeddings with structure-derived contact-map graphs (GCN / GAT / Hybrid / Hybrid-JK / MLP architectures), evaluated CAFA-style (Fmax, Smin, AUPRC, AUROC) across the three GO sub-ontologies.

<img width="2100" height="1500" alt="Methodology" src="https://github.com/user-attachments/assets/47f64b6c-4ca7-4d0c-9ff2-e730ab4f4128" />

---

## Setup

### Conda 

```bash
conda env create -f environment.yml
conda activate deepgreengo
```

> **PyTorch Geometric extras**: after activating the env, install the C++ extension wheels matching your PyTorch + CUDA version from https://data.pyg.org/whl/:
> ```bash
> pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
>     -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
> ```

### pip

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

