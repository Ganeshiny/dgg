# Rice AlphaFold inference bundle

This folder runs the five-seed DeepGreenGO ensemble on AlphaFold structures
without requiring the training repository or training datasets. It contains the
15 selected checkpoints, exact ontology vocabularies, and inference-only code.

## 1. Create a clean environment

Use Python 3.10 or newer. Create a new environment specifically for this
standalone bundle. Do not install these requirements into an existing training
environment because changing PyTorch can leave old torchvision and PyG binary
extensions behind.

```bash
cd rice_prediction_standalone
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_bundle.py
```

The reproducible requirements pin PyTorch 2.6.x. For GPU inference, install the
matching CUDA-enabled PyTorch 2.6 build for the machine before installing
`requirements.txt`. The first prediction downloads `Rostlab/prot_bert_bfd`
from Hugging Face (roughly 1.7 GB), unless it is already cached.

If verification reports `torchvision::nms does not exist`, the active
environment mixes incompatible PyTorch and torchvision versions. Activate the
clean `.venv`. In a disposable inference environment, another option is to
remove unused stale extensions before reinstalling the requirements:

```bash
python -m pip uninstall -y torchvision torch-scatter torch-sparse torch-cluster torch-spline-conv
python -m pip install --force-reinstall -r requirements.txt
```

Do not use that cleanup command on a training environment that needs those
packages.

## 2. Download AlphaFold structures

Put structures downloaded from the AlphaFold Protein Structure Database in
`input/structures/`. Standard website filenames work without renaming:

```text
input/structures/AF-Q6Z4U4-F1-model_v4.cif
input/structures/AF-A0A0P0VUH7-F1-model_v6.cif.gz
```

Supported formats are `.cif`, `.mmcif`, `.pdb`, and their gzip-compressed
forms. For a standard name such as `AF-Q6Z4U4-F1-model_v4.cif`, the default
output protein ID is the UniProt accession `Q6Z4U4`.

The amino-acid sequence is read directly from the structure, so a FASTA file is
not required.

### Optional FASTA ID mapping

Pass `--fasta input/rice.fasta` only when output should use identifiers from a
separate FASTA. The program resolves records in this order:

1. AFDB/UniProt filename aliases, including `sp|ACCESSION|NAME` and
   `tr|ACCESSION|NAME` headers.
2. A unique exact sequence match.
3. The normalized UniProt/structure ID when no FASTA match exists.

The structure-derived sequence is always used for prediction, preventing a
filename mismatch from pairing coordinates with the wrong sequence. Output
contains both `protein_id` and the original `structure_id`.

The first structure chain is used. Edges involving residues below the pLDDT
threshold (70 by default) are removed, matching training graph construction.

## 3. Run predictions

From the standalone folder:

```bash
python scripts/predict_alphafold.py \
  --input-dir input/structures \
  --ontology molecular_function \
  --output results/molecular_function.tsv

python scripts/predict_alphafold.py \
  --input-dir input/structures \
  --ontology biological_process \
  --output results/biological_process.tsv

python scripts/predict_alphafold.py \
  --input-dir input/structures \
  --ontology cellular_component \
  --output results/cellular_component.tsv
```

Add `--fasta input/rice.fasta` to any command when FASTA-based output ID
mapping is wanted.

Use `--plddt-threshold 0` to keep all contact edges or
`--call-threshold VALUE` to change the default 0.5 binary-call threshold.
Proteins longer than 1,022 residues are truncated to their first 1,022
residues, matching final training preprocessing.

Each result is a long-format TSV with `protein_id`, `structure_id`, `go_id`,
`score`, `called`, and `ontology`. Use the continuous score for ranking and
enrichment analyses.
