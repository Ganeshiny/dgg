#!/usr/bin/env bash
#SBATCH --job-name=dgg_sprof_setup
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sprof_go_setup_%j.out
#SBATCH --error=logs/sprof_go_setup_%j.err

set -euo pipefail

DGG_ROOT=${DGG_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}
EXTERNAL_ROOT=${SPROF_EXTERNAL_ROOT:-/home/ganeshiny.sridharan/dgg/external}
SPROF_ROOT=${SPROF_ROOT:-$EXTERNAL_ROOT/SPROF-GO}
PROTT5_ROOT=${PROTT5_ROOT:-$EXTERNAL_ROOT/prot_t5_xl_uniref50}
CONDA_ROOT=${DGG_CONDA_ROOT:-/home/ganeshiny.sridharan/miniconda3}
SPROF_CONDA_ENV=${SPROF_CONDA_ENV:-dgg_sprof_go}
SPROF_TORCH_INDEX_URL=${SPROF_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}

ARCHIVE=$EXTERNAL_ROOT/prot_t5_xl_uniref50.zip

mkdir -p "$EXTERNAL_ROOT"
if [[ ! -d "$SPROF_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/biomed-AI/SPROF-GO.git "$SPROF_ROOT"
fi
chmod +x "$SPROF_ROOT/script/diamond"
if [[ ! -d "$PROTT5_ROOT" ]]; then
  if [[ ! -s "$ARCHIVE" ]]; then
    wget -c -O "$ARCHIVE" https://zenodo.org/records/4644188/files/prot_t5_xl_uniref50.zip?download=1
  fi
  unzip -q "$ARCHIVE" -d "$EXTERNAL_ROOT"
fi
test -f "$SPROF_ROOT/script/predict.py"
test -d "$PROTT5_ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if ! conda run -n "$SPROF_CONDA_ENV" python -V >/dev/null 2>&1; then
  conda create -y -n "$SPROF_CONDA_ENV" python=3.10 pip
fi
conda run -n "$SPROF_CONDA_ENV" python -m pip install --upgrade \
  --index-url "$SPROF_TORCH_INDEX_URL" "torch==2.6.0"
conda run -n "$SPROF_CONDA_ENV" python -m pip install --upgrade \
  "transformers==4.56.2" "sentencepiece==0.2.0" \
  "numpy==1.23.5" "scipy==1.10.1" "scikit-learn==1.2.2" "tqdm==4.66.5"
conda run -n "$SPROF_CONDA_ENV" python \
  "$DGG_ROOT/scripts/check_sprof_go_environment.py" \
  --prott5-root "$PROTT5_ROOT" --require-cuda --load-prott5
echo "SPROF_ROOT=$SPROF_ROOT"
echo "PROTT5_ROOT=$PROTT5_ROOT"
echo "SPROF_CONDA_ENV=$SPROF_CONDA_ENV"
echo "Setup complete: secure PyTorch and the local ProtT5 checkpoint passed validation."
