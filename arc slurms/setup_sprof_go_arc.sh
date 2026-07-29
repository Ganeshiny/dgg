#!/usr/bin/env bash
set -euo pipefail

DGG_ROOT=${DGG_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}
EXTERNAL_ROOT=${SPROF_EXTERNAL_ROOT:-/home/ganeshiny.sridharan/dgg/external}
SPROF_ROOT=${SPROF_ROOT:-$EXTERNAL_ROOT/SPROF-GO}
PROTT5_ROOT=${PROTT5_ROOT:-$EXTERNAL_ROOT/prot_t5_xl_uniref50}
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
echo "SPROF_ROOT=$SPROF_ROOT"
echo "PROTT5_ROOT=$PROTT5_ROOT"
echo "Setup complete. Keep the SPROF-GO MIT LICENSE with benchmark materials."
