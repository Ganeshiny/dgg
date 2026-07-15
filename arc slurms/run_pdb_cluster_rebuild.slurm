#!/usr/bin/env bash
#SBATCH --job-name=dgg_pdb_rebuild
#SBATCH --partition=cpu2022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/pdb_rebuild_%j.out
#SBATCH --error=logs/pdb_rebuild_%j.err

set -euo pipefail

report_exit() {
  rc=$?
  echo "Job exiting with status ${rc} at $(date --iso-8601=seconds)" >&2
}
trap report_exit EXIT

PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"
cd "${PROJECT_DIR}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
export DGG_DATA_ROOT="${DATA_ROOT}"
echo "Data root: ${DGG_DATA_ROOT}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Some ARC Conda activation hooks reference optional variables while nounset
# is enabled. Temporarily disable nounset for activation, then restore it.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DGG_CONDA_ENV:-deepgreengo}"
set -u

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "ERROR: required command '${name}' is not available in environment '${CONDA_DEFAULT_ENV:-unknown}'." >&2
    echo "Install/update the environment from environment.yml, then resubmit." >&2
    exit 127
  fi
  echo "Using ${name}: $(command -v "${name}")"
}

require_command python3
require_command blastp
require_command makeblastdb

mkdir -p "${DATA_ROOT}"

# Lightweight modes for an already-built data directory. These never download
# data, rebuild CIF-derived contact maps, or rerun dataset preparation.
if [[ "${DGG_PLOTS_ONLY:-0}" == "1" ]]; then
  for required in "${DATA_ROOT}/all_sequences.fasta" "${DATA_ROOT}/pdb2go.tsv" "${DATA_ROOT}/entity_map.json" "${DATA_ROOT}/pdb_splits" "${DATA_ROOT}/datasets"; do
    [[ -e "${required}" ]] || { echo "ERROR: plot-only input missing: ${required}" >&2; exit 2; }
  done
  python3 preprocessing/pdb_clusters/verify_splits.py
  python3 preprocessing/pdb_clusters/analyse_and_plot.py
  echo "Plot-only run completed: $(date --iso-8601=seconds)"
  exit 0
fi

if [[ "${DGG_SPLITS_ONLY:-0}" == "1" ]]; then
  for required in "${DATA_ROOT}/all_sequences.fasta" "${DATA_ROOT}/pdb2go.tsv" "${DATA_ROOT}/entity_map.json" "${DATA_ROOT}/pdb_clusters" "${DATA_ROOT}/structure_files"; do
    [[ -e "${required}" ]] || { echo "ERROR: split-only input missing: ${required}" >&2; exit 2; }
  done
  python3 preprocessing/pdb_clusters/split_by_pdb_clusters.py --all
  python3 preprocessing/pdb_clusters/verify_splits.py
  python3 preprocessing/pdb_clusters/analyse_and_plot.py
  echo "Split-and-plot-only run completed: $(date --iso-8601=seconds)"
  exit 0
fi

# Retrieve external inputs unless an existing, complete data directory is explicitly reused.
if [[ "${DGG_SKIP_RETRIEVAL:-0}" == "1" ]]; then
  required_inputs=(
    "${DATA_ROOT}/go-basic.obo"
    "${DATA_ROOT}/pdb_chain_go.tsv"
    "${DATA_ROOT}/viridiplantae_pdb_ids.csv"
    "${DATA_ROOT}/retrieval_manifest.json"
    "${DATA_ROOT}/structure_files"
    "${DATA_ROOT}/pdb_clusters"
  )
  for input_path in "${required_inputs[@]}"; do
    if [[ ! -e "${input_path}" ]]; then
      echo "ERROR: DGG_SKIP_RETRIEVAL=1 but required input is missing: ${input_path}" >&2
      exit 2
    fi
  done
  python3 - "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "retrieval_manifest.json").read_text())
expected = [x.strip() for x in (root / "viridiplantae_pdb_ids.csv").read_text().replace("\n", "").split(",") if x.strip()]
missing = [pdb for pdb in expected if not (root / "structure_files" / f"{pdb}.cif.gz").is_file()]
clusters = [root / "pdb_clusters" / f"clusters-by-entity-{threshold}.txt" for threshold in (30, 40, 50, 70, 90, 95)]
missing_clusters = [str(path) for path in clusters if not path.is_file() or path.stat().st_size == 0]
if missing or missing_clusters:
    raise SystemExit(f"incomplete retrieval: missing CIFs={len(missing)}, missing cluster files={missing_clusters}")
failures = manifest.get("counts", {}).get("structure_download_failures", 0)
if failures:
    raise SystemExit(f"retrieval manifest records {failures} failed structures; do not skip retrieval")
print(f"Validated complete retrieval: {len(expected)} structures and all cluster files present")
PY
  echo "Skipping retrieval; reusing existing data under ${DATA_ROOT}"
else
  python3 preprocessing/pdb_clusters/fetch_data.py \
    --data-dir "${DATA_ROOT}" \
    --workers "${DGG_DOWNLOAD_WORKERS:-${SLURM_CPUS_PER_TASK}}"
fi

# Build the canonical eligible chain universe from the fresh CIF/SIFTS/GO inputs.
python3 preprocessing/pdb_clusters/prepare_dataset.py \
  --data-dir "${DATA_ROOT}" \
  --min-length 60

# Rebuild the structure-derived contact maps used by the model dataset.
python3 preprocessing/create_cmaps.py \
  -annot "${DATA_ROOT}/pdb2go.tsv" \
  -seqs "${DATA_ROOT}/all_sequences.fasta" \
  -struc_dir "${DATA_ROOT}/structure_files" \
  -num_threads "${SLURM_CPUS_PER_TASK}" \
  -chunk_size 50

# Generate all PDB-cluster thresholds. This does not read the legacy split_files.
python3 preprocessing/pdb_clusters/split_by_pdb_clusters.py --all

# Prove disjoint cluster components and exact-sequence non-leakage.
python3 preprocessing/pdb_clusters/verify_splits.py

# Fail before plotting if a split is incomplete, overlapping, or lacks its pickle.
python3 - <<'PY'
import pickle
from pathlib import Path
from preprocessing.pdb_clusters.common import DATA_DIR, DATASET_ROOT, ONTOLOGIES, THRESHOLDS, read_fasta

seqs = read_fasta(DATA_DIR / "all_sequences.fasta")
expected = set(seqs)
for threshold in THRESHOLDS:
    root = DATA_DIR / "pdb_splits" / f"threshold_{threshold}"
    split_ids = {}
    for split in ("train", "valid", "test"):
        ids = {line.strip() for line in (root / f"_{split}.txt").read_text().splitlines() if line.strip()}
        split_ids[split] = ids
        fasta_ids = set()
        for line in (root / f"_{split}_sequences.fasta").read_text().splitlines():
            if line.startswith(">"):
                fasta_ids.add(line[1:].split()[0])
        if ids != fasta_ids:
            raise SystemExit(f"{threshold}% {split}: ID/FASTA mismatch")
        for ontology in ONTOLOGIES:
            path = DATASET_ROOT / f"threshold_{threshold}" / f"{ontology}_{split}.pkl"
            if not path.exists():
                raise SystemExit(f"Missing dataset pickle: {path}")
            with path.open("rb") as handle:
                records = pickle.load(handle)
            if {record["id"] for record in records} != ids:
                raise SystemExit(f"{path}: pickle IDs do not match split IDs")
    if split_ids["train"] & split_ids["valid"] or split_ids["train"] & split_ids["test"] or split_ids["valid"] & split_ids["test"]:
        raise SystemExit(f"{threshold}%: split overlap")
    observed = set().union(*split_ids.values())
    if observed != expected:
        raise SystemExit(f"{threshold}%: split universe mismatch; expected {len(expected)}, got {len(observed)}")
    print(f"verified {threshold}%: {len(observed)} chains")
PY

# Generate the existing 12 diagnostics from the new pdb_splits root.
python3 preprocessing/pdb_clusters/analyse_and_plot.py

echo "Rebuild completed: $(date --iso-8601=seconds)"
