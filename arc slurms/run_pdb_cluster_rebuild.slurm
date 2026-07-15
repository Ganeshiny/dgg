#!/usr/bin/env bash
#SBATCH --job-name=dgg_pdb_rebuild
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=pdb_rebuild_%j.out
#SBATCH --error=pdb_rebuild_%j.err

set -euo pipefail

PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"
cd "${PROJECT_DIR}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
export DGG_DATA_ROOT="${DATA_ROOT}"
echo "Data root: ${DGG_DATA_ROOT}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${DGG_CONDA_ENV:-deepgreengo}"

command -v python3 >/dev/null
command -v blastp >/dev/null
command -v makeblastdb >/dev/null

mkdir -p "${DATA_ROOT}"

# Fresh retrieval: GO, SIFTS, RCSB entity query, cluster files, and CIF files.
python3 preprocessing/pdb_clusters/fetch_data.py \
  --data-dir "${DATA_ROOT}" \
  --workers "${DGG_DOWNLOAD_WORKERS:-${SLURM_CPUS_PER_TASK}}"

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
