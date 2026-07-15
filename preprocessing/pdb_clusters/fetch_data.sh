#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export DGG_DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data}"

cd "${PROJECT_DIR}"
python3 preprocessing/pdb_clusters/fetch_data.py \
    --data-dir "${DGG_DATA_ROOT}" \
    --workers "${DGG_DOWNLOAD_WORKERS:-16}" \
    "$@"
