#!/usr/bin/env bash
# Install and verify the external deep-learning methods used by the ARC benchmark.
# This script is intentionally strict: it exits non-zero unless every requested
# environment, repository, model archive, and InterProScan dependency is usable.

set -euo pipefail

PROJECT_DIR="${DGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOTA_ROOT="${DGG_SOTA_ROOT:-${PROJECT_DIR}/SOTA}"
DOWNLOAD_ROOT="${DGG_SOTA_DOWNLOAD_ROOT:-${SOTA_ROOT}/downloads}"
SETUP_METHODS="${DGG_SOTA_SETUP_METHODS:-deepfri,transfun,dpfunc,deepgoplus,deepgose,interproscan}"

DEEPFRI_ENV="${DGG_DEEPFRI_ENV:-dgg_sota_tf}"
TRANSFUN_ENV="${DGG_TRANSFUN_ENV:-dgg_transfun}"
DPFUNC_ENV="${DGG_DPFUNC_ENV:-dgg_dpfunc}"
DEEPGOPLUS_ENV="${DGG_DEEPGOPLUS_ENV:-dgg_deepgoplus_py37}"
DEEPGO_ENV="${DGG_DEEPGO_ENV:-dgg_deepgose}"

DEEPFRI_ROOT="${DGG_DEEPFRI_ROOT:-${PROJECT_DIR}/baselines/DeepFRI}"
TRANSFUN_ROOT="${DGG_TRANSFUN_ROOT:-${SOTA_ROOT}/TransFun}"
DPFUNC_ROOT="${DGG_DPFUNC_ROOT:-${SOTA_ROOT}/DPFunc}"
DEEPGOPLUS_ROOT="${DGG_DEEPGOPLUS_ROOT:-${SOTA_ROOT}/deepgoplus}"
DEEPGO_ROOT="${DGG_DEEPGO_ROOT:-${SOTA_ROOT}/deepgo2}"
INTERPRO_VERSION="${DGG_INTERPRO_VERSION:-5.78-109.0}"
INTERPRO_ROOT="${DGG_INTERPRO_ROOT:-${SOTA_ROOT}/interproscan-${INTERPRO_VERSION}}"

DEEPFRI_MODELS_URL="${DGG_DEEPFRI_MODELS_URL:-https://users.flatironinstitute.org/~renfrew/DeepFRI_data/trained_models.tar.gz}"
TRANSFUN_DATA_URL="${DGG_TRANSFUN_DATA_URL:-https://calla.rnet.missouri.edu/rnaminer/transfun/data}"
DPFUNC_MODELS_GDRIVE_ID="${DGG_DPFUNC_MODELS_GDRIVE_ID:-1V0VTFTiB29ilbAIOZn0okBQWPlbOI3wN}"
DEEPGOPLUS_DATA_URL="${DGG_DEEPGOPLUS_DATA_URL:-http://deepgoplus.bio2vec.net/data/data.tar.gz}"
DEEPGO_DATA_URL="${DGG_DEEPGO_DATA_URL:-https://deepgo.cbrc.kaust.edu.sa/data/deepgo2/data.tar.gz}"
INTERPRO_URL="${DGG_INTERPRO_URL:-https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/${INTERPRO_VERSION}/interproscan-${INTERPRO_VERSION}-64-bit.tar.gz}"

mkdir -p "${SOTA_ROOT}" "${DOWNLOAD_ROOT}"

if [[ -z "${SETUP_METHODS}" ]]; then
    echo "[SETUP ERROR] DGG_SOTA_SETUP_METHODS must not be empty" >&2
    exit 1
fi
IFS=',' read -r -a requested_methods <<< "${SETUP_METHODS}"
for requested_method in "${requested_methods[@]}"; do
    case "${requested_method}" in
        deepfri|transfun|dpfunc|deepgoplus|deepgose|interproscan) ;;
        *)
            echo "[SETUP ERROR] Unknown SOTA setup method: ${requested_method}" >&2
            exit 1
            ;;
    esac
done

method_enabled() {
    [[ ",${SETUP_METHODS}," == *",$1,"* ]]
}

require_file() {
    [[ -s "$1" ]] || { echo "[SETUP ERROR] Missing or empty file: $1" >&2; return 1; }
}

require_executable() {
    [[ -x "$1" ]] || { echo "[SETUP ERROR] Missing executable: $1" >&2; return 1; }
}

conda_env_exists() {
    conda run -n "$1" true >/dev/null 2>&1
}

clone_repo() {
    local url="$1"
    local destination="$2"
    if [[ -d "${destination}/.git" ]]; then
        echo "[OK] Repository exists: ${destination}"
        return
    fi
    if [[ -e "${destination}" ]]; then
        echo "[SETUP ERROR] ${destination} exists but is not a Git checkout" >&2
        return 1
    fi
    git clone --filter=blob:none "${url}" "${destination}"
}

download_file() {
    local url="$1"
    local destination="$2"
    if [[ -s "${destination}" ]]; then
        echo "[OK] Download exists: ${destination}"
        return
    fi
    local partial="${destination}.partial"
    local attempt
    local max_attempts="${DGG_DOWNLOAD_ATTEMPTS:-20}"
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        echo "[DOWNLOAD] Attempt ${attempt}/${max_attempts}: ${url}"
        if curl --fail --location --http1.1 --connect-timeout 60 \
                --continue-at - --output "${partial}" "${url}"; then
            require_file "${partial}"
            mv "${partial}" "${destination}"
            return
        fi
        echo "[DOWNLOAD WARNING] Transfer interrupted; preserving ${partial} for resume" >&2
        sleep 15
    done
    echo "[SETUP ERROR] Download failed after ${max_attempts} attempts: ${url}" >&2
    return 1
}

extract_archive() {
    local archive="$1"
    local destination="$2"
    mkdir -p "${destination}"
    if unzip -tq "${archive}" >/dev/null 2>&1; then
        unzip -oq "${archive}" -d "${destination}"
    else
        tar -xf "${archive}" -C "${destination}"
    fi
}

ensure_named_env() {
    local name="$1"
    local python_version="$2"
    if ! conda_env_exists "${name}"; then
        conda create -y -n "${name}" "python=${python_version}" pip
    fi
}

setup_deepfri() {
    clone_repo https://github.com/flatironinstitute/DeepFRI.git "${DEEPFRI_ROOT}"
    if [[ ! -s "${DEEPFRI_ROOT}/trained_models/model_config.json" ]]; then
        local archive="${DOWNLOAD_ROOT}/deepfri_trained_models.tar.gz"
        download_file "${DEEPFRI_MODELS_URL}" "${archive}"
        extract_archive "${archive}" "${DEEPFRI_ROOT}"
    fi
    ensure_named_env "${DEEPFRI_ENV}" 3.9
    conda run -n "${DEEPFRI_ENV}" python -m pip install \
        "tensorflow==2.13.0" "numpy<2" biopython scikit-learn h5py networkx
    # DeepFRI's setup.py pins its 2020 dependency stack (including NumPy 1.18.5
    # and tensorflow-gpu 2.3.1), which cannot be resolved on Python 3.9. The
    # compatible runtime dependencies are installed explicitly above.
    conda run -n "${DEEPFRI_ENV}" python -m pip install --no-deps -e "${DEEPFRI_ROOT}"
}

setup_transfun() {
    clone_repo https://github.com/jianlin-cheng/TransFun.git "${TRANSFUN_ROOT}"
    if ! conda_env_exists "${TRANSFUN_ENV}"; then
        conda env create -n "${TRANSFUN_ENV}" -f "${TRANSFUN_ROOT}/environment.yml"
    fi
    # TransFun's older PyTorch build cannot load against MKL 2024.1+ because
    # libtorch_cpu.so still expects the removed iJIT_NotifyEvent symbol.
    # ARC's defaults channel no longer exposes MKL 2024.0, but 2023.1.0 is
    # available there and also selects the matching pre-2024.1 OpenMP runtime.
    conda install -y -n "${TRANSFUN_ENV}" "mkl=2023.1.0"
    if [[ ! -s "${TRANSFUN_ROOT}/data/molecular_function.pt" ]]; then
        local archive="${DOWNLOAD_ROOT}/transfun_data.zip"
        download_file "${TRANSFUN_DATA_URL}" "${archive}"
        extract_archive "${archive}" "${TRANSFUN_ROOT}"
    fi
}

setup_dpfunc() {
    clone_repo https://github.com/CSUBioGroup/DPFunc.git "${DPFUNC_ROOT}"
    if ! conda_env_exists "${DPFUNC_ENV}"; then
        conda env create -n "${DPFUNC_ENV}" -f "${DPFUNC_ROOT}/DPFunc_env.yml"
    fi
    conda run -n "${DPFUNC_ENV}" python -m pip install "gdown>=5,<6"
    if [[ ! -s "${DPFUNC_ROOT}/save_models/DPFunc_model_mf_0of3model.pt" ]]; then
        local archive="${DOWNLOAD_ROOT}/dpfunc_models.archive"
        if [[ ! -s "${archive}" ]]; then
            conda run -n "${DPFUNC_ENV}" gdown \
                "https://drive.google.com/uc?id=${DPFUNC_MODELS_GDRIVE_ID}" -O "${archive}"
        fi
        extract_archive "${archive}" "${DPFUNC_ROOT}"
    fi
}

setup_deepgoplus() {
    clone_repo https://github.com/bio-ontology-research-group/deepgoplus.git "${DEEPGOPLUS_ROOT}"
    # PyPI metadata for DeepGOPlus 1.0.2 requires Python >=3.7,<3.8.
    ensure_named_env "${DEEPGOPLUS_ENV}" 3.7
    conda run -n "${DEEPGOPLUS_ENV}" python -m pip install "deepgoplus==1.0.2"
    if [[ ! -s "${DEEPGOPLUS_ROOT}/data/model.h5" ]]; then
        local archive="${DOWNLOAD_ROOT}/deepgoplus_data.tar.gz"
        download_file "${DEEPGOPLUS_DATA_URL}" "${archive}"
        extract_archive "${archive}" "${DEEPGOPLUS_ROOT}"
    fi
}

setup_deepgose() {
    clone_repo https://github.com/bio-ontology-research-group/deepgo2.git "${DEEPGO_ROOT}"
    ensure_named_env "${DEEPGO_ENV}" 3.10
    conda run -n "${DEEPGO_ENV}" python -m pip install \
        "torch==2.0.1" "torchvision==0.15.2" "torchaudio==2.0.2"
    conda run -n "${DEEPGO_ENV}" python -m pip install \
        "dgl==1.1.2+cu117" -f https://data.dgl.ai/wheels/cu117/repo.html
    conda run -n "${DEEPGO_ENV}" python -m pip install -r "${DEEPGO_ROOT}/requirements.txt"
    if [[ ! -s "${DEEPGO_ROOT}/data/go-plus.norm" ]]; then
        local archive="${DOWNLOAD_ROOT}/deepgose_data.tar.gz"
        download_file "${DEEPGO_DATA_URL}" "${archive}"
        extract_archive "${archive}" "${DEEPGO_ROOT}"
    fi
}

setup_interproscan() {
    if [[ ! -x "${INTERPRO_ROOT}/interproscan.sh" ]]; then
        local archive="${DOWNLOAD_ROOT}/interproscan-${INTERPRO_VERSION}-64-bit.tar.gz"
        local checksum="${archive}.md5"
        download_file "${INTERPRO_URL}" "${archive}"
        download_file "${INTERPRO_URL}.md5" "${checksum}"
        (cd "${DOWNLOAD_ROOT}" && md5sum -c "$(basename "${checksum}")")
        extract_archive "${archive}" "${SOTA_ROOT}"
    fi
    require_executable "${INTERPRO_ROOT}/interproscan.sh"
    if [[ -L "${SOTA_ROOT}/interproscan" || ! -e "${SOTA_ROOT}/interproscan" ]]; then
        ln -sfn "${INTERPRO_ROOT}" "${SOTA_ROOT}/interproscan"
    fi
}

verify_setup() {
    local failures=0
    check() { "$@" || failures=$((failures + 1)); }

    if method_enabled deepfri; then
        check require_file "${DEEPFRI_ROOT}/predict.py"
        check require_file "${DEEPFRI_ROOT}/trained_models/model_config.json"
        check conda run -n "${DEEPFRI_ENV}" python -c "import tensorflow"
    fi
    if method_enabled transfun; then
        check require_file "${TRANSFUN_ROOT}/predict.py"
        for ontology in molecular_function biological_process cellular_component; do
            check require_file "${TRANSFUN_ROOT}/data/${ontology}.pt"
        done
        check conda run -n "${TRANSFUN_ENV}" python -c "import biopandas, esm, torch, torch_geometric"
    fi
    if method_enabled dpfunc; then
        check require_file "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/build_data_demo.py"
        check require_file "${DPFUNC_ROOT}/data/inter_idx.pkl"
        for ontology in mf bp cc; do
            check require_file "${DPFUNC_ROOT}/mlb/${ontology}_go.mlb"
            for model_index in 0 1 2; do
                check require_file "${DPFUNC_ROOT}/save_models/DPFunc_model_${ontology}_${model_index}of3model.pt"
            done
        done
        check conda run -n "${DPFUNC_ENV}" python -c "import dgl, logzero, torch"
    fi
    if method_enabled deepgoplus; then
        for file in go.obo model.h5 terms.pkl train_data.pkl; do
            check require_file "${DEEPGOPLUS_ROOT}/data/${file}"
        done
        check conda run -n "${DEEPGOPLUS_ENV}" deepgoplus --help
    fi
    if method_enabled deepgose; then
        check require_file "${DEEPGO_ROOT}/predict.py"
        check require_file "${DEEPGO_ROOT}/data/go.obo"
        check require_file "${DEEPGO_ROOT}/data/go-plus.norm"
        for ontology in mf bp cc; do
            check require_file "${DEEPGO_ROOT}/data/${ontology}/terms.pkl"
        done
        for model_index in 0 1 2 5 6 8; do
            check require_file "${DEEPGO_ROOT}/data/mf/deepgozero_esm_plus_${model_index}.th"
        done
        for model_index in 2 5 6 7 8 9; do
            check require_file "${DEEPGO_ROOT}/data/bp/deepgozero_esm_plus_${model_index}.th"
        done
        for model_index in 1 3 4 5 6 7; do
            check require_file "${DEEPGO_ROOT}/data/cc/deepgozero_esm_plus_${model_index}.th"
        done
        check conda run -n "${DEEPGO_ENV}" python -c "import dgl, torch"
    fi
    if method_enabled interproscan; then
        check require_executable "${INTERPRO_ROOT}/interproscan.sh"
        check require_executable "${SOTA_ROOT}/interproscan/interproscan.sh"
    fi

    if ((failures > 0)); then
        echo "[SETUP FAILED] ${failures} verification check(s) failed" >&2
        return 1
    fi
    echo "[SETUP COMPLETE] All requested SOTA dependencies and model files passed verification"
    echo "Use DGG_INTERPROSCAN=${INTERPRO_ROOT}/interproscan.sh when submitting the benchmark."
}

source "$(conda info --base)/etc/profile.d/conda.sh"

method_enabled deepfri && setup_deepfri
method_enabled transfun && setup_transfun
method_enabled dpfunc && setup_dpfunc
method_enabled deepgoplus && setup_deepgoplus
method_enabled deepgose && setup_deepgose
method_enabled interproscan && setup_interproscan

verify_setup
