#!/usr/bin/env bash
# Install and verify the external deep-learning methods used by the ARC benchmark.
# This script is intentionally strict: it exits non-zero unless every requested
# environment, repository, model archive, and InterProScan dependency is usable.

set -euo pipefail

PROJECT_DIR="${DGG_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOTA_ROOT="${DGG_SOTA_ROOT:-${PROJECT_DIR}/SOTA}"
DOWNLOAD_ROOT="${DGG_SOTA_DOWNLOAD_ROOT:-${SOTA_ROOT}/downloads}"
SETUP_METHODS="${DGG_SOTA_SETUP_METHODS:-deepfri,dpfunc,deepgoplus,deepgose,interproscan}"

DEEPFRI_ENV="${DGG_DEEPFRI_ENV:-dgg_sota_tf}"
TRANSFUN_ENV="${DGG_TRANSFUN_ENV:-dgg_transfun}"
DPFUNC_ENV="${DGG_DPFUNC_ENV:-dgg_dpfunc}"
DEEPGOPLUS_ENV="${DGG_DEEPGOPLUS_ENV:-dgg_deepgoplus_py37}"
DEEPGO_ENV="${DGG_DEEPGO_ENV:-dgg_deepgose}"
HEAL_ENV="${DGG_HEAL_ENV:-dgg_heal}"
INTERPRO_JAVA_ENV="${DGG_INTERPROSCAN_JAVA_ENV:-dgg_interproscan_java11}"

DEEPFRI_ROOT="${DGG_DEEPFRI_ROOT:-${PROJECT_DIR}/baselines/DeepFRI}"
TRANSFUN_ROOT="${DGG_TRANSFUN_ROOT:-${SOTA_ROOT}/TransFun}"
DPFUNC_ROOT="${DGG_DPFUNC_ROOT:-${SOTA_ROOT}/DPFunc}"
DEEPGOPLUS_ROOT="${DGG_DEEPGOPLUS_ROOT:-${SOTA_ROOT}/deepgoplus}"
DEEPGO_ROOT="${DGG_DEEPGO_ROOT:-${SOTA_ROOT}/deepgo2}"
HEAL_ROOT="${DGG_HEAL_ROOT:-${SOTA_ROOT}/HEAL}"
STRUCT2GO_ROOT="${DGG_STRUCT2GO_ROOT:-${SOTA_ROOT}/Struct2GO}"
INTERPRO_VERSION="${DGG_INTERPRO_VERSION:-5.78-109.0}"
INTERPRO_ROOT="${DGG_INTERPRO_ROOT:-${SOTA_ROOT}/interproscan-${INTERPRO_VERSION}}"
TORCH_HOME="${DGG_TORCH_HOME:-${SOTA_ROOT}/torch_cache}"
export TORCH_HOME

DEEPFRI_MODELS_URL="${DGG_DEEPFRI_MODELS_URL:-https://users.flatironinstitute.org/~renfrew/DeepFRI_data/trained_models.tar.gz}"
TRANSFUN_DATA_URL="${DGG_TRANSFUN_DATA_URL:-https://calla.rnet.missouri.edu/rnaminer/transfun/data}"
DPFUNC_MODELS_GDRIVE_ID="${DGG_DPFUNC_MODELS_GDRIVE_ID:-1V0VTFTiB29ilbAIOZn0okBQWPlbOI3wN}"
DPFUNC_ESM2_MODEL_URL="${DGG_DPFUNC_ESM2_MODEL_URL:-https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt}"
DPFUNC_ESM2_REGRESSION_URL="${DGG_DPFUNC_ESM2_REGRESSION_URL:-https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt}"
HEAL_ESM1B_MODEL_URL="${DGG_HEAL_ESM1B_MODEL_URL:-https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt}"
DEEPGOPLUS_DATA_URL="${DGG_DEEPGOPLUS_DATA_URL:-http://deepgoplus.bio2vec.net/data/data.tar.gz}"
DEEPGO_DATA_URL="${DGG_DEEPGO_DATA_URL:-https://deepgo.cbrc.kaust.edu.sa/data/deepgo2/data.tar.gz}"
INTERPRO_URL="${DGG_INTERPRO_URL:-https://ftp.ebi.ac.uk/pub/software/unix/iprscan/5/${INTERPRO_VERSION}/interproscan-${INTERPRO_VERSION}-64-bit.tar.gz}"

mkdir -p "${SOTA_ROOT}" "${DOWNLOAD_ROOT}" "${TORCH_HOME}/hub/checkpoints"

if [[ -z "${SETUP_METHODS}" ]]; then
    echo "[SETUP ERROR] DGG_SOTA_SETUP_METHODS must not be empty" >&2
    exit 1
fi
IFS=',' read -r -a requested_methods <<< "${SETUP_METHODS}"
for requested_method in "${requested_methods[@]}"; do
    case "${requested_method}" in
        deepfri|transfun|dpfunc|deepgoplus|deepgose|heal|struct2go|interproscan) ;;
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

require_java11_env() {
    local environment="$1"
    local version
    version="$(conda run -n "${environment}" java -version 2>&1 | head -n 1)"
    case "${version}" in
        *'version "11.'*|*'version "11"'*)
            echo "[OK] ${environment}: ${version}"
            return 0
            ;;
        *)
            echo "[SETUP ERROR] ${environment} does not provide Java 11: ${version:-no java found}" >&2
            return 1
            ;;
    esac
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
    # TransFun's older PyTorch build cannot load against MKL 2024.1+, which
    # removed the iJIT_NotifyEvent symbol libtorch_cpu.so still expects.
    #
    # Do NOT pin an explicit MKL version here. TransFun's environment.yml pins
    # pytorch=1.10.2, and that package already requires mkl >=2021.4.0,<2022.0a0
    # — comfortably below 2024.1 — so a correctly solved environment needs no
    # intervention at all. Two previous attempts to force one both made the
    # solve unsatisfiable: mkl=2024.0 is absent from ARC's defaults channel,
    # and mkl=2023.1.0 contradicts pytorch's own <2022 bound, producing
    #   "pytorch 1.10.2 would require mkl >=2021.4.0,<2022.0a0, which conflicts"
    # A bare `conda install` also resolves against defaults alone, dropping the
    # pytorch/pyg/conda-forge channels the environment was built from, so the
    # solver cannot see the builds it needs.
    #
    # Inspect first, act only if the installed MKL is genuinely too new, and
    # then use a range plus the original channels so the solver has room.
    local mkl_version
    mkl_version="$(conda list -n "${TRANSFUN_ENV}" '^mkl$' 2>/dev/null \
        | awk '$1 == "mkl" { print $2; exit }')"
    if [[ -z "${mkl_version}" ]]; then
        echo "[SETUP] ${TRANSFUN_ENV}: no conda-managed MKL present; nothing to constrain."
    elif [[ "$(printf '2024.1\n%s\n' "${mkl_version}" | sort -V | head -1)" == "2024.1" ]]; then
        echo "[SETUP] ${TRANSFUN_ENV}: MKL ${mkl_version} >= 2024.1 lacks iJIT_NotifyEvent; constraining to <2024.1"
        conda install -y -n "${TRANSFUN_ENV}" \
            -c pytorch -c pyg -c conda-forge -c defaults "mkl<2024.1"
    else
        echo "[SETUP] ${TRANSFUN_ENV}: MKL ${mkl_version} is already compatible with pytorch 1.10.2."
    fi
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
    # Upstream DPFunc passes mmap=True to torch.load(), but the PyTorch version
    # pinned by DPFunc_env.yml predates that keyword. Remove only that optional
    # optimization so pretrained checkpoints load normally on ARC.
    if grep -q 'mmap=True' "${DPFUNC_ROOT}/DPFunc_pred.py"; then
        sed -i 's/, mmap=True//g' "${DPFUNC_ROOT}/DPFunc_pred.py"
        echo "[SETUP] Patched DPFunc checkpoint loading for the pinned PyTorch runtime."
    fi
    # The current DPFunc preprocessing pipeline imports `esm` and loads
    # esm2_t33_650M_UR50D, but upstream DPFunc_env.yml does not declare the
    # package that provides that module. Install it explicitly and populate the
    # shared Torch checkpoint cache during setup, rather than discovering the
    # missing dependency/download after the GPU benchmark has started.
    conda run -n "${DPFUNC_ENV}" python -m pip install \
        "gdown>=5,<6" "fair-esm==2.0.0"
    download_file "${DPFUNC_ESM2_MODEL_URL}" \
        "${TORCH_HOME}/hub/checkpoints/esm2_t33_650M_UR50D.pt"
    download_file "${DPFUNC_ESM2_REGRESSION_URL}" \
        "${TORCH_HOME}/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt"
    conda run -n "${DPFUNC_ENV}" python -c \
        "import esm; print('DPFunc ESM import ok; cached checkpoints verified separately')"
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
    # The 1.0.2 console entry point requires release-specific ensemble weights,
    # but the published data archive can omit this metadata file. These are the
    # latest-model weights documented by the upstream predictor.
    local metadata="${DEEPGOPLUS_ROOT}/data/metadata/last_release.json"
    if [[ ! -s "${metadata}" ]]; then
        mkdir -p "$(dirname "${metadata}")"
        printf '%s\n' '{"alphas":{"mf":0.55,"bp":0.59,"cc":0.46}}' > "${metadata}"
        echo "[SETUP] Restored required DeepGOPlus 1.0.2 release metadata."
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


setup_heal() {
    clone_repo https://github.com/ZhonghuiGu/HEAL.git "${HEAL_ROOT}"
    ensure_named_env "${HEAL_ENV}" 3.10
    conda install -y -n "${HEAL_ENV}" -c pytorch -c nvidia "pytorch=2.1.2" "pytorch-cuda=11.8"
    conda run -n "${HEAL_ENV}" python -m pip install "torch-geometric==2.5.3" "fair-esm==2.0.0" "biopython>=1.81,<2" "numpy<2" "scikit-learn<2" joblib tqdm
    download_file "${HEAL_ESM1B_MODEL_URL}" "${TORCH_HOME}/hub/checkpoints/esm1b_t33_650M_UR50S.pt"
}

setup_struct2go() {
    clone_repo https://github.com/lyjps/Struct2GO.git "${STRUCT2GO_ROOT}"
    echo "[SETUP NOTICE] Struct2GO ships fixed human-protein datasets, not a"
    echo "general PDB inference entry point. The ARC benchmark accepts externally"
    echo "prepared Struct2GO score files but does not relabel its bundled test set"
    echo "as predictions for the locked ARC proteins."
}

setup_interproscan() {
    if ! conda_env_exists "${INTERPRO_JAVA_ENV}"; then
        conda create -y -n "${INTERPRO_JAVA_ENV}" --override-channels \
            -c conda-forge "openjdk=11"
    elif ! require_java11_env "${INTERPRO_JAVA_ENV}"; then
        conda install -y -n "${INTERPRO_JAVA_ENV}" --override-channels \
            -c conda-forge "openjdk=11"
    fi
    require_java11_env "${INTERPRO_JAVA_ENV}"
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
        check conda run -n "${DEEPFRI_ENV}" python "${DEEPFRI_ROOT}/predict.py" --help
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
        check require_file "${TORCH_HOME}/hub/checkpoints/esm2_t33_650M_UR50D.pt"
        check require_file "${TORCH_HOME}/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt"
        check conda run -n "${DPFUNC_ENV}" python "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/build_data_demo.py" --help
        check conda run -n "${DPFUNC_ENV}" python "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/run_dpfunc.py" --help
        for ontology in mf bp cc; do
            check require_file "${DPFUNC_ROOT}/mlb/${ontology}_go.mlb"
            for model_index in 0 1 2; do
                check require_file "${DPFUNC_ROOT}/save_models/DPFunc_model_${ontology}_${model_index}of3model.pt"
            done
        done
        check conda run -n "${DPFUNC_ENV}" python -c \
            "import dgl, esm, logzero, torch; print('DPFunc runtime imports and ESM2 cache files verified')"
    fi
    if method_enabled deepgoplus; then
        for file in go.obo model.h5 terms.pkl train_data.pkl train_data.dmnd; do
            check require_file "${DEEPGOPLUS_ROOT}/data/${file}"
        done
        check require_file "${DEEPGOPLUS_ROOT}/data/metadata/last_release.json"
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
        check conda run -n "${DEEPGO_ENV}" python -c "import dgl, esm, torch"
        check conda run -n "${DEEPGO_ENV}" python "${DEEPGO_ROOT}/predict.py" --help
    fi
    if method_enabled heal; then
        check require_file "${HEAL_ROOT}/network.py"
        check require_file "${HEAL_ROOT}/utils.py"
        check require_file "${HEAL_ROOT}/data/nrPDB-GO_2019.06.18_annot.tsv"
        for ontology in mf bp cc; do
            check require_file "${HEAL_ROOT}/model/model_${ontology}CL.pt"
            check require_file "${HEAL_ROOT}/model/model_${ontology}CLaf.pt"
        done
        check require_file "${TORCH_HOME}/hub/checkpoints/esm1b_t33_650M_UR50S.pt"
        check conda run -n "${HEAL_ENV}" python -c "import Bio, esm, numpy, torch, torch_geometric; print('HEAL runtime imports ok')"
        check conda run -n "${HEAL_ENV}" python "${PROJECT_DIR}/scripts/run_heal_arc.py" --help
    fi
    if method_enabled struct2go; then
        check require_file "${STRUCT2GO_ROOT}/eval_Struct2GO.py"
        for ontology in mf bp cc; do
            check require_file "${STRUCT2GO_ROOT}/save_models/mymodel_${ontology}_1_0.0005_0.45.pkl"
            check require_file "${STRUCT2GO_ROOT}/processed_data/label_${ontology}_network"
        done
    fi
    if method_enabled interproscan; then
        check require_executable "${INTERPRO_ROOT}/interproscan.sh"
        check require_executable "${SOTA_ROOT}/interproscan/interproscan.sh"
        check require_java11_env "${INTERPRO_JAVA_ENV}"
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
method_enabled heal && setup_heal
method_enabled struct2go && setup_struct2go
method_enabled interproscan && setup_interproscan

verify_setup
