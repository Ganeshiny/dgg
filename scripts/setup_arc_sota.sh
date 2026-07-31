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
GATGO_ENV="${DGG_GATGO_ENV:-dgg_gat_go}"
DEEPGRAPHGO_ENV="${DGG_DEEPGRAPHGO_ENV:-dgg_deepgraphgo}"
SETUP_TOOLS_ENV="${DGG_SETUP_TOOLS_ENV:-dgg_setup_tools}"
INTERPRO_JAVA_ENV="${DGG_INTERPROSCAN_JAVA_ENV:-dgg_interproscan_java11}"

DEEPFRI_ROOT="${DGG_DEEPFRI_ROOT:-${PROJECT_DIR}/baselines/DeepFRI}"
TRANSFUN_ROOT="${DGG_TRANSFUN_ROOT:-${SOTA_ROOT}/TransFun}"
DPFUNC_ROOT="${DGG_DPFUNC_ROOT:-${SOTA_ROOT}/DPFunc}"
DEEPGOPLUS_ROOT="${DGG_DEEPGOPLUS_ROOT:-${SOTA_ROOT}/deepgoplus}"
DEEPGO_ROOT="${DGG_DEEPGO_ROOT:-${SOTA_ROOT}/deepgo2}"
HEAL_ROOT="${DGG_HEAL_ROOT:-${SOTA_ROOT}/HEAL}"
GATGO_ROOT="${DGG_GATGO_ROOT:-${SOTA_ROOT}/GAT-GO}"
DEEPGRAPHGO_ROOT="${DGG_DEEPGRAPHGO_ROOT:-${SOTA_ROOT}/DeepGraphGO}"
# Only read during verification, to report released-feature coverage against
# the locked query set. Setup never writes into the benchmark workspace.
BENCHMARK_ROOT="${DGG_BENCHMARK_ROOT:-${PROJECT_DIR}/arc_benchmark/nominal_30_identity_80_coverage}"
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
HEAL_ESM1B_REGRESSION_URL="${DGG_HEAL_ESM1B_REGRESSION_URL:-https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt}"
GATGO_DATA_URL="${DGG_GATGO_DATA_URL:-https://drive.google.com/drive/folders/1--1zHFqOzB7pZ75G_td_T2e05qfoSlz6?usp=sharing}"
GATGO_REVISION="${DGG_GATGO_REVISION:-90ec6d1067a893d4a51be715e41daf9fa4732952}"
DEEPGRAPHGO_REVISION="${DGG_DEEPGRAPHGO_REVISION:-efdb1cb9425f4f48e4613c0a89e603f5542bcb19}"
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
        deepfri|transfun|dpfunc|deepgoplus|deepgose|heal|gat_go|deepgraphgo|interproscan) ;;
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
    ensure_setup_git
    git_run clone --filter=blob:none "${url}" "${destination}"
}

ensure_setup_tools_env() {
    if ! conda_env_exists "${SETUP_TOOLS_ENV}"; then
        conda create -y -n "${SETUP_TOOLS_ENV}" -c conda-forge git p7zip
    fi
}

# Conda package that provides a given command, where the two names differ.
tool_package() {
    case "$1" in
        7z) printf 'p7zip\n' ;;
        *) printf '%s\n' "$1" ;;
    esac
}

# ARC compute nodes do not expose git, and do not expose 7z either. Prefer a
# system binary; otherwise provision it once in the shared setup-tools
# environment. The per-tool check also repairs environments that an earlier
# revision of this script created with git alone.
tool_run() {
    local tool="$1"
    shift
    if command -v "${tool}" >/dev/null 2>&1; then
        command "${tool}" "$@"
        return
    fi
    ensure_setup_tools_env
    if ! conda run -n "${SETUP_TOOLS_ENV}" sh -c "command -v ${tool}" >/dev/null 2>&1; then
        conda install -y -n "${SETUP_TOOLS_ENV}" -c conda-forge "$(tool_package "${tool}")"
    fi
    conda run --no-capture-output -n "${SETUP_TOOLS_ENV}" "${tool}" "$@"
}

ensure_setup_git() {
    [[ -n "${GIT_MODE:-}" ]] && return
    if command -v git >/dev/null 2>&1; then
        GIT_MODE=system
        echo "[OK] Setup Git: $(command -v git)"
        return
    fi
    GIT_MODE=conda
    ensure_setup_tools_env
    conda run -n "${SETUP_TOOLS_ENV}" git --version
}

git_run() {
    if [[ "${GIT_MODE:-}" == "system" ]]; then
        command git "$@"
    else
        conda run -n "${SETUP_TOOLS_ENV}" git "$@"
    fi
}

pin_repo() {
    ensure_setup_git
    local destination="$1"
    local revision="$2"
    local current
    current="$(git_run -C "${destination}" rev-parse HEAD | tail -n 1)"
    if [[ "${current}" == "${revision}" ]]; then
        echo "[OK] Pinned repository: ${destination} @ ${revision}"
        printf '%s\n' "${revision}" > "${destination}/.dgg_upstream_revision"
        return
    fi
    if ! git_run -C "${destination}" cat-file -e "${revision}^{commit}" 2>/dev/null; then
        git_run -C "${destination}" fetch origin "${revision}"
    fi
    git_run -C "${destination}" checkout --detach "${revision}"
    printf '%s\n' "${revision}" > "${destination}/.dgg_upstream_revision"
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

# Info-ZIP spanned archives keep the central directory in the final ".zip"
# member and the payload in sibling ".z01 ... .zNN" parts, with per-entry
# offsets relative to their own disk. unzip, dtrx, and `zip -s 0` all mangle
# that layout; 7-Zip reads the volume set directly and verifies every CRC,
# which is why p7zip is a hard setup dependency here.
extract_split_zip() {
    local archive="$1"
    local destination="$2"
    local directory base staging top entry part
    directory="$(cd "$(dirname "${archive}")" && pwd)"
    base="$(basename "${archive}" .zip)"
    require_file "${archive}" || return 1
    local parts=()
    shopt -s nullglob
    parts=("${directory}/${base}".z[0-9][0-9])
    shopt -u nullglob
    if ((${#parts[@]} == 0)); then
        echo "[SETUP ERROR] ${archive} is a split archive but no ${base}.zNN parts exist" >&2
        return 1
    fi
    # A partial Git checkout would leave gaps in the volume sequence, which 7z
    # reports only as a truncated extraction. Check for contiguity up front.
    for part in "${parts[@]}"; do
        require_file "${part}" || return 1
    done
    local index expected
    for ((index = 1; index <= ${#parts[@]}; index++)); do
        expected="$(printf '%s/%s.z%02d' "${directory}" "${base}" "${index}")"
        require_file "${expected}" || {
            echo "[SETUP ERROR] Split archive volume sequence is not contiguous: ${expected} is missing" >&2
            return 1
        }
    done
    echo "[SETUP] Extracting ${base}.zip with ${#parts[@]} sibling volumes via 7-Zip."
    staging="${directory}/.${base}_staging"
    rm -rf "${staging}"
    mkdir -p "${staging}" "${destination}"
    if ! tool_run 7z x -y -bso0 -o"${staging}" "${archive}"; then
        echo "[SETUP ERROR] 7-Zip could not extract the split archive: ${archive}" >&2
        rm -rf "${staging}"
        return 1
    fi
    # The release may or may not nest its payload under one top-level folder.
    # Normalize both layouts so callers always find files directly under
    # ${destination}.
    local entries=()
    shopt -s nullglob dotglob
    entries=("${staging}"/*)
    shopt -u nullglob dotglob
    if ((${#entries[@]} == 0)); then
        echo "[SETUP ERROR] Split archive extracted no files: ${archive}" >&2
        rm -rf "${staging}"
        return 1
    fi
    if ((${#entries[@]} == 1)) && [[ -d "${entries[0]}" ]]; then
        top="${entries[0]}"
    else
        top="${staging}"
    fi
    shopt -s nullglob dotglob
    for entry in "${top}"/*; do
        mv -f "${entry}" "${destination}/"
    done
    shopt -u nullglob dotglob
    rm -rf "${staging}"
}

# Unpack any archive a folder download left behind, in place, once. Releases
# that ship loose files are unaffected because the search simply finds nothing.
extract_downloaded_archives() {
    local root="$1"
    local archive extracted=0
    while IFS= read -r -d '' archive; do
        [[ -e "${archive}.dgg_extracted" ]] && continue
        echo "[SETUP] Unpacking ${archive}"
        case "${archive}" in
            *.tar.gz|*.tgz|*.tar.bz2|*.tar.xz|*.tar)
                tar -xf "${archive}" -C "$(dirname "${archive}")"
                ;;
            *.zip)
                if [[ -e "${archive%.zip}.z01" ]]; then
                    extract_split_zip "${archive}" "$(dirname "${archive}")"
                else
                    tool_run 7z x -y -bso0 -o"$(dirname "${archive}")" "${archive}"
                fi
                ;;
        esac
        touch "${archive}.dgg_extracted"
        extracted=$((extracted + 1))
    done < <(find "${root}" \( -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.bz2' \
        -o -name '*.tar.xz' -o -name '*.tar' -o -name '*.zip' \) -print0)
    echo "[SETUP] Unpacked ${extracted} archive(s) under ${root}"
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
    conda install -y -n "${HEAL_ENV}" -c pytorch -c nvidia -c defaults \
        "pytorch=2.1.2" "pytorch-cuda=11.8" "mkl<2024.1"
    conda run -n "${HEAL_ENV}" python -m pip install "torch-geometric==2.5.3" "fair-esm==2.0.0" "biopython>=1.81,<2" "numpy<2" "scikit-learn<2" joblib tqdm
    download_file "${HEAL_ESM1B_MODEL_URL}" "${TORCH_HOME}/hub/checkpoints/esm1b_t33_650M_UR50S.pt"
    # load_model_and_alphabet_local() always reads the contact-regression sidecar
    # next to the checkpoint, even when return_contacts=False. Without it HEAL
    # fails at model load time, after the GPU job has already started.
    download_file "${HEAL_ESM1B_REGRESSION_URL}" \
        "${TORCH_HOME}/hub/checkpoints/esm1b_t33_650M_UR50S-contact-regression.pt"
}

setup_gat_go() {
    clone_repo https://github.com/bl-2633/GAT-GO.git "${GATGO_ROOT}"
    pin_repo "${GATGO_ROOT}" "${GATGO_REVISION}"
    ensure_named_env "${GATGO_ENV}" 3.10
    conda install -y -n "${GATGO_ENV}" -c pytorch -c nvidia -c defaults \
        "pytorch=2.1.2" "pytorch-cuda=11.8" "mkl<2024.1"
    conda run -n "${GATGO_ENV}" python -m pip install \
        "torch-geometric==2.5.3" "numpy<2" "gdown>=5,<6"
    if [[ ! -s "${GATGO_ROOT}/trained_models/GAT-GO_modelweights.pt" \
            || ! -s "${GATGO_ROOT}/data/data_splits/go2index.pt" \
            || ! -d "${GATGO_ROOT}/data/seq_features" ]]; then
        echo "[SETUP] Downloading the official GAT-GO pretrained model and precomputed features."
        # --remaining-ok was previously passed here. It suppresses gdown's
        # 50-file folder listing limit, so an over-limit folder produced a
        # silently truncated feature set that only surfaced as "missing
        # feature file" errors much later, on the GPU node. Fail here instead.
        conda run --no-capture-output -n "${GATGO_ENV}" gdown --folder \
            "${GATGO_DATA_URL}" -O "${GATGO_ROOT}"
        # The release distributes the per-chain features as archives rather
        # than loose .pt files; nothing downstream unpacks them.
        extract_downloaded_archives "${GATGO_ROOT}"
    fi
}

setup_deepgraphgo() {
    clone_repo https://github.com/yourh/DeepGraphGO.git "${DEEPGRAPHGO_ROOT}"
    pin_repo "${DEEPGRAPHGO_ROOT}" "${DEEPGRAPHGO_REVISION}"
    ensure_named_env "${DEEPGRAPHGO_ENV}" 3.8
    conda install -y -n "${DEEPGRAPHGO_ENV}" -c pytorch \
        "pytorch=1.6.0" cpuonly
    conda install -y -n "${DEEPGRAPHGO_ENV}" -c conda-forge -c bioconda \
        "blast>=2.10,<3"
    conda run -n "${DEEPGRAPHGO_ENV}" python -m pip install "pip<24.1"
    # `future` is an undeclared runtime requirement of the PyTorch 1.6.0 wheel;
    # without it `import torch` fails. dtrx is deliberately absent: it cannot
    # read the spanned archive DeepGraphGO publishes (see extract_split_zip).
    conda run -n "${DEEPGRAPHGO_ENV}" python -m pip install \
        "numpy==1.19.2" "scipy==1.5.0" "scikit-learn==0.22.1" \
        "networkx==2.4" "dgl==0.4.3post2" "click==7.1.2" \
        "ruamel.yaml==0.16.6" "biopython==1.78" "tqdm==4.47.0" \
        "logzero==1.5.0" "joblib==0.16.0" "future==0.18.3"
    if [[ ! -s "${DEEPGRAPHGO_ROOT}/data/ppi_interpro.npz" ]]; then
        extract_split_zip "${DEEPGRAPHGO_ROOT}/data/data.zip" "${DEEPGRAPHGO_ROOT}/data"
    fi
    # The published code hard-codes CUDA. Its exact PyTorch 1.6/DGL 0.4 stack
    # cannot execute on ARC's L40 GPUs, so use the same operations on CPU.
    if grep -q 'nn.DataParallel(self.network.cuda())' "${DEEPGRAPHGO_ROOT}/deepgraphgo/models.py"; then
        sed -i 's/\.float()\.cuda()/\.float()/g' "${DEEPGRAPHGO_ROOT}/main.py"
        sed -i 's/self.dp_network = nn.DataParallel(self.network.cuda())/self.device = torch.device("cpu")\n        self.dp_network = self.network.to(self.device)/' \
            "${DEEPGRAPHGO_ROOT}/deepgraphgo/models.py"
        sed -i 's/\.cuda()\.long()/\.to(self.device).long()/g; s/\.cuda()\.float()/\.to(self.device).float()/g; s/train_y\.cuda()/train_y.to(self.device)/g; s/torch.load(self.model_path)/torch.load(self.model_path, map_location=self.device)/g' \
            "${DEEPGRAPHGO_ROOT}/deepgraphgo/models.py"
    fi
    if ! grep -q '^import os$' "${DEEPGRAPHGO_ROOT}/deepgraphgo/psiblast_utils.py"; then
        sed -i '/^from pathlib import Path$/i import os' "${DEEPGRAPHGO_ROOT}/deepgraphgo/psiblast_utils.py"
    fi
    sed -i 's/num_threads=40/num_threads=int(os.environ.get("DGG_DEEPGRAPHGO_THREADS", "8"))/' \
        "${DEEPGRAPHGO_ROOT}/deepgraphgo/psiblast_utils.py"
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
        check require_file "${TORCH_HOME}/hub/checkpoints/esm1b_t33_650M_UR50S-contact-regression.pt"
        check conda run -n "${HEAL_ENV}" python -c "import Bio, esm, numpy, torch, torch_geometric; print('HEAL runtime imports ok')"
        # Load the checkpoint pair exactly as run_heal_arc.py does, so a missing
        # or truncated sidecar is caught here rather than on the GPU node.
        check conda run -n "${HEAL_ENV}" python -c \
            "import esm; esm.pretrained.load_model_and_alphabet_local('${TORCH_HOME}/hub/checkpoints/esm1b_t33_650M_UR50S.pt'); print('HEAL ESM-1b checkpoint pair loads')"
        check conda run -n "${HEAL_ENV}" python "${PROJECT_DIR}/scripts/run_heal_arc.py" --help
    fi
    if method_enabled gat_go; then
        [[ "$(git_run -C "${GATGO_ROOT}" rev-parse HEAD | tail -n 1)" == "${GATGO_REVISION}" ]] || failures=$((failures + 1))
        check require_file "${GATGO_ROOT}/GAT-GO.py"
        check require_file "${GATGO_ROOT}/src/GnnPF.py"
        check require_file "${GATGO_ROOT}/trained_models/GAT-GO_modelweights.pt"
        check require_file "${GATGO_ROOT}/data/data_splits/go2index.pt"
        [[ -d "${GATGO_ROOT}/data/seq_features" ]] || failures=$((failures + 1))
        check conda run -n "${GATGO_ENV}" python -c \
            "import torch, torch_geometric; from torch_geometric.nn import GATConv, SAGPooling; print(torch.__version__, torch_geometric.__version__, torch.cuda.is_available())"
        check conda run -n "${GATGO_ENV}" python "${PROJECT_DIR}/scripts/run_gat_go_arc.py" --help
        check conda run -n "${GATGO_ENV}" python "${PROJECT_DIR}/scripts/run_gat_go_arc.py" \
            --gat-root "${GATGO_ROOT}" --feature-root "${GATGO_ROOT}/data/seq_features" \
            --model "${GATGO_ROOT}/trained_models/GAT-GO_modelweights.pt" \
            --go-map "${GATGO_ROOT}/data/data_splits/go2index.pt" --runtime-smoke-test
        # Whether the released feature bundle covers the locked ARC query set is
        # a property of the release, not of this environment, so it is reported
        # rather than counted as a setup failure. Measuring it here means an
        # uncoverable query set is known before any GPU job is queued.
        if [[ -s "${BENCHMARK_ROOT}/inputs/valid_test.fasta" ]]; then
            conda run --no-capture-output -n "${GATGO_ENV}" python \
                "${PROJECT_DIR}/scripts/run_gat_go_arc.py" \
                --gat-root "${GATGO_ROOT}" --feature-root "${GATGO_ROOT}/data/seq_features" \
                --model "${GATGO_ROOT}/trained_models/GAT-GO_modelweights.pt" \
                --go-map "${GATGO_ROOT}/data/data_splits/go2index.pt" \
                --fasta "${BENCHMARK_ROOT}/inputs/valid_test.fasta" \
                --workspace "${BENCHMARK_ROOT}" \
                --output-dir "${BENCHMARK_ROOT}/raw/gat_go" \
                --report-coverage-only || true
        else
            echo "[SETUP] Benchmark inputs are not prepared yet; skipping the GAT-GO coverage report."
        fi
    fi
    if method_enabled deepgraphgo; then
        [[ "$(git_run -C "${DEEPGRAPHGO_ROOT}" rev-parse HEAD | tail -n 1)" == "${DEEPGRAPHGO_REVISION}" ]] || failures=$((failures + 1))
        check require_file "${DEEPGRAPHGO_ROOT}/main.py"
        check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_pid_list.txt"
        check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_interpro.npz"
        check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_dgl_top_100"
        check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_blastdb.pin"
        # run_deepgraphgo_arc.py writes ppi_mat.npz into every generated data
        # config, so a missing weight matrix must fail setup, not inference.
        check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_mat.npz"
        for ontology in mf bp cc; do
            check require_file "${DEEPGRAPHGO_ROOT}/data/${ontology}_go.mlb"
            for model_index in 0 1 2; do
                check require_file "${DEEPGRAPHGO_ROOT}/models/DeepGraphGO-Model-${model_index}-${ontology}"
            done
        done
        check conda run -n "${DEEPGRAPHGO_ENV}" psiblast -version
        check conda run -n "${DEEPGRAPHGO_ENV}" python -c \
            "import dgl, numpy, scipy, sklearn, torch; assert not torch.cuda.is_available(); print('DeepGraphGO legacy CPU runtime ok')"
        check conda run -n "${DEEPGRAPHGO_ENV}" python "${DEEPGRAPHGO_ROOT}/main.py" --help
        check conda run -n "${DEEPGRAPHGO_ENV}" python "${PROJECT_DIR}/scripts/run_deepgraphgo_arc.py" --help
        check conda run -n "${DEEPGRAPHGO_ENV}" python "${PROJECT_DIR}/scripts/run_deepgraphgo_arc.py" \
            --deepgraphgo-root "${DEEPGRAPHGO_ROOT}" --runtime-smoke-test
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
method_enabled gat_go && setup_gat_go
method_enabled deepgraphgo && setup_deepgraphgo
method_enabled interproscan && setup_interproscan

verify_setup
