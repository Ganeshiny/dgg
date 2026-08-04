#!/usr/bin/env bash
# One-entry ARC benchmark for the locked nominal 30%-identity/80%-coverage split.
# Submit from the project root with:
#   sbatch 'arc slurms/run_full_benchmark.slurm'
# Re-submit the same command after a timeout; completed stages are skipped.

#SBATCH --job-name=dgg_benchmark
#SBATCH --partition=gpu-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
# ARC GPU partitions currently have a 24-hour MaxWall limit. Leave a small
# margin below that limit; the stage markers make the same script resumable.
#SBATCH --time=23:55:00
#SBATCH --output=full_benchmark_%j.out
#SBATCH --error=full_benchmark_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -n "${DGG_PROJECT_ROOT:-${DGG_PROJECT_DIR:-}}" ]]; then
    PROJECT_DIR="${DGG_PROJECT_ROOT:-${DGG_PROJECT_DIR}}"
elif PROJECT_DIR="$(git -C "${SUBMIT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    echo "ERROR: submit from inside the DeepGreenGO checkout or set DGG_PROJECT_ROOT." >&2
    exit 2
fi
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
TUNING_ROOT="${DGG_TUNING_ROOT:-${PROJECT_DIR}/arc_tuning_cafa}"
# The CAFA tuning root contains the five-seed checkpoints. Graphs were built
# once in the shared project-level ARC tuning cache and reused by all runs.
GRAPH_ROOT="${DGG_GRAPH_ROOT:-${PROJECT_DIR}/arc_tuning/graphs_protbert}"
CHECKPOINT_ROOT="${DGG_CHECKPOINT_ROOT:-${TUNING_ROOT}/five_seed_hybrid}"
BENCHMARK_ROOT="${DGG_BENCHMARK_ROOT:-${PROJECT_DIR}/arc_benchmark/nominal_30_identity_80_coverage}"
# Baseline-only benchmark requested for the manuscript comparison. The search
# methods emit both top-10 and maximum-identity variants. InterProScan still
# runs internally when DPFunc needs its features, but is not itself evaluated.
# Override DGG_BENCHMARK_METHODS only for a deliberate diagnostic run.
METHODS="${DGG_BENCHMARK_METHODS:-naive,blast,diamond,foldseek,deepfri_sequence,deepfri_structure,dpfunc,heal}"
# Strict by default: a requested comparator must pass preflight before any
# expensive stage starts. Set to 1 only for an intentionally incomplete run.
SKIP_UNAVAILABLE="${DGG_BENCHMARK_SKIP_UNAVAILABLE:-0}"
FORCE_STAGES="${DGG_BENCHMARK_FORCE_STAGES:-}"
# Prediction-only mode. The 1000-bootstrap evaluation and every figure depend on
# all methods, so when several method jobs are chained they should be computed
# once at the end rather than repeated by each job.
SKIP_EVALUATION="${DGG_BENCHMARK_SKIP_EVALUATION:-0}"
# Methods whose prediction stage needs CUDA. The core preflight demands a
# visible GPU only when one of these is selected, so a CPU-only selection
# (DeepGraphGO, alignment baselines, or a pure evaluation pass) can be
# scheduled on a CPU partition instead of idling an L40. Set
# DGG_BENCHMARK_REQUIRE_GPU to 0 or 1 to override the inference.
GPU_METHODS="hybrid deepfri_sequence deepfri_structure transfun dpfunc deepgose heal gat_go"

BASE_ENV="${DGG_BASE_ENV:-deepgreengo}"
TF_ENV="${DGG_DEEPFRI_ENV:-dgg_sota_tf}"
TRANSFUN_ENV="${DGG_TRANSFUN_ENV:-dgg_transfun}"
DPFUNC_ENV="${DGG_DPFUNC_ENV:-dgg_dpfunc}"
DEEPGO_ENV="${DGG_DEEPGO_ENV:-dgg_deepgose}"
DEEPGOPLUS_ENV="${DGG_DEEPGOPLUS_ENV:-dgg_deepgoplus_py37}"
HEAL_ENV="${DGG_HEAL_ENV:-dgg_heal}"
GATGO_ENV="${DGG_GATGO_ENV:-dgg_gat_go}"
DEEPGRAPHGO_ENV="${DGG_DEEPGRAPHGO_ENV:-dgg_deepgraphgo}"
SETUP_TOOLS_ENV="${DGG_SETUP_TOOLS_ENV:-dgg_setup_tools}"
EGGNOG_ENV="${DGG_EGGNOG_ENV:-dgg_eggnog}"
# Keep Foldseek isolated from the trained-model environment by default. If it
# was already installed in BASE_ENV, preflight falls back to that environment.
FOLDSEEK_ENV="${DGG_FOLDSEEK_ENV:-dgg_foldseek}"
HAYAI_ENV="${DGG_HAYAI_ENV:-hayai_v3.2}"

DEEPFRI_ROOT="${DGG_DEEPFRI_ROOT:-${PROJECT_DIR}/baselines/DeepFRI}"
TRANSFUN_ROOT="${DGG_TRANSFUN_ROOT:-${PROJECT_DIR}/SOTA/TransFun}"
DPFUNC_ROOT="${DGG_DPFUNC_ROOT:-${PROJECT_DIR}/SOTA/DPFunc}"
DEEPGO_ROOT="${DGG_DEEPGO_ROOT:-${PROJECT_DIR}/SOTA/deepgo2}"
DEEPGOPLUS_ROOT="${DGG_DEEPGOPLUS_ROOT:-${PROJECT_DIR}/SOTA/deepgoplus}"
HEAL_ROOT="${DGG_HEAL_ROOT:-${PROJECT_DIR}/SOTA/HEAL}"
GATGO_ROOT="${DGG_GATGO_ROOT:-${PROJECT_DIR}/SOTA/GAT-GO}"
DEEPGRAPHGO_ROOT="${DGG_DEEPGRAPHGO_ROOT:-${PROJECT_DIR}/SOTA/DeepGraphGO}"
EGGNOG_ROOT="${DGG_EGGNOG_ROOT:-${PROJECT_DIR}/SOTA/eggnog-mapper}"
EGGNOG_DATA="${DGG_EGGNOG_DATA:-${EGGNOG_ROOT}/data}"
HAYAI_ROOT="${DGG_HAYAI_ROOT:-${PROJECT_DIR}/SOTA/HayaiAnnotation}"
INTERPROSCAN="${DGG_INTERPROSCAN:-${PROJECT_DIR}/SOTA/interproscan/interproscan.sh}"
INTERPRO_JAVA_ENV="${DGG_INTERPROSCAN_JAVA_ENV:-dgg_interproscan_java11}"
INTERPRO_JAVA_HOME="${DGG_INTERPROSCAN_JAVA_HOME:-}"
TORCH_HOME="${DGG_TORCH_HOME:-${PROJECT_DIR}/SOTA/torch_cache}"
export TORCH_HOME
HEAL_ESM1B_MODEL="${DGG_HEAL_ESM1B_MODEL:-${TORCH_HOME}/hub/checkpoints/esm1b_t33_650M_UR50S.pt}"
HEAL_MODEL_VARIANT="${DGG_HEAL_MODEL_VARIANT:-pdb-only}"
GATGO_FEATURE_ROOT="${DGG_GATGO_FEATURE_ROOT:-${GATGO_ROOT}/data/seq_features}"
GATGO_MODEL="${DGG_GATGO_MODEL:-${GATGO_ROOT}/trained_models/GAT-GO_modelweights.pt}"
GATGO_GO_MAP="${DGG_GATGO_GO_MAP:-${GATGO_ROOT}/data/data_splits/go2index.pt}"
GATGO_REVISION="${DGG_GATGO_REVISION:-90ec6d1067a893d4a51be715e41daf9fa4732952}"
DEEPGRAPHGO_REVISION="${DGG_DEEPGRAPHGO_REVISION:-efdb1cb9425f4f48e4613c0a89e603f5542bcb19}"
FOLDSEEK="${DGG_FOLDSEEK:-foldseek}"
DPFUNC_PRENAME="${DGG_DPFUNC_PRENAME:-DPFunc_model}"
DEEPGOPLUS_DIAMOND_DIR=""

THREADS="${SLURM_CPUS_PER_TASK:-16}"
STAGES="${BENCHMARK_ROOT}/stages"
RAW="${BENCHMARK_ROOT}/raw"
PYTHON_FILES=(
    "${PROJECT_DIR}/src/benchmark/cli.py"
    "${PROJECT_DIR}/src/benchmark/core.py"
    "${PROJECT_DIR}/src/benchmark/methods.py"
    "${PROJECT_DIR}/scripts/run_heal_arc.py"
    "${PROJECT_DIR}/scripts/run_gat_go_arc.py"
    "${PROJECT_DIR}/scripts/run_deepgraphgo_arc.py"
    "${PROJECT_DIR}/src/benchmark/external.py"
    "${PROJECT_DIR}/src/benchmark/evaluate.py"
)

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${BENCHMARK_ROOT}" "${STAGES}" "${RAW}"
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREADS}"

method_enabled() {
    [[ ",${METHODS}," == *",$1,"* ]]
}

gpu_required() {
    case "${DGG_BENCHMARK_REQUIRE_GPU:-auto}" in
        1) return 0 ;;
        0) return 1 ;;
    esac
    local method
    for method in ${GPU_METHODS}; do
        if method_enabled "${method}"; then
            return 0
        fi
    done
    return 1
}

conda_run() {
    local environment="$1"
    shift
    conda run --no-capture-output -n "${environment}" "$@"
}

benchmark_cli() {
    conda_run "${BASE_ENV}" python -m src.benchmark.cli "$@"
}

nonempty_file() {
    [[ -s "$1" ]]
}

prediction_complete() {
    local method="$1"
    local ontology
    for ontology in mf bp cc; do
        nonempty_file "${BENCHMARK_ROOT}/predictions/${method}/${ontology}.npz" || return 1
    done
}

best_hit_prediction_current() {
    local method="$1"
    local ontology
    local metadata
    for ontology in mf bp cc; do
        metadata="${BENCHMARK_ROOT}/predictions/${method}/${ontology}.metadata.json"
        nonempty_file "${metadata}" || return 1
        grep -q "one highest-identity.*top-k pool" "${metadata}" || return 1
    done
}

stage_artifacts_ok() {
    local name="$1"
    local ontology
    case "${name}" in
        01_prepare_inputs)
            nonempty_file "${BENCHMARK_ROOT}/benchmark_manifest.json" || return 1
            nonempty_file "${BENCHMARK_ROOT}/inputs/train.fasta" || return 1
            nonempty_file "${BENCHMARK_ROOT}/inputs/valid.fasta" || return 1
            nonempty_file "${BENCHMARK_ROOT}/inputs/test.fasta" || return 1
            nonempty_file "${BENCHMARK_ROOT}/inputs/structure_manifest.tsv" || return 1
            for ontology in mf bp cc; do
                nonempty_file "${BENCHMARK_ROOT}/labels/${ontology}_train.npz" || return 1
                nonempty_file "${BENCHMARK_ROOT}/labels/${ontology}_valid.npz" || return 1
                nonempty_file "${BENCHMARK_ROOT}/labels/${ontology}_test.npz" || return 1
            done
            ;;
        01b_structure_views)
            [[ "$(find "${BENCHMARK_ROOT}/inputs/structures/foldseek/train" -maxdepth 1 -type l | wc -l)" -eq 6026 ]] || return 1
            [[ "$(find "${BENCHMARK_ROOT}/inputs/structures/foldseek/valid_test" -maxdepth 1 -type l | wc -l)" -eq 1508 ]] || return 1
            [[ "$(find "${BENCHMARK_ROOT}/inputs/structures/valid_test" -maxdepth 1 -type l | wc -l)" -eq 1508 ]] || return 1
            ;;
        02_hybrid)
            prediction_complete deepgreengo || return 1
            ;;
        03_sequence_baselines_*)
            prediction_complete naive || return 1
            if method_enabled blast; then
                nonempty_file "${RAW}/blast_hits.tsv" || return 1
                prediction_complete blast || return 1
                prediction_complete blast_max || return 1
                best_hit_prediction_current blast_max || return 1
            fi
            if method_enabled diamond; then
                nonempty_file "${RAW}/diamond_hits.tsv" || return 1
                prediction_complete diamond || return 1
                prediction_complete diamond_max || return 1
                best_hit_prediction_current diamond_max || return 1
            fi
            ;;
        04_foldseek_unique_structures)
            nonempty_file "${RAW}/foldseek_hits.tsv" || return 1
            prediction_complete foldseek || return 1
            prediction_complete foldseek_max || return 1
            best_hit_prediction_current foldseek_max || return 1
            ;;
        05_interproscan)
            nonempty_file "${RAW}/interpro/interpro.tsv" || return 1
            if method_enabled interproscan; then prediction_complete interproscan || return 1; fi
            ;;
        05b_eggnog_mapper) prediction_complete eggnog_mapper || return 1 ;;
        06_deepfri_sequence) prediction_complete deepfri_sequence || return 1 ;;
        07_deepfri_structure) prediction_complete deepfri_structure || return 1 ;;
        08_transfun) prediction_complete transfun || return 1 ;;
        09_dpfunc) prediction_complete dpfunc || return 1 ;;
        09b_deepgoplus) prediction_complete deepgoplus || return 1 ;;
        10_deepgose) prediction_complete deepgose || return 1 ;;
        10b_heal) prediction_complete heal || return 1 ;;
        10c_gat_go) prediction_complete gat_go || return 1 ;;
        10d_deepgraphgo) prediction_complete deepgraphgo || return 1 ;;
        11_hayai) prediction_complete hayai || return 1 ;;
        12_gomap) prediction_complete gomap || return 1 ;;
    esac
}

require_file() {
    [[ -f "$1" ]] || { echo "[PREFLIGHT ERROR] Missing file: $1" >&2; return 1; }
}

require_dir() {
    [[ -d "$1" ]] || { echo "[PREFLIGHT ERROR] Missing directory: $1" >&2; return 1; }
}

require_git_revision() {
    local repository="$1"
    local expected="$2"
    local actual
    if command -v git >/dev/null 2>&1; then
        actual="$(command git -C "${repository}" rev-parse HEAD 2>/dev/null)"
    else
        actual="$(conda run -n "${SETUP_TOOLS_ENV}" git -C "${repository}" rev-parse HEAD 2>/dev/null | tail -n 1)"
    fi
    [[ -n "${actual}" ]] || {
        echo "[PREFLIGHT ERROR] Not a Git checkout: ${repository}" >&2
        return 1
    }
    [[ "${actual}" == "${expected}" ]] || {
        echo "[PREFLIGHT ERROR] ${repository} is at ${actual}, expected ${expected}" >&2
        return 1
    }
}

java_major() {
    local java_bin="$1"
    local version
    version="$("${java_bin}" -version 2>&1 | head -n 1)"
    if [[ "${version}" =~ \"1\.([0-9]+) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    elif [[ "${version}" =~ \"([0-9]+) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    else
        return 1
    fi
}

resolve_interpro_java() {
    local java_bin=""
    if [[ -n "${INTERPRO_JAVA_HOME}" ]]; then
        java_bin="${INTERPRO_JAVA_HOME}/bin/java"
    else
        java_bin="$(conda run -n "${INTERPRO_JAVA_ENV}" sh -c 'command -v java' 2>/dev/null | tail -n 1 || true)"
        if [[ -z "${java_bin}" ]]; then
            java_bin="$(command -v java 2>/dev/null || true)"
        fi
    fi
    if [[ ! -x "${java_bin}" ]]; then
        echo "[PREFLIGHT ERROR] Java 11 was not found. Run the SOTA setup job to create ${INTERPRO_JAVA_ENV}." >&2
        return 1
    fi
    local major
    major="$(java_major "${java_bin}")" || {
        echo "[PREFLIGHT ERROR] Could not parse Java version from ${java_bin}" >&2
        return 1
    }
    if [[ "${major}" != "11" ]]; then
        echo "[PREFLIGHT ERROR] InterProScan requires Java 11; ${java_bin} reports major version ${major}." >&2
        return 1
    fi
    INTERPRO_JAVA_HOME="$(cd "$(dirname "${java_bin}")/.." && pwd)"
    echo "InterProScan Java 11: ${INTERPRO_JAVA_HOME}/bin/java"
}

resolve_deepgoplus_diamond() {
    local diamond_bin
    diamond_bin="$(conda run -n "${BASE_ENV}" sh -c 'command -v diamond' 2>/dev/null | tail -n 1 || true)"
    if [[ ! -x "${diamond_bin}" ]]; then
        echo "[PREFLIGHT ERROR] DeepGOPlus needs the DIAMOND executable from ${BASE_ENV}." >&2
        return 1
    fi
    DEEPGOPLUS_DIAMOND_DIR="$(dirname "${diamond_bin}")"
}

PREFLIGHT_FAILURES=0
declare -A METHOD_FAILURES=()
CURRENT_METHOD="core"

# Attribute each failed check to the method being checked, so one unavailable
# comparator does not abort a run whose other comparators are fully installed.
# Anything checked while CURRENT_METHOD=core (interpreter, split data, GO file,
# base environment) is a genuine pipeline fault and always aborts.
preflight_check() {
    if ! "$@"; then
        ((PREFLIGHT_FAILURES += 1))
        METHOD_FAILURES["${CURRENT_METHOD}"]=$(( ${METHOD_FAILURES["${CURRENT_METHOD}"]:-0} + 1 ))
    fi
    return 0
}

note_failure() {
    ((PREFLIGHT_FAILURES += 1))
    METHOD_FAILURES["${CURRENT_METHOD}"]=$(( ${METHOD_FAILURES["${CURRENT_METHOD}"]:-0} + 1 ))
}

stage() {
    local name="$1"
    shift
    local marker="${STAGES}/${name}.done"
    if [[ -f "${marker}" ]] && [[ ",${FORCE_STAGES}," != *",${name},"* ]] \
            && stage_artifacts_ok "${name}"; then
        echo "[SKIP] ${name} already completed and required artifacts are present"
        return 0
    fi
    if [[ -f "${marker}" ]]; then
        echo "[RERUN] ${name}: forced or a required artifact is missing"
    fi
    echo "[START] ${name} $(date --iso-8601=seconds)"
    "$@"
    stage_artifacts_ok "${name}" || {
        echo "[STAGE ERROR] ${name} returned successfully but required artifacts are missing" >&2
        return 1
    }
    printf '%s\n' "completed $(date --iso-8601=seconds) job=${SLURM_JOB_ID}" > "${marker}"
    echo "[DONE] ${name}"
}

preflight() {
    PREFLIGHT_FAILURES=0
    local python_file
    for python_file in "${PYTHON_FILES[@]}"; do
        preflight_check require_file "${python_file}"
    done
    preflight_check require_dir "${DATA_ROOT}/datasets/threshold_30"
    preflight_check require_file "${DATA_ROOT}/go-basic.obo"
    preflight_check conda_run "${BASE_ENV}" python -c "import Bio,numpy,sklearn,torch; print('base environment ok')"
    if method_enabled hybrid; then
        CURRENT_METHOD=hybrid
        preflight_check require_dir "${CHECKPOINT_ROOT}"
        preflight_check require_dir "${GRAPH_ROOT}"
        preflight_check require_file "${GRAPH_ROOT}/2WSC_1.pt"
        local ontology seed
        for ontology in molecular_function biological_process cellular_component; do
            for seed in 1103 2207 3301 4409 5501; do
                preflight_check require_file "${CHECKPOINT_ROOT}/${ontology}/seed_${seed}/best_checkpoint.pt"
            done
        done
    fi
    if method_enabled blast; then
        CURRENT_METHOD=blast
        preflight_check conda_run "${BASE_ENV}" makeblastdb -version
        preflight_check conda_run "${BASE_ENV}" blastp -version
    fi
    if method_enabled diamond; then
        CURRENT_METHOD=diamond
        preflight_check conda_run "${BASE_ENV}" diamond version
    fi
    if method_enabled foldseek; then
        CURRENT_METHOD=foldseek
        local requested_foldseek="${FOLDSEEK}"
        local resolved_foldseek=""
        if [[ "${requested_foldseek}" == */* && -x "${requested_foldseek}" ]]; then
            resolved_foldseek="${requested_foldseek}"
        elif resolved_foldseek="$(conda run -n "${FOLDSEEK_ENV}" sh -c \
                'command -v "$1"' sh "${requested_foldseek}" 2>/dev/null)" \
                && [[ -n "${resolved_foldseek}" ]]; then
            :
        elif [[ "${FOLDSEEK_ENV}" != "${BASE_ENV}" ]] \
                && resolved_foldseek="$(conda run -n "${BASE_ENV}" sh -c \
                    'command -v "$1"' sh "${requested_foldseek}" 2>/dev/null)" \
                && [[ -n "${resolved_foldseek}" ]]; then
            echo "Foldseek found in fallback environment ${BASE_ENV}"
            FOLDSEEK_ENV="${BASE_ENV}"
        else
            echo "[PREFLIGHT ERROR] Foldseek was not found in ${FOLDSEEK_ENV} or ${BASE_ENV}." >&2
            echo "On the ARC login node, create it once with:" >&2
            echo "  conda create -y -n dgg_foldseek -c conda-forge -c bioconda foldseek" >&2
            echo "Then resubmit the unchanged Slurm command." >&2
            note_failure
        fi
        if [[ -n "${resolved_foldseek}" ]]; then
            FOLDSEEK="$(printf '%s\n' "${resolved_foldseek}" | tail -n 1)"
            preflight_check "${FOLDSEEK}" version
        fi
    fi
    if method_enabled interproscan; then
        CURRENT_METHOD=interproscan
        [[ -x "${INTERPROSCAN}" ]] || command -v "${INTERPROSCAN}" >/dev/null || {
            echo "[PREFLIGHT ERROR] Set DGG_INTERPROSCAN to the executable InterProScan script" >&2
            note_failure
        }
        preflight_check resolve_interpro_java
    fi
    if method_enabled dpfunc; then
        CURRENT_METHOD=dpfunc
        [[ -x "${INTERPROSCAN}" ]] || command -v "${INTERPROSCAN}" >/dev/null || {
            echo "[PREFLIGHT ERROR] DPFunc requires executable InterProScan at DGG_INTERPROSCAN" >&2
            note_failure
        }
        preflight_check resolve_interpro_java
    fi
    if method_enabled deepfri_sequence || method_enabled deepfri_structure; then
        CURRENT_METHOD=deepfri
        preflight_check require_file "${DEEPFRI_ROOT}/predict.py"
        preflight_check require_file "${DEEPFRI_ROOT}/trained_models/model_config.json"
        preflight_check conda_run "${TF_ENV}" python -c "import tensorflow; print('DeepFRI environment ok')"
        preflight_check conda_run "${TF_ENV}" env --chdir="${DEEPFRI_ROOT}" python predict.py --help
    fi
    if method_enabled transfun; then
        CURRENT_METHOD=transfun
        preflight_check require_file "${TRANSFUN_ROOT}/predict.py"
        preflight_check require_file "${TRANSFUN_ROOT}/data/molecular_function.pt"
        preflight_check require_file "${TRANSFUN_ROOT}/data/biological_process.pt"
        preflight_check require_file "${TRANSFUN_ROOT}/data/cellular_component.pt"
        preflight_check conda_run "${TRANSFUN_ENV}" python -c \
            "import biopandas, esm, torch, torch_geometric; print('TransFun environment ok')"
    fi
    if method_enabled dpfunc; then
        CURRENT_METHOD=dpfunc
        preflight_check require_file "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/build_data_demo.py"
        preflight_check require_file "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/run_dpfunc.py"
        preflight_check conda_run "${DPFUNC_ENV}" python "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/build_data_demo.py" --help
        preflight_check conda_run "${DPFUNC_ENV}" python "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/run_dpfunc.py" --help
        preflight_check require_file "${DPFUNC_ROOT}/data/inter_idx.pkl"
        preflight_check require_file "${TORCH_HOME}/hub/checkpoints/esm2_t33_650M_UR50D.pt"
        preflight_check require_file "${TORCH_HOME}/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt"
        local ontology model_index
        for ontology in mf bp cc; do
            preflight_check require_file "${DPFUNC_ROOT}/mlb/${ontology}_go.mlb"
            for model_index in 0 1 2; do
                preflight_check require_file "${DPFUNC_ROOT}/save_models/${DPFUNC_PRENAME}_${ontology}_${model_index}of3model.pt"
            done
        done
        preflight_check conda_run "${DPFUNC_ENV}" python -c \
            "import dgl, esm, logzero, torch; print('DPFunc environment and ESM import ok')"
    fi
    if method_enabled eggnog_mapper; then
        CURRENT_METHOD=eggnog_mapper
        preflight_check require_file "${EGGNOG_ROOT}/emapper.py"
        preflight_check require_dir "${EGGNOG_DATA}"
        preflight_check conda_run "${EGGNOG_ENV}" python "${EGGNOG_ROOT}/emapper.py" --version
    fi
    if method_enabled deepgoplus; then
        CURRENT_METHOD=deepgoplus
        preflight_check require_dir "${DEEPGOPLUS_ROOT}/data"
        preflight_check require_file "${DEEPGOPLUS_ROOT}/data/go.obo"
        preflight_check require_file "${DEEPGOPLUS_ROOT}/data/model.h5"
        preflight_check require_file "${DEEPGOPLUS_ROOT}/data/terms.pkl"
        preflight_check require_file "${DEEPGOPLUS_ROOT}/data/train_data.pkl"
        preflight_check require_file "${DEEPGOPLUS_ROOT}/data/train_data.dmnd"
        preflight_check require_file "${DEEPGOPLUS_ROOT}/data/metadata/last_release.json"
        preflight_check conda_run "${DEEPGOPLUS_ENV}" deepgoplus --help >/dev/null
        preflight_check resolve_deepgoplus_diamond
    fi
    if method_enabled deepgose; then
        CURRENT_METHOD=deepgose
        preflight_check require_file "${DEEPGO_ROOT}/predict.py"
        preflight_check require_file "${DEEPGO_ROOT}/data/go.obo"
        preflight_check require_file "${DEEPGO_ROOT}/data/go-plus.norm"
        local ontology model_index
        for ontology in mf bp cc; do
            preflight_check require_file "${DEEPGO_ROOT}/data/${ontology}/terms.pkl"
        done
        for model_index in 0 1 2 5 6 8; do preflight_check require_file "${DEEPGO_ROOT}/data/mf/deepgozero_esm_plus_${model_index}.th"; done
        for model_index in 2 5 6 7 8 9; do preflight_check require_file "${DEEPGO_ROOT}/data/bp/deepgozero_esm_plus_${model_index}.th"; done
        for model_index in 1 3 4 5 6 7; do preflight_check require_file "${DEEPGO_ROOT}/data/cc/deepgozero_esm_plus_${model_index}.th"; done
        preflight_check conda_run "${DEEPGO_ENV}" python -c "import dgl, esm, torch; print('DeepGO-SE environment ok')"
        preflight_check conda_run "${DEEPGO_ENV}" python "${DEEPGO_ROOT}/predict.py" --help
    fi
    if method_enabled heal; then
        CURRENT_METHOD=heal
        preflight_check require_file "${HEAL_ROOT}/network.py"
        preflight_check require_file "${HEAL_ROOT}/utils.py"
        preflight_check require_file "${HEAL_ROOT}/data/nrPDB-GO_2019.06.18_annot.tsv"
        local heal_suffix
        case "${HEAL_MODEL_VARIANT}" in
            pdb-only) heal_suffix=CL ;;
            *) echo "[PREFLIGHT ERROR] The benchmark method 'heal' is the PDB-only checkpoint variant" >&2; note_failure; heal_suffix=CL ;;
        esac
        for ontology in mf bp cc; do
            preflight_check require_file "${HEAL_ROOT}/model/model_${ontology}${heal_suffix}.pt"
        done
        preflight_check require_file "${HEAL_ESM1B_MODEL}"
        # esm.pretrained.load_model_and_alphabet_local() reads this sidecar
        # unconditionally; catch it here instead of after model construction.
        preflight_check require_file "${HEAL_ESM1B_MODEL%.pt}-contact-regression.pt"
        preflight_check conda_run "${HEAL_ENV}" python -c "import Bio, esm, numpy, torch, torch_geometric; print('HEAL environment ok')"
        preflight_check conda_run "${HEAL_ENV}" python "${PROJECT_DIR}/scripts/run_heal_arc.py" --help
    fi
    if method_enabled gat_go; then
        CURRENT_METHOD=gat_go
        preflight_check require_git_revision "${GATGO_ROOT}" "${GATGO_REVISION}"
        preflight_check require_file "${GATGO_ROOT}/src/GnnPF.py"
        preflight_check require_dir "${GATGO_FEATURE_ROOT}"
        preflight_check require_file "${GATGO_MODEL}"
        preflight_check require_file "${GATGO_GO_MAP}"
        preflight_check conda_run "${GATGO_ENV}" python -c \
            "import numpy, torch, torch_geometric; assert torch.cuda.is_available(); print('GAT-GO environment ok')"
        preflight_check conda_run "${GATGO_ENV}" python "${PROJECT_DIR}/scripts/run_gat_go_arc.py" --help
    fi
    if method_enabled deepgraphgo; then
        CURRENT_METHOD=deepgraphgo
        preflight_check require_git_revision "${DEEPGRAPHGO_ROOT}" "${DEEPGRAPHGO_REVISION}"
        preflight_check require_file "${DEEPGRAPHGO_ROOT}/main.py"
        preflight_check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_pid_list.txt"
        preflight_check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_interpro.npz"
        preflight_check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_dgl_top_100"
        preflight_check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_blastdb.pin"
        preflight_check require_file "${DEEPGRAPHGO_ROOT}/data/ppi_mat.npz"
        local deepgraph_ontology deepgraph_model_index
        for deepgraph_ontology in mf bp cc; do
            preflight_check require_file "${DEEPGRAPHGO_ROOT}/data/${deepgraph_ontology}_go.mlb"
            for deepgraph_model_index in 0 1 2; do
                preflight_check require_file "${DEEPGRAPHGO_ROOT}/models/DeepGraphGO-Model-${deepgraph_model_index}-${deepgraph_ontology}"
            done
        done
        preflight_check conda_run "${DEEPGRAPHGO_ENV}" psiblast -version
        preflight_check conda_run "${DEEPGRAPHGO_ENV}" python -c \
            "import dgl, numpy, scipy, sklearn, torch; assert not torch.cuda.is_available(); print('DeepGraphGO legacy CPU environment ok')"
        preflight_check conda_run "${DEEPGRAPHGO_ENV}" python "${PROJECT_DIR}/scripts/run_deepgraphgo_arc.py" --help
    fi
    if method_enabled hayai; then
        CURRENT_METHOD=hayai
        preflight_check require_file "${HAYAI_ROOT}/db/zen.dmnd"
        preflight_check conda_run "${HAYAI_ENV}" diamond version
        if [[ "${DGG_HAYAI_USE_ORTHOLOGER:-1}" == "1" ]]; then
            preflight_check conda_run "${HAYAI_ENV}" ODB-mapper --help >/dev/null
        fi
    fi
    if method_enabled gomap && [[ ! -f "${DGG_GOMAP_GAF:-}" && -z "${DGG_GOMAP_COMMAND:-}" ]]; then
        CURRENT_METHOD=gomap
        echo "[PREFLIGHT ERROR] GOMAP has a manual Argot2.5 step upstream." >&2
        echo "Set DGG_GOMAP_GAF to a completed GAF, or DGG_GOMAP_COMMAND to a site-specific command that writes ${RAW}/gomap_predictions.gaf" >&2
        note_failure
    fi
    CURRENT_METHOD=core
    if gpu_required; then
        preflight_check nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "No GPU-backed method selected; skipping the GPU core check."
    fi
    if ((PREFLIGHT_FAILURES == 0)); then
        echo "Preflight passed"
        return 0
    fi

    echo "" >&2
    echo "[PREFLIGHT] ${PREFLIGHT_FAILURES} check(s) failed, by method:" >&2
    local method
    for method in "${!METHOD_FAILURES[@]}"; do
        echo "  ${method}: ${METHOD_FAILURES[$method]} failed check(s)" >&2
    done

    # A core failure means the pipeline itself cannot run; no method selection
    # can rescue that, so it always aborts.
    if (( ${METHOD_FAILURES[core]:-0} > 0 )); then
        echo "[PREFLIGHT FAILED] ${METHOD_FAILURES[core]} core check(s) failed (interpreter, split data," >&2
        echo "GO file, base environment or GPU). Fix these before rerunning." >&2
        return 1
    fi

    if [[ "${SKIP_UNAVAILABLE}" != "1" ]]; then
        echo "[PREFLIGHT FAILED] Every requested method must be installed because" >&2
        echo "DGG_BENCHMARK_SKIP_UNAVAILABLE=0. Set it to 1 to benchmark the available" >&2
        echo "subset instead, or narrow DGG_BENCHMARK_METHODS." >&2
        return 1
    fi

    # Drop only the methods that failed; keep the rest.
    local kept=() dropped=()
    local candidate
    IFS=',' read -r -a candidate <<< "${METHODS}"
    for method in "${candidate[@]}"; do
        local key="${method}"
        case "${method}" in
            deepfri_sequence|deepfri_structure) key=deepfri ;;
        esac
        if (( ${METHOD_FAILURES[$key]:-0} > 0 )); then
            dropped+=("${method}")
        else
            kept+=("${method}")
        fi
    done

    if ((${#kept[@]} == 0)); then
        echo "[PREFLIGHT FAILED] No requested method passed its dependency checks." >&2
        return 1
    fi

    METHODS="$(IFS=','; echo "${kept[*]}")"
    echo "" >&2
    echo "[PREFLIGHT] Continuing with the available subset." >&2
    echo "  DROPPED (not installed / not configured): ${dropped[*]}" >&2
    echo "  RUNNING: ${METHODS}" >&2
    echo "  These results describe only the running set. Any comparison written up" >&2
    echo "  from this run must state which methods were absent." >&2
    mkdir -p "${BENCHMARK_ROOT}"
    printf '%s\n' "${dropped[@]}" > "${BENCHMARK_ROOT}/preflight_dropped_methods.txt"
    echo "  Recorded in ${BENCHMARK_ROOT}/preflight_dropped_methods.txt" >&2
    echo "" >&2
    return 0
}

prepare_inputs() {
    benchmark_cli prepare \
        --data-root "${DATA_ROOT}" --workspace "${BENCHMARK_ROOT}" \
        --expected-test-size 754 --extract-structures
}

prepare_structure_views() {
    local structure_root="${BENCHMARK_ROOT}/inputs/structures"
    local foldseek_root="${structure_root}/foldseek"
    local query_view="${foldseek_root}/valid_test"
    local train_view="${foldseek_root}/train"
    local deepfri_view="${structure_root}/valid_test"
    mkdir -p "${query_view}" "${train_view}" "${deepfri_view}"

    # These directories contain generated symlinks only. Rebuild them from the
    # plain PDB files so .pdb/.pdb.gz duplicates cannot crowd Foldseek's top-k
    # hits or make DeepFRI process every protein twice.
    find "${query_view}" -mindepth 1 -maxdepth 1 -type l -delete
    find "${train_view}" -mindepth 1 -maxdepth 1 -type l -delete
    find "${deepfri_view}" -mindepth 1 -maxdepth 1 -type l -delete

    local structure
    for structure in "${structure_root}"/train/*.pdb; do
        ln -sfn "${structure}" "${train_view}/$(basename "${structure}")"
    done
    for structure in "${structure_root}"/valid/*.pdb "${structure_root}"/test/*.pdb; do
        ln -sfn "${structure}" "${query_view}/$(basename "${structure}")"
        ln -sfn "${structure}" "${deepfri_view}/$(basename "${structure}")"
    done
}

run_hybrid() {
    benchmark_cli export-hybrid \
        --data-root "${DATA_ROOT}" --workspace "${BENCHMARK_ROOT}" \
        --checkpoint-root "${CHECKPOINT_ROOT}" --graph-root "${GRAPH_ROOT}" \
        --batch-size 16 --workers 0 --require-cuda
}

run_sequence_baselines() {
    local selected=""
    method_enabled blast && selected="blast"
    method_enabled diamond && selected="${selected:+${selected},}diamond"
    benchmark_cli sequence-baselines \
        --workspace "${BENCHMARK_ROOT}" --methods "${selected}" --threads "${THREADS}" \
        --top-k 10 --max-hits 50 --min-qcov 0.5 --min-tcov 0.5
}

run_foldseek() {
    local force_search=()
    [[ "${DGG_FORCE_FOLDSEEK_SEARCH:-0}" == "1" ]] && force_search=(--force-search)
    benchmark_cli foldseek \
        --workspace "${BENCHMARK_ROOT}" --foldseek "${FOLDSEEK}" --threads "${THREADS}" \
        --top-k 10 --max-hits 50 --min-qcov 0.5 --min-tcov 0.5 "${force_search[@]}"
}

run_interpro() {
    local output="${RAW}/interpro/interpro.tsv"
    local partial="${output}.partial"
    mkdir -p "${RAW}/interpro"
    if [[ -s "${output}" ]]; then
        echo "[benchmark] reusing complete InterProScan output ${output}"
    else
        rm -f "${partial}"
        env JAVA_HOME="${INTERPRO_JAVA_HOME}" \
            PATH="${INTERPRO_JAVA_HOME}/bin:${PATH}" \
            "${INTERPROSCAN}" -i "${BENCHMARK_ROOT}/inputs/valid_test.fasta" \
            -f TSV -o "${partial}" -goterms -iprlookup -cpu "${THREADS}" -dp
        nonempty_file "${partial}" || {
            echo "[STAGE ERROR] InterProScan did not produce ${partial}" >&2
            return 1
        }
        mv "${partial}" "${output}"
    fi
    benchmark_cli normalize-interpro \
        --workspace "${BENCHMARK_ROOT}" --input "${output}"
}

run_eggnog() {
    mkdir -p "${RAW}/eggnog"
    conda_run "${EGGNOG_ENV}" python "${EGGNOG_ROOT}/emapper.py" \
        -i "${BENCHMARK_ROOT}/inputs/valid_test.fasta" --itype proteins \
        -m diamond --data_dir "${EGGNOG_DATA}" --cpu "${THREADS}" \
        -o dgg_benchmark --output_dir "${RAW}/eggnog" \
        --go_evidence all --override
    benchmark_cli normalize-eggnog \
        --workspace "${BENCHMARK_ROOT}" \
        --input "${RAW}/eggnog/dgg_benchmark.emapper.annotations"
}

run_deepgoplus() {
    local output="${RAW}/deepgoplus/predictions.tsv"
    local partial="${output}.partial"
    mkdir -p "${RAW}/deepgoplus"
    if [[ -s "${output}" ]]; then
        echo "[benchmark] reusing complete DeepGOPlus output ${output}"
    else
        rm -f "${partial}"
        PATH="${DEEPGOPLUS_DIAMOND_DIR}:${PATH}" conda_run "${DEEPGOPLUS_ENV}" deepgoplus \
            --data-root "${DEEPGOPLUS_ROOT}/data" \
            --in-file "${BENCHMARK_ROOT}/inputs/valid_test.fasta" \
            --out-file "${partial}" --threshold 0
        nonempty_file "${partial}" || {
            echo "[STAGE ERROR] DeepGOPlus did not produce ${partial}" >&2
            return 1
        }
        mv "${partial}" "${output}"
    fi
    benchmark_cli normalize-deepgoplus \
        --workspace "${BENCHMARK_ROOT}" --input "${output}"
}

run_deepfri_sequence() {
    mkdir -p "${RAW}/deepfri"
    local ont output
    for ont in mf bp cc; do
        output="${RAW}/deepfri/sequence_${ont^^}_pred_scores.json"
        if conda_run "${BASE_ENV}" python -m json.tool "${output}" >/dev/null 2>&1; then
            echo "[benchmark] reusing complete DeepFRI sequence output ${output}"
            continue
        fi
        rm -f "${output}"
        conda_run "${TF_ENV}" env --chdir="${DEEPFRI_ROOT}" python predict.py \
            --fasta_fn "${BENCHMARK_ROOT}/inputs/valid_test.fasta" -ont "${ont}" \
            -o "${RAW}/deepfri/sequence"
        conda_run "${BASE_ENV}" python -m json.tool "${output}" >/dev/null
    done
    benchmark_cli normalize-deepfri \
        --workspace "${BENCHMARK_ROOT}" --method deepfri_sequence \
        --mf "${RAW}/deepfri/sequence_MF_pred_scores.json" \
        --bp "${RAW}/deepfri/sequence_BP_pred_scores.json" \
        --cc "${RAW}/deepfri/sequence_CC_pred_scores.json"
}

run_deepfri_structure() {
    mkdir -p "${RAW}/deepfri"
    local ont output
    for ont in mf bp cc; do
        output="${RAW}/deepfri/structure_${ont^^}_pred_scores.json"
        if conda_run "${BASE_ENV}" python -m json.tool "${output}" >/dev/null 2>&1; then
            echo "[benchmark] reusing complete DeepFRI structure output ${output}"
            continue
        fi
        rm -f "${output}"
        conda_run "${TF_ENV}" env --chdir="${DEEPFRI_ROOT}" python predict.py \
            --pdb_dir "${BENCHMARK_ROOT}/inputs/structures/valid_test" -ont "${ont}" \
            -o "${RAW}/deepfri/structure"
        conda_run "${BASE_ENV}" python -m json.tool "${output}" >/dev/null
    done
    benchmark_cli normalize-deepfri \
        --workspace "${BENCHMARK_ROOT}" --method deepfri_structure \
        --mf "${RAW}/deepfri/structure_MF_pred_scores.json" \
        --bp "${RAW}/deepfri/structure_BP_pred_scores.json" \
        --cc "${RAW}/deepfri/structure_CC_pred_scores.json"
}

run_transfun() {
    local data_dir="${TRANSFUN_ROOT}/data"
    ln -sfn "${BENCHMARK_ROOT}/inputs/structures/valid_test" "${data_dir}/dgg_benchmark_pdbs"
    cp "${BENCHMARK_ROOT}/inputs/valid_test.fasta" "${data_dir}/dgg_benchmark.fasta"
    local short full
    for short in mf bp cc; do
        case "${short}" in
            mf) full=molecular_function ;;
            bp) full=biological_process ;;
            cc) full=cellular_component ;;
        esac
        conda_run "${TRANSFUN_ENV}" python "${TRANSFUN_ROOT}/predict.py" \
            --data-path "${data_dir}" --ontology "${full}" --input-type fasta \
            --fasta-path dgg_benchmark.fasta --pdb-path dgg_benchmark_pdbs \
            --cut-off 0 --output "dgg_benchmark_${short}.txt"
    done
    benchmark_cli normalize-scored \
        --workspace "${BENCHMARK_ROOT}" --method transfun \
        --mf "${data_dir}/dgg_benchmark_mf.txt" --bp "${data_dir}/dgg_benchmark_bp.txt" \
        --cc "${data_dir}/dgg_benchmark_cc.txt"
}

run_dpfunc() {
    require_file "${RAW}/interpro/interpro.tsv"
    mkdir -p "${RAW}/dpfunc"
    benchmark_cli make-dpfunc-manifest \
        --workspace "${BENCHMARK_ROOT}" --interpro-tsv "${RAW}/interpro/interpro.tsv" \
        --output "${RAW}/dpfunc/manifest.tsv"

    local source_workspace="${RAW}/dpfunc/mf"
    conda_run "${DPFUNC_ENV}" python "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/build_data_demo.py" \
        --manifest "${RAW}/dpfunc/manifest.tsv" --workspace "${source_workspace}" \
        --ontology mf --pdb-dir "${BENCHMARK_ROOT}/inputs/structures/valid_test" \
        --inter-idx "${DPFUNC_ROOT}/data/inter_idx.pkl" --go-obo "${DATA_ROOT}/go-basic.obo" \
        --mlb-source "${DPFUNC_ROOT}/mlb/mf_go.mlb" --esm-batch-size 4 --device cuda

    local short dp_workspace result
    for short in mf bp cc; do
        dp_workspace="${RAW}/dpfunc/${short}"
        result="${dp_workspace}/results/${DPFUNC_PRENAME}_${short}_final.pkl"
        if [[ -s "${result}" ]]; then
            echo "[benchmark] reusing complete DPFunc output ${result}"
            continue
        fi
        if [[ "${short}" != "mf" ]]; then
            benchmark_cli clone-dpfunc-workspace \
                --source "${source_workspace}" --target "${dp_workspace}" \
                --source-ontology mf --target-ontology "${short}" \
                --mlb-source "${DPFUNC_ROOT}/mlb/${short}_go.mlb"
        fi
        cp "${DPFUNC_ROOT}"/save_models/"${DPFUNC_PRENAME}_${short}_"*of3model.pt "${dp_workspace}/save_models/"
        conda_run "${DPFUNC_ENV}" python "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/run_dpfunc.py" predict \
            --workspace "${dp_workspace}" --ontology "${short}" --gpu 0 --pre-name "${DPFUNC_PRENAME}"
        nonempty_file "${result}" || return 1
    done
    benchmark_cli normalize-dpfunc \
        --workspace "${BENCHMARK_ROOT}" --method dpfunc \
        --mf "${RAW}/dpfunc/mf/results/${DPFUNC_PRENAME}_mf_final.pkl" \
        --bp "${RAW}/dpfunc/bp/results/${DPFUNC_PRENAME}_bp_final.pkl" \
        --cc "${RAW}/dpfunc/cc/results/${DPFUNC_PRENAME}_cc_final.pkl"
}

run_deepgose() {
    mkdir -p "${RAW}/deepgose"
    cp "${BENCHMARK_ROOT}/inputs/valid_test.fasta" "${RAW}/deepgose/queries.fasta"
    if gzip -t "${RAW}/deepgose/queries_preds_mf.tsv.gz" \
            "${RAW}/deepgose/queries_preds_bp.tsv.gz" \
            "${RAW}/deepgose/queries_preds_cc.tsv.gz" 2>/dev/null; then
        echo "[benchmark] reusing complete DeepGO-SE outputs"
    else
        rm -f "${RAW}/deepgose/queries_preds_mf.tsv.gz" \
            "${RAW}/deepgose/queries_preds_bp.tsv.gz" \
            "${RAW}/deepgose/queries_preds_cc.tsv.gz"
        conda_run "${DEEPGO_ENV}" python "${DEEPGO_ROOT}/predict.py" \
            -if "${RAW}/deepgose/queries.fasta" -dr "${DEEPGO_ROOT}/data" \
            -t 0 -bs 8 -d cuda
        gzip -t "${RAW}/deepgose/queries_preds_mf.tsv.gz" \
            "${RAW}/deepgose/queries_preds_bp.tsv.gz" \
            "${RAW}/deepgose/queries_preds_cc.tsv.gz"
    fi
    benchmark_cli normalize-scored \
        --workspace "${BENCHMARK_ROOT}" --method deepgose --delimiter tab \
        --mf "${RAW}/deepgose/queries_preds_mf.tsv.gz" \
        --bp "${RAW}/deepgose/queries_preds_bp.tsv.gz" \
        --cc "${RAW}/deepgose/queries_preds_cc.tsv.gz"
}

run_heal() {
    mkdir -p "${RAW}/heal"
    conda_run "${HEAL_ENV}" python "${PROJECT_DIR}/scripts/run_heal_arc.py" --heal-root "${HEAL_ROOT}" --structure-dir "${BENCHMARK_ROOT}/inputs/structures/valid_test" --esm1b-model "${HEAL_ESM1B_MODEL}" --output-dir "${RAW}/heal" --device cuda:0 --model-variant "${HEAL_MODEL_VARIANT}" --expected-structures 1508
    benchmark_cli normalize-scored --workspace "${BENCHMARK_ROOT}" --method heal --delimiter tab --mf "${RAW}/heal/heal_mf.tsv" --bp "${RAW}/heal/heal_bp.tsv" --cc "${RAW}/heal/heal_cc.tsv"
}

run_gat_go() {
    mkdir -p "${RAW}/gat_go"
    conda_run "${GATGO_ENV}" python "${PROJECT_DIR}/scripts/run_gat_go_arc.py" \
        --gat-root "${GATGO_ROOT}" --feature-root "${GATGO_FEATURE_ROOT}" \
        --model "${GATGO_MODEL}" --go-map "${GATGO_GO_MAP}" \
        --fasta "${BENCHMARK_ROOT}/inputs/valid_test.fasta" \
        --workspace "${BENCHMARK_ROOT}" --output-dir "${RAW}/gat_go" \
        --expected-proteins 1508 --device cuda:0
    benchmark_cli normalize-scored --workspace "${BENCHMARK_ROOT}" \
        --method gat_go --delimiter tab \
        --mf "${RAW}/gat_go/gat_go_mf.tsv" \
        --bp "${RAW}/gat_go/gat_go_bp.tsv" \
        --cc "${RAW}/gat_go/gat_go_cc.tsv"
}

run_deepgraphgo() {
    mkdir -p "${RAW}/deepgraphgo"
    conda_run "${DEEPGRAPHGO_ENV}" env DGG_DEEPGRAPHGO_THREADS="${THREADS}" \
        python "${PROJECT_DIR}/scripts/run_deepgraphgo_arc.py" \
        --deepgraphgo-root "${DEEPGRAPHGO_ROOT}" \
        --fasta "${BENCHMARK_ROOT}/inputs/valid_test.fasta" \
        --workspace "${BENCHMARK_ROOT}" --output-dir "${RAW}/deepgraphgo" \
        --expected-proteins 1508
    benchmark_cli normalize-scored --workspace "${BENCHMARK_ROOT}" \
        --method deepgraphgo --delimiter tab \
        --mf "${RAW}/deepgraphgo/deepgraphgo_mf.tsv" \
        --bp "${RAW}/deepgraphgo/deepgraphgo_bp.tsv" \
        --cc "${RAW}/deepgraphgo/deepgraphgo_cc.tsv"
}

run_hayai() {
    local orthologer=()
    [[ "${DGG_HAYAI_USE_ORTHOLOGER:-1}" == "1" ]] && orthologer=(--orthologer)
    conda_run "${HAYAI_ENV}" python -m src.benchmark.cli run-hayai \
        --workspace "${BENCHMARK_ROOT}" --hayai-root "${HAYAI_ROOT}" \
        --output "${RAW}/hayai_annotations.tsv" --threads "${THREADS}" "${orthologer[@]}"
    benchmark_cli normalize-hayai \
        --workspace "${BENCHMARK_ROOT}" --input "${RAW}/hayai_annotations.tsv"
}

run_gomap() {
    local output="${DGG_GOMAP_GAF:-${RAW}/gomap_predictions.gaf}"
    if [[ ! -f "${output}" ]]; then
        bash -lc "${DGG_GOMAP_COMMAND}"
    fi
    require_file "${output}"
    benchmark_cli normalize-gomap \
        --workspace "${BENCHMARK_ROOT}" --input "${output}"
}

run_evaluation() {
    # Evaluate only this run's selected methods, so stale prediction folders
    # from older, broader benchmarks cannot leak into results/*.csv. Search
    # baselines also emit a separately reported maximum-identity variant.
    local selected="${METHODS//hybrid/deepgreengo}"
    method_enabled blast && selected="${selected},blast_max"
    method_enabled diamond && selected="${selected},diamond_max"
    method_enabled foldseek && selected="${selected},foldseek_max"
    # Uses ${x-default}, not ${x:-default}, so an intentionally empty required
    # list is honoured while --methods still locks the evaluated result set.
    local required="${DGG_BENCHMARK_REQUIRE_METHODS-${selected}}"
    benchmark_cli evaluate \
        --workspace "${BENCHMARK_ROOT}" --bootstraps 1000 --aupr-workers "${THREADS}" \
        --methods "${selected}" --require-methods "${required}"
    benchmark_cli plot --workspace "${BENCHMARK_ROOT}"
}

run_stratified_analysis() {
    # Homology / term-IC / GO-depth stratification and IC-weighted PR curves,
    # computed from the stored per-protein score matrices. Depends only on
    # predictions/ and labels/, so it runs after every method has been
    # normalized but needs no GPU.
    conda_run "${BASE_ENV}" python -m src.benchmark.run_stratified \
        --workspace "${BENCHMARK_ROOT}" --data-root "${DATA_ROOT}"
    conda_run "${BASE_ENV}" python "${PROJECT_DIR}/src/plot_stratified.py" \
        --workspace "${BENCHMARK_ROOT}"
    # plots/supplementary_tables was previously only regenerated by the
    # separate run_supplementary_analysis.slurm, so the tables silently kept
    # an older method set than plots/main_comparison and plots/stratified.
    # It reads the same stratified tables just written above, needs no GPU,
    # and takes seconds - so build all three together and keep them in sync.
    conda_run "${BASE_ENV}" python "${PROJECT_DIR}/src/export_supplementary_tables.py" \
        --workspace "${BENCHMARK_ROOT}"
}

run_comparison_figures() {
    # Keep the workspace comparison bundle used by the supplementary pipeline,
    # then write the locked baseline-only manuscript bundle to plots/figures.
    conda_run "${BASE_ENV}" python "${PROJECT_DIR}/src/plot_baselines_only.py" \
        --workspace "${BENCHMARK_ROOT}"
    conda_run "${BASE_ENV}" python "${PROJECT_DIR}/src/plot_benchmark.py" \
        --workspace "${BENCHMARK_ROOT}" \
        --output-dir "${PROJECT_DIR}/plots/figures/benchmark"
}

echo "Project: ${PROJECT_DIR}"
echo "Benchmark: ${BENCHMARK_ROOT}"
echo "Methods: ${METHODS}"
preflight
stage 01_prepare_inputs prepare_inputs
stage 01b_structure_views prepare_structure_views
method_enabled hybrid && stage 02_hybrid run_hybrid
SEQUENCE_TAG="naive"
method_enabled blast && SEQUENCE_TAG="${SEQUENCE_TAG}_blast"
method_enabled diamond && SEQUENCE_TAG="${SEQUENCE_TAG}_diamond"
stage "03_sequence_baselines_${SEQUENCE_TAG}" run_sequence_baselines
method_enabled foldseek && stage 04_foldseek_unique_structures run_foldseek
if method_enabled interproscan || method_enabled dpfunc; then stage 05_interproscan run_interpro; fi
method_enabled eggnog_mapper && stage 05b_eggnog_mapper run_eggnog
method_enabled deepfri_sequence && stage 06_deepfri_sequence run_deepfri_sequence
method_enabled deepfri_structure && stage 07_deepfri_structure run_deepfri_structure
method_enabled transfun && stage 08_transfun run_transfun
method_enabled dpfunc && stage 09_dpfunc run_dpfunc
method_enabled deepgoplus && stage 09b_deepgoplus run_deepgoplus
method_enabled deepgose && stage 10_deepgose run_deepgose
method_enabled heal && stage 10b_heal run_heal
method_enabled gat_go && stage 10c_gat_go run_gat_go
method_enabled deepgraphgo && stage 10d_deepgraphgo run_deepgraphgo
method_enabled hayai && stage 11_hayai run_hayai
method_enabled gomap && stage 12_gomap run_gomap
if [[ "${SKIP_EVALUATION}" == "1" ]]; then
    echo "Predictions complete. Evaluation and figures skipped (DGG_BENCHMARK_SKIP_EVALUATION=1);"
    echo "run this script once more without that flag to score every method together."
else
    run_evaluation
    run_comparison_figures
    run_stratified_analysis
fi
echo "Benchmark completed: ${BENCHMARK_ROOT}"
