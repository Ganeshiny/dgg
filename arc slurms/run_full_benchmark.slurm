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
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/full_benchmark_%j.out
#SBATCH --error=/home/ganeshiny.sridharan/dgg/deep-green-GO/logs/full_benchmark_%j.err

set -euo pipefail

PROJECT_DIR="${DGG_PROJECT_ROOT:-/home/ganeshiny.sridharan/dgg/deep-green-GO}"
DATA_ROOT="${DGG_DATA_ROOT:-${PROJECT_DIR}/preprocessing/data_arc_rebuild_2026_07_14}"
TUNING_ROOT="${DGG_TUNING_ROOT:-${PROJECT_DIR}/arc_tuning_cafa}"
GRAPH_ROOT="${DGG_GRAPH_ROOT:-${TUNING_ROOT}/graphs_protbert}"
CHECKPOINT_ROOT="${DGG_CHECKPOINT_ROOT:-${TUNING_ROOT}/five_seed_hybrid}"
BENCHMARK_ROOT="${DGG_BENCHMARK_ROOT:-${PROJECT_DIR}/arc_benchmark/nominal_30_identity_80_coverage}"
METHODS="${DGG_BENCHMARK_METHODS:-hybrid,naive,blast,diamond,foldseek,interproscan,eggnog_mapper,deepfri_sequence,deepfri_structure,transfun,dpfunc,deepgoplus,deepgose,hayai,gomap}"

BASE_ENV="${DGG_BASE_ENV:-deepgreengo}"
TORCH_ENV="${DGG_SOTA_TORCH_ENV:-dgg_sota_torch}"
TF_ENV="${DGG_DEEPFRI_ENV:-dgg_sota_tf}"
DEEPGO_ENV="${DGG_DEEPGO_ENV:-dgg_deepgose}"
DEEPGOPLUS_ENV="${DGG_DEEPGOPLUS_ENV:-dgg_deepgoplus}"
EGGNOG_ENV="${DGG_EGGNOG_ENV:-dgg_eggnog}"
HAYAI_ENV="${DGG_HAYAI_ENV:-hayai_v3.2}"

DEEPFRI_ROOT="${DGG_DEEPFRI_ROOT:-${PROJECT_DIR}/baselines/DeepFRI}"
TRANSFUN_ROOT="${DGG_TRANSFUN_ROOT:-${PROJECT_DIR}/SOTA/TransFun}"
DPFUNC_ROOT="${DGG_DPFUNC_ROOT:-${PROJECT_DIR}/SOTA/DPFunc}"
DEEPGO_ROOT="${DGG_DEEPGO_ROOT:-${PROJECT_DIR}/SOTA/deepgo2}"
DEEPGOPLUS_ROOT="${DGG_DEEPGOPLUS_ROOT:-${PROJECT_DIR}/SOTA/deepgoplus}"
EGGNOG_ROOT="${DGG_EGGNOG_ROOT:-${PROJECT_DIR}/SOTA/eggnog-mapper}"
EGGNOG_DATA="${DGG_EGGNOG_DATA:-${EGGNOG_ROOT}/data}"
HAYAI_ROOT="${DGG_HAYAI_ROOT:-${PROJECT_DIR}/SOTA/HayaiAnnotation}"
INTERPROSCAN="${DGG_INTERPROSCAN:-interproscan.sh}"
FOLDSEEK="${DGG_FOLDSEEK:-foldseek}"
DPFUNC_PRENAME="${DGG_DPFUNC_PRENAME:-DPFunc_model}"

THREADS="${SLURM_CPUS_PER_TASK:-16}"
STAGES="${BENCHMARK_ROOT}/stages"
RAW="${BENCHMARK_ROOT}/raw"
PYTHON_SCRIPT="${PROJECT_DIR}/scripts/run_arc_benchmark.py"

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

conda_run() {
    local environment="$1"
    shift
    conda run --no-capture-output -n "${environment}" "$@"
}

require_file() {
    [[ -f "$1" ]] || { echo "[PREFLIGHT ERROR] Missing file: $1" >&2; return 1; }
}

require_dir() {
    [[ -d "$1" ]] || { echo "[PREFLIGHT ERROR] Missing directory: $1" >&2; return 1; }
}

stage() {
    local name="$1"
    shift
    local marker="${STAGES}/${name}.done"
    if [[ -f "${marker}" ]]; then
        echo "[SKIP] ${name} already completed"
        return 0
    fi
    echo "[START] ${name} $(date --iso-8601=seconds)"
    "$@"
    printf '%s\n' "completed $(date --iso-8601=seconds) job=${SLURM_JOB_ID}" > "${marker}"
    echo "[DONE] ${name}"
}

preflight() {
    require_file "${PYTHON_SCRIPT}"
    require_dir "${DATA_ROOT}/datasets/threshold_30"
    require_file "${DATA_ROOT}/go-basic.obo"
    conda_run "${BASE_ENV}" python -c "import Bio,numpy,sklearn,torch; print('base environment ok')"
    if method_enabled hybrid; then
        require_dir "${CHECKPOINT_ROOT}"
        require_dir "${GRAPH_ROOT}"
        local ontology seed
        for ontology in molecular_function biological_process cellular_component; do
            for seed in 1103 2207 3301 4409 5501; do
                require_file "${CHECKPOINT_ROOT}/${ontology}/seed_${seed}/best_checkpoint.pt"
            done
        done
    fi
    if method_enabled blast; then
        conda_run "${BASE_ENV}" makeblastdb -version
        conda_run "${BASE_ENV}" blastp -version
    fi
    if method_enabled diamond; then
        conda_run "${BASE_ENV}" diamond version
    fi
    if method_enabled foldseek; then
        conda_run "${BASE_ENV}" "${FOLDSEEK}" version
    fi
    if method_enabled interproscan || method_enabled dpfunc; then
        [[ -x "${INTERPROSCAN}" ]] || command -v "${INTERPROSCAN}" >/dev/null || {
            echo "[PREFLIGHT ERROR] Set DGG_INTERPROSCAN to the executable InterProScan script" >&2
            return 1
        }
    fi
    if method_enabled deepfri_sequence || method_enabled deepfri_structure; then
        require_file "${DEEPFRI_ROOT}/predict.py"
        require_file "${DEEPFRI_ROOT}/trained_models/model_config.json"
        conda_run "${TF_ENV}" python -c "import tensorflow; print('DeepFRI environment ok')"
    fi
    if method_enabled transfun; then
        require_file "${TRANSFUN_ROOT}/predict.py"
        require_file "${TRANSFUN_ROOT}/data/molecular_function.pt"
        require_file "${TRANSFUN_ROOT}/data/biological_process.pt"
        require_file "${TRANSFUN_ROOT}/data/cellular_component.pt"
    fi
    if method_enabled dpfunc; then
        require_file "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/build_data_demo.py"
        require_file "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/run_dpfunc.py"
        require_file "${DPFUNC_ROOT}/data/inter_idx.pkl"
        local ontology model_index
        for ontology in mf bp cc; do
            require_file "${DPFUNC_ROOT}/mlb/${ontology}_go.mlb"
            for model_index in 0 1 2; do
                require_file "${DPFUNC_ROOT}/save_models/${DPFUNC_PRENAME}_${ontology}_${model_index}of3model.pt"
            done
        done
    fi
    if method_enabled eggnog_mapper; then
        require_file "${EGGNOG_ROOT}/emapper.py"
        require_dir "${EGGNOG_DATA}"
        conda_run "${EGGNOG_ENV}" python "${EGGNOG_ROOT}/emapper.py" --version
    fi
    if method_enabled deepgoplus; then
        require_dir "${DEEPGOPLUS_ROOT}/data"
        require_file "${DEEPGOPLUS_ROOT}/data/go.obo"
        require_file "${DEEPGOPLUS_ROOT}/data/model.h5"
        require_file "${DEEPGOPLUS_ROOT}/data/terms.pkl"
        require_file "${DEEPGOPLUS_ROOT}/data/train_data.pkl"
        conda_run "${DEEPGOPLUS_ENV}" deepgoplus --help >/dev/null
    fi
    if method_enabled deepgose; then
        require_file "${DEEPGO_ROOT}/predict.py"
        require_file "${DEEPGO_ROOT}/data/go.obo"
        require_file "${DEEPGO_ROOT}/data/go-plus.norm"
        local ontology model_index
        for ontology in mf bp cc; do
            require_file "${DEEPGO_ROOT}/data/${ontology}/terms.pkl"
        done
        for model_index in 0 1 2 5 6 8; do require_file "${DEEPGO_ROOT}/data/mf/deepgozero_esm_plus_${model_index}.th"; done
        for model_index in 2 5 6 7 8 9; do require_file "${DEEPGO_ROOT}/data/bp/deepgozero_esm_plus_${model_index}.th"; done
        for model_index in 1 3 4 5 6 7; do require_file "${DEEPGO_ROOT}/data/cc/deepgozero_esm_plus_${model_index}.th"; done
        conda_run "${DEEPGO_ENV}" python -c "import torch; print('DeepGO-SE environment ok')"
    fi
    if method_enabled hayai; then
        require_file "${HAYAI_ROOT}/db/zen.dmnd"
        conda_run "${HAYAI_ENV}" diamond version
        if [[ "${DGG_HAYAI_USE_ORTHOLOGER:-1}" == "1" ]]; then
            conda_run "${HAYAI_ENV}" ODB-mapper --help >/dev/null
        fi
    fi
    if method_enabled gomap && [[ ! -f "${DGG_GOMAP_GAF:-}" && -z "${DGG_GOMAP_COMMAND:-}" ]]; then
        echo "[PREFLIGHT ERROR] GOMAP has a manual Argot2.5 step upstream." >&2
        echo "Set DGG_GOMAP_GAF to a completed GAF, or DGG_GOMAP_COMMAND to a site-specific command that writes ${RAW}/gomap_predictions.gaf" >&2
        return 1
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

prepare_inputs() {
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" prepare \
        --data-root "${DATA_ROOT}" --workspace "${BENCHMARK_ROOT}" \
        --expected-test-size 754 --extract-structures
    mkdir -p "${BENCHMARK_ROOT}/inputs/structures/valid_test"
    local structure
    for structure in "${BENCHMARK_ROOT}"/inputs/structures/valid/*.pdb; do
        ln -sfn "${structure}" "${BENCHMARK_ROOT}/inputs/structures/valid_test/$(basename "${structure}")"
    done
    for structure in "${BENCHMARK_ROOT}"/inputs/structures/test/*.pdb; do
        ln -sfn "${structure}" "${BENCHMARK_ROOT}/inputs/structures/valid_test/$(basename "${structure}")"
    done
    for structure in "${BENCHMARK_ROOT}"/inputs/structures/valid/*.pdb.gz; do
        ln -sfn "${structure}" "${BENCHMARK_ROOT}/inputs/structures/valid_test/$(basename "${structure}")"
    done
    for structure in "${BENCHMARK_ROOT}"/inputs/structures/test/*.pdb.gz; do
        ln -sfn "${structure}" "${BENCHMARK_ROOT}/inputs/structures/valid_test/$(basename "${structure}")"
    done
}

run_hybrid() {
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" export-hybrid \
        --data-root "${DATA_ROOT}" --workspace "${BENCHMARK_ROOT}" \
        --checkpoint-root "${CHECKPOINT_ROOT}" --graph-root "${GRAPH_ROOT}" \
        --batch-size 16 --workers 0 --require-cuda
}

run_sequence_baselines() {
    local selected=""
    method_enabled blast && selected="blast"
    method_enabled diamond && selected="${selected:+${selected},}diamond"
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" sequence-baselines \
        --workspace "${BENCHMARK_ROOT}" --methods "${selected}" --threads "${THREADS}" \
        --top-k 10 --max-hits 50 --min-qcov 0.5 --min-tcov 0.5
}

run_foldseek() {
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" foldseek \
        --workspace "${BENCHMARK_ROOT}" --foldseek "${FOLDSEEK}" --threads "${THREADS}" \
        --top-k 10 --max-hits 50 --min-qcov 0.5 --min-tcov 0.5
}

run_interpro() {
    mkdir -p "${RAW}/interpro"
    "${INTERPROSCAN}" -i "${BENCHMARK_ROOT}/inputs/valid_test.fasta" \
        -f TSV -o "${RAW}/interpro/interpro.tsv" -goterms -iprlookup \
        -cpu "${THREADS}" -dp
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-interpro \
        --workspace "${BENCHMARK_ROOT}" --input "${RAW}/interpro/interpro.tsv"
}

run_eggnog() {
    mkdir -p "${RAW}/eggnog"
    conda_run "${EGGNOG_ENV}" python "${EGGNOG_ROOT}/emapper.py" \
        -i "${BENCHMARK_ROOT}/inputs/valid_test.fasta" --itype proteins \
        -m diamond --data_dir "${EGGNOG_DATA}" --cpu "${THREADS}" \
        -o dgg_benchmark --output_dir "${RAW}/eggnog" \
        --go_evidence all --override
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-eggnog \
        --workspace "${BENCHMARK_ROOT}" \
        --input "${RAW}/eggnog/dgg_benchmark.emapper.annotations"
}

run_deepgoplus() {
    mkdir -p "${RAW}/deepgoplus"
    gzip -c "${BENCHMARK_ROOT}/inputs/valid_test.fasta" > "${RAW}/deepgoplus/valid_test.fasta.gz"
    conda_run "${DEEPGOPLUS_ENV}" deepgoplus \
        --data-root "${DEEPGOPLUS_ROOT}/data" \
        --in-file "${RAW}/deepgoplus/valid_test.fasta.gz" \
        --out-file "${RAW}/deepgoplus/predictions.tsv.gz" --threshold 0
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-scored \
        --workspace "${BENCHMARK_ROOT}" --method deepgoplus --delimiter tab \
        --score-column 5 \
        --mf "${RAW}/deepgoplus/predictions.tsv.gz" \
        --bp "${RAW}/deepgoplus/predictions.tsv.gz" \
        --cc "${RAW}/deepgoplus/predictions.tsv.gz"
}

run_deepfri_sequence() {
    mkdir -p "${RAW}/deepfri"
    local ont
    for ont in mf bp cc; do
        conda_run "${TF_ENV}" python "${DEEPFRI_ROOT}/predict.py" \
            --fasta_fn "${BENCHMARK_ROOT}/inputs/valid_test.fasta" -ont "${ont}" \
            -o "${RAW}/deepfri/sequence"
    done
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-deepfri \
        --workspace "${BENCHMARK_ROOT}" --method deepfri_sequence \
        --mf "${RAW}/deepfri/sequence_MF_pred_scores.json" \
        --bp "${RAW}/deepfri/sequence_BP_pred_scores.json" \
        --cc "${RAW}/deepfri/sequence_CC_pred_scores.json"
}

run_deepfri_structure() {
    mkdir -p "${RAW}/deepfri"
    local ont
    for ont in mf bp cc; do
        conda_run "${TF_ENV}" python "${DEEPFRI_ROOT}/predict.py" \
            --pdb_dir "${BENCHMARK_ROOT}/inputs/structures/valid_test" -ont "${ont}" \
            -o "${RAW}/deepfri/structure"
    done
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-deepfri \
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
        conda_run "${TORCH_ENV}" python "${TRANSFUN_ROOT}/predict.py" \
            --data-path "${data_dir}" --ontology "${full}" --input-type fasta \
            --fasta-path dgg_benchmark.fasta --pdb-path dgg_benchmark_pdbs \
            --cut-off 0 --output "dgg_benchmark_${short}.txt"
    done
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-scored \
        --workspace "${BENCHMARK_ROOT}" --method transfun \
        --mf "${data_dir}/dgg_benchmark_mf.txt" --bp "${data_dir}/dgg_benchmark_bp.txt" \
        --cc "${data_dir}/dgg_benchmark_cc.txt"
}

run_dpfunc() {
    require_file "${RAW}/interpro/interpro.tsv"
    mkdir -p "${RAW}/dpfunc"
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" make-dpfunc-manifest \
        --workspace "${BENCHMARK_ROOT}" --interpro-tsv "${RAW}/interpro/interpro.tsv" \
        --output "${RAW}/dpfunc/manifest.tsv"

    local source_workspace="${RAW}/dpfunc/mf"
    conda_run "${TORCH_ENV}" python "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/build_data_demo.py" \
        --manifest "${RAW}/dpfunc/manifest.tsv" --workspace "${source_workspace}" \
        --ontology mf --pdb-dir "${BENCHMARK_ROOT}/inputs/structures/valid_test" \
        --inter-idx "${DPFUNC_ROOT}/data/inter_idx.pkl" --go-obo "${DATA_ROOT}/go-basic.obo" \
        --mlb-source "${DPFUNC_ROOT}/mlb/mf_go.mlb" --esm-batch-size 4 --device cuda

    local short dp_workspace
    for short in mf bp cc; do
        dp_workspace="${RAW}/dpfunc/${short}"
        if [[ "${short}" != "mf" ]]; then
            conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" clone-dpfunc-workspace \
                --source "${source_workspace}" --target "${dp_workspace}" \
                --source-ontology mf --target-ontology "${short}" \
                --mlb-source "${DPFUNC_ROOT}/mlb/${short}_go.mlb"
        fi
        cp "${DPFUNC_ROOT}"/save_models/"${DPFUNC_PRENAME}_${short}_"*of3model.pt "${dp_workspace}/save_models/"
        conda_run "${TORCH_ENV}" python "${DPFUNC_ROOT}/DPFunc_demo_pipeline/scripts/run_dpfunc.py" predict \
            --workspace "${dp_workspace}" --ontology "${short}" --gpu 0 --pre-name "${DPFUNC_PRENAME}"
    done
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-dpfunc \
        --workspace "${BENCHMARK_ROOT}" --method dpfunc \
        --mf "${RAW}/dpfunc/mf/results/${DPFUNC_PRENAME}_mf_final.pkl" \
        --bp "${RAW}/dpfunc/bp/results/${DPFUNC_PRENAME}_bp_final.pkl" \
        --cc "${RAW}/dpfunc/cc/results/${DPFUNC_PRENAME}_cc_final.pkl"
}

run_deepgose() {
    mkdir -p "${RAW}/deepgose"
    cp "${BENCHMARK_ROOT}/inputs/valid_test.fasta" "${RAW}/deepgose/queries.fasta"
    conda_run "${DEEPGO_ENV}" python "${DEEPGO_ROOT}/predict.py" \
        -if "${RAW}/deepgose/queries.fasta" -dr "${DEEPGO_ROOT}/data" \
        -t 0 -bs 8 -d cuda
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-scored \
        --workspace "${BENCHMARK_ROOT}" --method deepgose --delimiter tab \
        --mf "${RAW}/deepgose/queries_preds_mf.tsv.gz" \
        --bp "${RAW}/deepgose/queries_preds_bp.tsv.gz" \
        --cc "${RAW}/deepgose/queries_preds_cc.tsv.gz"
}

run_hayai() {
    local orthologer=()
    [[ "${DGG_HAYAI_USE_ORTHOLOGER:-1}" == "1" ]] && orthologer=(--orthologer)
    conda_run "${HAYAI_ENV}" python "${PYTHON_SCRIPT}" run-hayai \
        --workspace "${BENCHMARK_ROOT}" --hayai-root "${HAYAI_ROOT}" \
        --output "${RAW}/hayai_annotations.tsv" --threads "${THREADS}" "${orthologer[@]}"
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-hayai \
        --workspace "${BENCHMARK_ROOT}" --input "${RAW}/hayai_annotations.tsv"
}

run_gomap() {
    local output="${DGG_GOMAP_GAF:-${RAW}/gomap_predictions.gaf}"
    if [[ ! -f "${output}" ]]; then
        bash -lc "${DGG_GOMAP_COMMAND}"
    fi
    require_file "${output}"
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" normalize-gomap \
        --workspace "${BENCHMARK_ROOT}" --input "${output}"
}

run_evaluation() {
    local required="${METHODS//hybrid/deepgreengo}"
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" evaluate \
        --workspace "${BENCHMARK_ROOT}" --bootstraps 1000 --require-methods "${required}"
    conda_run "${BASE_ENV}" python "${PYTHON_SCRIPT}" plot --workspace "${BENCHMARK_ROOT}"
}

echo "Project: ${PROJECT_DIR}"
echo "Benchmark: ${BENCHMARK_ROOT}"
echo "Methods: ${METHODS}"
preflight
stage 01_prepare_inputs prepare_inputs
method_enabled hybrid && stage 02_hybrid run_hybrid
SEQUENCE_TAG="naive"
method_enabled blast && SEQUENCE_TAG="${SEQUENCE_TAG}_blast"
method_enabled diamond && SEQUENCE_TAG="${SEQUENCE_TAG}_diamond"
stage "03_sequence_baselines_${SEQUENCE_TAG}" run_sequence_baselines
method_enabled foldseek && stage 04_foldseek run_foldseek
if method_enabled interproscan || method_enabled dpfunc; then stage 05_interproscan run_interpro; fi
method_enabled eggnog_mapper && stage 05b_eggnog_mapper run_eggnog
method_enabled deepfri_sequence && stage 06_deepfri_sequence run_deepfri_sequence
method_enabled deepfri_structure && stage 07_deepfri_structure run_deepfri_structure
method_enabled transfun && stage 08_transfun run_transfun
method_enabled dpfunc && stage 09_dpfunc run_dpfunc
method_enabled deepgoplus && stage 09b_deepgoplus run_deepgoplus
method_enabled deepgose && stage 10_deepgose run_deepgose
method_enabled hayai && stage 11_hayai run_hayai
method_enabled gomap && stage 12_gomap run_gomap
run_evaluation
echo "Benchmark completed: ${BENCHMARK_ROOT}"

