from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = PROJECT_ROOT / "arc slurms" / "run_full_benchmark.slurm"
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "setup_arc_sota.sh"


def test_deepgoplus_preflight_matches_official_1_0_2_cli_contract():
    benchmark = BENCHMARK_SCRIPT.read_text()
    setup = SETUP_SCRIPT.read_text()
    required_files = {
        "data/go.obo",
        "data/model.h5",
        "data/terms.pkl",
        "data/train_data.pkl",
        "data/train_data.dmnd",
        "data/metadata/last_release.json",
    }

    for path in required_files:
        assert path in benchmark
    setup_file_loop = "for file in go.obo model.h5 terms.pkl train_data.pkl train_data.dmnd; do"
    assert setup_file_loop in setup
    metadata_check = 'require_file "${DEEPGOPLUS_ROOT}/data/metadata/last_release.json"'
    assert metadata_check in benchmark
    assert metadata_check in setup
    assert '{"alphas":{"mf":0.55,"bp":0.59,"cc":0.46}}' in setup
