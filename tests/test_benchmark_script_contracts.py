from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = PROJECT_ROOT / "arc slurms" / "run_full_benchmark.slurm"
SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "setup_arc_sota.sh"


def test_deepgoplus_preflight_matches_official_1_0_2_archive():
    benchmark = BENCHMARK_SCRIPT.read_text()
    setup = SETUP_SCRIPT.read_text()
    required_files = {
        "data/go.obo",
        "data/model.h5",
        "data/terms.pkl",
        "data/train_data.pkl",
        "data/train_data.dmnd",
    }

    for path in required_files:
        assert path in benchmark
    setup_file_loop = "for file in go.obo model.h5 terms.pkl train_data.pkl train_data.dmnd; do"
    assert setup_file_loop in setup
    obsolete_check = 'require_file "${DEEPGOPLUS_ROOT}/data/metadata/last_release.json"'
    assert obsolete_check not in benchmark
    assert obsolete_check not in setup
