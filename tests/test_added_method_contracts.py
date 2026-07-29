from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "arc slurms" / "run_full_benchmark.slurm"
SETUP = PROJECT_ROOT / "scripts" / "setup_arc_sota.sh"
PLOT = PROJECT_ROOT / "src" / "plot_baselines_only.py"


class AddedMethodContractTests(unittest.TestCase):
    def test_heal_has_setup_preflight_runner_normalization_and_stage(self):
        benchmark = BENCHMARK.read_text()
        setup = SETUP.read_text()

        self.assertIn("setup_heal()", setup)
        self.assertIn("github.com/ZhonghuiGu/HEAL.git", setup)
        self.assertIn("esm1b_t33_650M_UR50S.pt", setup)
        self.assertIn('CURRENT_METHOD=heal', benchmark)
        self.assertIn('scripts/run_heal_arc.py', benchmark)
        self.assertIn('--method heal', benchmark)
        self.assertIn('stage 10b_heal run_heal', benchmark)

    def test_struct2go_requires_real_external_arc_scores(self):
        benchmark = BENCHMARK.read_text()
        setup = SETUP.read_text()

        self.assertIn("github.com/lyjps/Struct2GO.git", setup)
        self.assertIn('CURRENT_METHOD=struct2go', benchmark)
        self.assertIn('require_file "${STRUCT2GO_MF}"', benchmark)
        self.assertIn('require_file "${STRUCT2GO_BP}"', benchmark)
        self.assertIn('require_file "${STRUCT2GO_CC}"', benchmark)
        self.assertIn('--method struct2go', benchmark)
        self.assertIn('stage 10c_struct2go run_struct2go', benchmark)

    def test_comparison_plot_registers_both_methods(self):
        source = PLOT.read_text()

        self.assertIn('"heal": "HEAL (PDB-only)"', source)
        self.assertIn('"struct2go": "Struct2GO"', source)
