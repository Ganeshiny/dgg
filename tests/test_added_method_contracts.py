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
        self.assertIn('"mkl<2024.1"', setup)
        self.assertIn('CURRENT_METHOD=heal', benchmark)
        self.assertIn('scripts/run_heal_arc.py', benchmark)
        self.assertIn('--method heal', benchmark)
        self.assertIn('stage 10b_heal run_heal', benchmark)

    def test_gat_go_has_strict_official_feature_integration(self):
        benchmark = BENCHMARK.read_text()
        setup = SETUP.read_text()
        runner = (PROJECT_ROOT / "scripts" / "run_gat_go_arc.py").read_text()

        self.assertIn("github.com/bl-2633/GAT-GO.git", setup)
        self.assertIn("90ec6d1067a893d4a51be715e41daf9fa4732952", setup)
        self.assertIn("--runtime-smoke-test", setup)
        self.assertIn("dgg_setup_tools", setup)
        self.assertIn("git_run clone", setup)
        self.assertIn("proteins_missing_or_invalid", runner)
        self.assertIn("labels_used_for_inference", runner)
        self.assertIn("Never access obj['label']", runner)
        self.assertIn('CURRENT_METHOD=gat_go', benchmark)
        self.assertIn('--method gat_go', benchmark)
        self.assertIn('stage 10c_gat_go run_gat_go', benchmark)

    def test_deepgraphgo_has_cpu_environment_and_synthetic_query_mapping(self):
        benchmark = BENCHMARK.read_text()
        setup = SETUP.read_text()
        runner = (PROJECT_ROOT / "scripts" / "run_deepgraphgo_arc.py").read_text()

        self.assertIn("github.com/yourh/DeepGraphGO.git", setup)
        self.assertIn("efdb1cb9425f4f48e4613c0a89e603f5542bcb19", setup)
        self.assertIn("--runtime-smoke-test", setup)
        self.assertIn('"pytorch=1.6.0" cpuonly', setup)
        self.assertIn("query_identifiers_synthetic", runner)
        self.assertIn("DGGQ{index:06d}", runner)
        self.assertIn('CURRENT_METHOD=deepgraphgo', benchmark)
        self.assertIn('--method deepgraphgo', benchmark)
        self.assertIn('stage 10d_deepgraphgo run_deepgraphgo', benchmark)

    def test_struct2go_is_removed_and_plot_registers_new_methods(self):
        benchmark = BENCHMARK.read_text()
        setup = SETUP.read_text()
        source = PLOT.read_text()

        self.assertNotIn("struct2go", benchmark.lower())
        self.assertNotIn("struct2go", setup.lower())
        self.assertNotIn('"struct2go":', source.lower())
        self.assertIn('"heal": "HEAL (PDB-only)"', source)
        self.assertIn('"gat_go": "GAT-GO"', source)
        self.assertIn('"deepgraphgo": "DeepGraphGO"', source)
