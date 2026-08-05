from __future__ import annotations

import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ALL_FIGURES = PROJECT / "arc slurms" / "run_all_figures.slurm"
BIN_EVAL = PROJECT / "arc slurms" / "run_arc_bin_eval.slurm"
FULL_BENCHMARK = PROJECT / "arc slurms" / "run_full_benchmark.slurm"


class CompleteFigureScriptContractTests(unittest.TestCase):
    def test_every_required_output_group_is_rendered_and_count_checked(self):
        script = ALL_FIGURES.read_text()
        required = {
            "main_text": 3,
            "supplementary": 35,
            "supplementary_tables": 10,
            "supplementary_tuning": 6,
            "reviewer": 2,
            "benchmark": 27,
        }
        for folder, count in required.items():
            self.assertIn(f'require_count "$OUT/{folder}"', script)
            self.assertIn(f" {count} {folder}", script)
        self.assertIn(
            """require_count "$BENCHMARK_BINS" '*.pdf' 6 benchmark_bin_evaluation""",
            script,
        )
        self.assertIn(
            """require_count "$BENCHMARK_BINS" '*.csv' 5 benchmark_bin_tables""",
            script,
        )
        self.assertIn('--output "$BENCHMARK_BINS"', script)
        self.assertIn('--output "$OUT/benchmark"', script)
        self.assertIn('--output-dir "$BENCHMARK_TABLES"', script)
        for suffix in ("svg", "png", "tiff"):
            self.assertIn(f'''require_count "$OUT/benchmark" '*.{suffix}' 27''', script)

    def test_ablation_metrics_are_regenerated_and_all_are_plotted(self):
        figure_script = ALL_FIGURES.read_text()
        bin_job = BIN_EVAL.read_text()
        evaluator = (PROJECT / "src" / "evaluate_arc_bins.py").read_text()
        builder = (PROJECT / "src" / "make_figures.py").read_text()
        bin_plotter = (PROJECT / "src" / "plot_arc_bins.py").read_text()
        ablation_plotter = (PROJECT / "src" / "plot_arc_ablations.py").read_text()
        self.assertIn("--overall-output", bin_job)
        self.assertIn("--splits valid test", bin_job)
        self.assertIn("#SBATCH --partition=cpu", bin_job)
        self.assertIn('"auprc_estimator": "sklearn.metrics.average_precision_score"', evaluator)
        self.assertIn('"smin_weighting": "training-frequency information content"', evaluator)
        self.assertIn('"smin_zero_frequency_policy": "one-count floor"', evaluator)
        self.assertIn('manifest.get("smin_zero_frequency_policy")', figure_script)
        self.assertIn("load_pickle_compat(handle)", evaluator)
        self.assertIn("resolve_complete_graph_cache", evaluator)
        self.assertIn('${PROJECT_DIR}/arc_tuning/graphs_protbert', bin_job)
        self.assertNotIn('${PROJECT_DIR}/arc_tuning_cafa/graphs_protbert', bin_job)
        self.assertIn("ARC bin evaluation job:", bin_job)
        self.assertIn("load_pickle_compat(handle)", bin_plotter)
        self.assertIn("load_pickle_compat(handle)", ablation_plotter)
        self.assertIn('manifest.get("rows") != 225', figure_script)
        self.assertIn('for metric in METRIC_ORDER:', builder)
        self.assertNotIn('metric == "Smin" or metric not in subset', builder)
        self.assertNotIn('metric != "Smin"', bin_plotter)
        self.assertIn('"Smin": "S$_{min}^{freq}$"', (PROJECT / "src" / "plot_style.py").read_text())
        self.assertIn('"Smin_freq"', bin_plotter)
        grid_plotter = (PROJECT / "src" / "plot_ablation_summary_grids.py").read_text()
        self.assertIn("plot_ablation_summary_grids.py", figure_script)
        self.assertIn('KEY_METRICS = ("Micro_Fmax", "Micro_AUPRC", "Macro_AUPRC")', grid_plotter)
        self.assertIn('BIN_MODELS = ("Hybrid", "Hybrid_JK")', grid_plotter)
        self.assertIn('"--allow-unverified-auprc"', grid_plotter)
        self.assertIn('"--grid"', grid_plotter)
        self.assertIn('choices=("both", "metrics", "bins")', grid_plotter)
        self.assertIn('verified, reason = True, ""', grid_plotter)
        for suffix in ("png", "svg", "tiff"):
            self.assertIn(
                f'''require_count "$OUT/ablation_grids" '*.{suffix}' 2''',
                figure_script,
            )
        for suffix in ("svg", "png", "tiff"):
            self.assertIn(
                f'''require_count "$OUT/supplementary" '*.{suffix}' 35''',
                figure_script,
            )
    def test_comparison_keeps_deepgreengo_and_only_requested_baselines(self):
        script = ALL_FIGURES.read_text()
        self.assertIn("'deepgreengo'", script)
        for excluded in ("deepgose", "deepgoplus", "interproscan", "gat_go", "sprof_go"):
            self.assertNotIn(f"'{excluded}'", script)

    def test_full_benchmark_regenerates_entire_figure_tree(self):
        script = FULL_BENCHMARK.read_text()
        self.assertIn("hybrid,naive,blast,diamond,foldseek", script)
        self.assertIn('bash "${PROJECT_DIR}/arc slurms/run_all_figures.slurm"', script)
        self.assertIn("sbatch --parsable", script)
        self.assertIn('--dependency="afterok:${SLURM_JOB_ID}"', script)
        figure_script = ALL_FIGURES.read_text()
        self.assertIn("#SBATCH --partition=cpu", figure_script)


if __name__ == "__main__":
    unittest.main()
