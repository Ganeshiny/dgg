from __future__ import annotations

import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ALL_FIGURES = PROJECT / "arc slurms" / "run_all_figures.slurm"
FULL_BENCHMARK = PROJECT / "arc slurms" / "run_full_benchmark.slurm"


class CompleteFigureScriptContractTests(unittest.TestCase):
    def test_every_required_output_group_is_rendered_and_count_checked(self):
        script = ALL_FIGURES.read_text()
        required = {
            "main_text": 3,
            "supplementary": 19,
            "supplementary_tables": 10,
            "supplementary_tuning": 6,
            "reviewer": 2,
            "benchmark": 4,
        }
        for folder, count in required.items():
            self.assertIn(f'require_count "$OUT/{folder}"', script)
            self.assertIn(f" {count} {folder}", script)

    def test_comparison_keeps_deepgreengo_and_only_requested_baselines(self):
        script = ALL_FIGURES.read_text()
        self.assertIn("'deepgreengo'", script)
        for excluded in ("deepgose", "deepgoplus", "interproscan", "gat_go", "sprof_go"):
            self.assertNotIn(f"'{excluded}'", script)

    def test_full_benchmark_regenerates_entire_figure_tree(self):
        script = FULL_BENCHMARK.read_text()
        self.assertIn("hybrid,naive,blast,diamond,foldseek", script)
        self.assertIn('bash "${PROJECT_DIR}/arc slurms/run_all_figures.slurm"', script)


if __name__ == "__main__":
    unittest.main()
