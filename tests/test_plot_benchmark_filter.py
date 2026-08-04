from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import plot_benchmark as plot  # noqa: E402


class BaselineOnlyFilterTests(unittest.TestCase):
    def test_allowlist_is_exact_and_excludes_unrequested_methods(self):
        self.assertEqual(
            plot.METHOD_ORDER,
            [
                "deepgreengo", "naive", "blast", "blast_max", "diamond", "diamond_max",
                "foldseek", "foldseek_max", "deepfri_sequence",
                "deepfri_structure", "dpfunc", "heal",
            ],
        )
        self.assertIn("deepgreengo", plot.REQUESTED_METHODS)
        self.assertTrue(
            {"deepgose", "deepgoplus", "interproscan", "gat_go", "sprof_go"}
            .isdisjoint(plot.REQUESTED_METHODS)
        )

    def test_table_filter_drops_every_method_outside_allowlist(self):
        frame = pd.DataFrame(
            {"method": ["naive", "deepgreengo", "dpfunc", "interproscan"], "value": range(4)}
        )
        selected = plot.select_requested_methods(frame)
        self.assertEqual(selected["method"].tolist(), ["naive", "deepgreengo", "dpfunc"])

    def test_every_allowed_method_has_a_label_and_family(self):
        self.assertTrue(plot.REQUESTED_METHODS <= plot.METHOD_LABEL.keys())
        self.assertTrue(plot.REQUESTED_METHODS <= plot.METHOD_FAMILY.keys())


if __name__ == "__main__":
    unittest.main()
