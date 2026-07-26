import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "src" / "plot_baselines_only.py"
sys.path.insert(0, str(PROJECT / "src"))
SPEC = importlib.util.spec_from_file_location("plot_baselines_only", MODULE_PATH)
plot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot)


class MainComparisonPlotTests(unittest.TestCase):
    def test_deepgreengo_is_focal_and_visually_unique(self):
        self.assertEqual(plot.METHOD_ORDER[0], "deepgreengo")
        self.assertEqual(
            plot.METHOD_LABEL["deepgreengo"], "DeepGreenGO (this work)"
        )
        self.assertEqual(plot.marker("deepgreengo"), "*")
        proposed = plot.FAMILY_COLOR["proposed"]
        self.assertNotIn(
            proposed,
            [
                color
                for family, color in plot.FAMILY_COLOR.items()
                if family != "proposed"
            ],
        )

    def test_real_benchmark_has_one_ensemble_row_per_ontology(self):
        metrics, _ = plot.load_comparison(plot.DEFAULT_WORKSPACE)
        rows = metrics.loc[metrics["method"] == "deepgreengo"]
        self.assertEqual(
            rows.groupby("ontology").size().to_dict(),
            {ontology: 1 for ontology in plot.ONTOLOGY_ORDER},
        )
        self.assertEqual(plot.ordered_methods(metrics)[0], "deepgreengo")

    def test_legend_explains_families_and_transfer_rules(self):
        methods = [
            "deepgreengo", "naive", "blast", "blast_max", "foldseek"
        ]
        labels = [
            handle.get_label()
            for handle in plot.build_legend_handles(methods)
        ]
        self.assertIn("DeepGreenGO (this work)", labels)
        self.assertIn("Sequence alignment", labels)
        self.assertIn("Top-10 hits (summed)", labels)
        self.assertIn("Best single hit (max identity)", labels)

    def test_bmc_export_is_pdf_and_high_resolution_png(self):
        self.assertEqual(plot.SPEC["raster"], "png")
        self.assertGreaterEqual(plot.SPEC["main_dpi"], 300)


if __name__ == "__main__":
    unittest.main()
