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
            plot.METHOD_LABEL["deepgreengo"], "DeepGreenGO Hybrid (this work)"
        )

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
        metrics, bootstrap = plot.load_comparison(plot.DEFAULT_WORKSPACE)
        rows = metrics.loc[metrics["method"] == "deepgreengo"]
        self.assertEqual(
            rows.groupby("ontology").size().to_dict(),
            {ontology: 1 for ontology in plot.ONTOLOGY_ORDER},
        )
        self.assertEqual(plot.ordered_methods(metrics)[0], "deepgreengo")
        bootstrap_counts = bootstrap.groupby(["method", "ontology"]).size()
        self.assertTrue((bootstrap_counts == 1000).all())
        self.assertEqual(
            plot.load_deepgreengo_provenance(plot.DEFAULT_WORKSPACE),
            ([1103, 2207, 3301, 4409, 5501], "Hybrid"),
        )

    def test_legend_explains_families_and_transfer_rules(self):
        methods = [
            "deepgreengo", "naive", "blast", "blast_max", "foldseek"
        ]
        labels = [
            handle.get_label()
            for handle in plot.build_legend_handles(methods)
        ]
        self.assertIn("DeepGreenGO Hybrid (this work)", labels)
        self.assertIn("Sequence alignment", labels)
        self.assertIn("Top-10 weighted transfer", labels)
        self.assertIn("Single best identity (within top-10)", labels)

    def test_coverage_excludes_dense_output_methods(self):
        metrics, _ = plot.load_comparison(plot.DEFAULT_WORKSPACE)
        methods = plot.ordered_methods(metrics)
        coverage_methods = [
            method for method in plot.COVERAGE_METHOD_ORDER if method in methods
        ]
        self.assertNotIn("deepgreengo", coverage_methods)
        self.assertNotIn("naive", coverage_methods)
        self.assertTrue(set(coverage_methods).issubset(set(methods)))

    def test_paired_fmax_report_uses_1000_matched_draws(self):
        metrics, bootstrap = plot.load_comparison(plot.DEFAULT_WORKSPACE)
        report = plot.paired_fmax_report(metrics, bootstrap)
        self.assertEqual(set(report["ontology"]), set(plot.ONTOLOGY_ORDER))
        self.assertTrue((report["bootstrap_replicates"] == 1000).all())

    def test_bmc_export_is_pdf_and_high_resolution_png(self):
        self.assertEqual(plot.SPEC["raster"], "png")
        self.assertGreaterEqual(plot.SPEC["main_dpi"], 300)


if __name__ == "__main__":
    unittest.main()
