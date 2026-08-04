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
            plot.METHOD_LABEL["deepgreengo"], "DeepGreenGO"
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
        self.assertEqual(
            plot.METHOD_LABEL["deepgreengo"],
            "DeepGreenGO",
        )
        self.assertEqual(plot.FAMILY_LABEL["sequence"], "Sequence alignment")
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

    def test_caption_discloses_duplicate_sequences_and_external_pretraining(self):
        report = plot.pd.DataFrame([
            {
                "ontology": ontology,
                "fmax_difference": -0.1,
                "competitor_label": "DeepGOPlus",
                "fraction_bootstraps_deepgreengo_better": 0.0,
            }
            for ontology in plot.ONTOLOGY_ORDER
        ])
        caption = plot.build_captions(
            [1103, 2207, 3301, 4409, 5501],
            "Hybrid",
            report,
            True,
            bootstrap_unit="protein",
            unique_sequences=140,
        )
        self.assertIn("only 140 unique sequences among 754 chains", caption)
        self.assertIn("not leakage-controlled generalization estimates", caption)

    def _paired_report(self, competitor: str):
        return plot.pd.DataFrame([
            {
                "ontology": ontology,
                "fmax_difference": -0.1,
                "competitor_label": competitor,
                "fraction_bootstraps_deepgreengo_better": 0.0,
            }
            for ontology in plot.ONTOLOGY_ORDER
        ])

    def test_caption_mentions_only_completed_external_methods(self):
        caption = plot.build_captions(
            [1103, 2207, 3301, 4409, 5501],
            "Hybrid",
            self._paired_report("HEAL (PDB-only)"),
            True,
            methods=["deepgreengo", "heal"],
        )
        self.assertIn("HEAL (PDB-only) uses externally released", caption)
        self.assertNotIn("GAT-GO", caption)
        self.assertNotIn("Struct2GO", caption)

    def test_caption_omits_the_external_caveat_when_none_are_plotted(self):
        """An empty external set must not leave a subjectless sentence."""
        caption = plot.build_captions(
            [1103, 2207, 3301, 4409, 5501],
            "Hybrid",
            self._paired_report("BLAST (top-10)"),
            True,
            methods=["deepgreengo", "blast"],
        )
        self.assertNotIn("use externally released", caption)
        self.assertNotIn("uses externally released", caption)
        self.assertIn("Paired Fmax comparisons", caption)

    def test_deepgoplus_and_deepgose_are_excluded_from_plotted_methods(self):
        metrics = plot.pd.DataFrame([
            {"method": method, "ontology": "molecular_function"}
            for method in ("deepgreengo", "blast", "deepgoplus", "deepgose", "heal")
        ])
        methods = plot.ordered_methods(metrics)
        self.assertNotIn("deepgoplus", methods)
        self.assertNotIn("deepgose", methods)
        self.assertIn("heal", methods)
        self.assertIn("deepgreengo", methods)
        self.assertNotIn("deepgoplus", plot.EXTERNAL_PRETRAINED_METHODS)
        self.assertNotIn("deepgose", plot.EXTERNAL_PRETRAINED_METHODS)

    def test_manuscript_note_reports_baseline_win_in_correct_direction(self):
        metrics = plot.pd.DataFrame([
            {
                "method": "deepgreengo",
                "ontology": "molecular_function",
                "cafa_fmax": 0.384,
                "cafa_smin": 5.02,
            },
            {
                "method": "deepgoplus",
                "ontology": "molecular_function",
                "cafa_fmax": 0.543,
                "cafa_smin": 2.07,
            },
        ])
        note = plot.build_manuscript_notes(metrics, "Hybrid")
        self.assertIn("underperforms the strongest baseline on both MF metrics", note)
        self.assertIn("does not support an MF accuracy-gain claim", note)
        self.assertNotIn("MF gain is clearer", note)


if __name__ == "__main__":
    unittest.main()
