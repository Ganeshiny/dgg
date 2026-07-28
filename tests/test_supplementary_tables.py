import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.export_supplementary_tables import comparison_audit, ic_weighted_aupr


class SupplementaryTableTests(unittest.TestCase):
    def test_ic_weighted_aupr_integrates_recall_order(self):
        curves = pd.DataFrame({
            "method": ["deepgreengo"] * 3,
            "ontology": ["molecular_function"] * 3,
            "ic_weighted_precision": [1.0, 0.75, 0.5],
            "ic_weighted_recall": [0.0, 0.5, 1.0],
        })
        result = ic_weighted_aupr(curves)
        self.assertAlmostEqual(float(result.iloc[0].ic_weighted_aupr), 0.75)
        self.assertEqual(int(result.iloc[0].threshold_points), 3)

    def test_external_method_audit_does_not_claim_leakage_test(self):
        audit = comparison_audit(["deepgreengo", "deepgoplus"])
        external = audit.set_index("method").loc["deepgoplus"]
        self.assertIn("not audited", external.external_training_overlap_status)
        self.assertIn("cannot establish", external.project_homology_bin_interpretation)

    def test_undefined_curve_remains_nan(self):
        curves = pd.DataFrame({
            "method": ["interproscan"],
            "ontology": ["molecular_function"],
            "ic_weighted_precision": [np.nan],
            "ic_weighted_recall": [np.nan],
        })
        self.assertTrue(np.isnan(ic_weighted_aupr(curves).iloc[0].ic_weighted_aupr))


if __name__ == "__main__":
    unittest.main()
