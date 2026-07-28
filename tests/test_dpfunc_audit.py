#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audit_dpfunc_integration import score_summary


class DPFuncAuditTests(unittest.TestCase):
    def test_score_summary_includes_implicit_zeros(self):
        frame = pd.DataFrame({
            "protein_id": ["p1", "p2"],
            "predictions": [
                {"GO:1": 0.8, "GO:outside": 0.9},
                {"GO:2": 0.2},
            ],
        })
        result = score_summary(
            frame,
            ["p1", "p2"],
            ["GO:1", "GO:2", "GO:3"],
        )
        self.assertEqual(result["observed_common_term_scores"], 2)
        self.assertEqual(result["implicit_zero_scores"], 4)
        self.assertEqual(result["proteins_with_any_positive_score"], 2)
        self.assertAlmostEqual(result["score_q100"], 0.8)
        self.assertGreater(result["score_sd"], 0)

    def test_score_summary_flags_invalid_ranges(self):
        frame = pd.DataFrame({
            "protein_id": ["p1"],
            "predictions": [{"GO:1": 1.2, "GO:2": float("nan")}],
        })
        result = score_summary(frame, ["p1"], ["GO:1", "GO:2"])
        self.assertEqual(result["invalid_score_count"], 1)
        self.assertEqual(result["out_of_range_score_count"], 1)


if __name__ == "__main__":
    unittest.main()
