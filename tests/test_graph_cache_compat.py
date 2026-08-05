from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.graph_cache_compat import resolve_complete_graph_cache


class GraphCacheResolutionTests(unittest.TestCase):
    def test_partial_requested_cache_falls_back_to_complete_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = root / "partial"
            complete = root / "complete"
            partial.mkdir()
            complete.mkdir()
            (partial / "P1.pt").touch()
            for protein_id in ("P1", "P2", "P3"):
                (complete / f"{protein_id}.pt").touch()

            selected = resolve_complete_graph_cache(
                ["P1", "P2", "P3"],
                [("requested", partial), ("training", complete)],
            )

            self.assertEqual(selected, complete.resolve())

    def test_missing_cache_error_audits_every_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partial = root / "partial"
            partial.mkdir()
            (partial / "P1.pt").touch()

            with self.assertRaisesRegex(
                FileNotFoundError,
                r"missing 1/2 graphs.*P2",
            ):
                resolve_complete_graph_cache(
                    ["P1", "P2"],
                    [("missing", root / "absent"), ("partial", partial)],
                )

    def test_empty_protein_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty protein set"):
            resolve_complete_graph_cache([], [])


if __name__ == "__main__":
    unittest.main()
