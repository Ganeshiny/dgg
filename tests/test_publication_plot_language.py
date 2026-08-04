from __future__ import annotations

import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PLOT_FILES = [
    PROJECT / "src" / "benchmark" / "evaluate.py",
    *sorted((PROJECT / "src").glob("plot*.py")),
    *sorted((PROJECT / "plots").glob("*.py")),
]


class PublicationPlotLanguageTests(unittest.TestCase):
    def test_no_directional_guidance_is_embedded_in_plots_or_captions(self):
        banned = (
            "higher is better",
            "higher values are better",
            "higher values indicate",
            "lower is better",
            "lower-is-better",
            "upper-right is better",
            "\u2191",
            "\u2193",
        )
        for path in PLOT_FILES:
            source = path.read_text()
            lowered = source.lower()
            for phrase in banned:
                self.assertNotIn(phrase.lower(), lowered, str(path))

    def test_fmax_and_smin_are_not_prefixed_with_cafa_in_plotting_code(self):
        banned = ("cafa fmax", "cafa smin", "cafa f$_", "cafa s$_")
        for path in PLOT_FILES:
            lowered = path.read_text().lower()
            for phrase in banned:
                self.assertNotIn(phrase, lowered, str(path))


if __name__ == "__main__":
    unittest.main()
