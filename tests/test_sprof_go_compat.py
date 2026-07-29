import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "pickle_compat.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("pickle_compat", MODULE_PATH)
pickle_compat = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pickle_compat)
PREP_SPEC = importlib.util.spec_from_file_location(
    "prepare_sprof_go_input", MODULE_PATH.parent / "prepare_sprof_go_input.py"
)
prepare = importlib.util.module_from_spec(PREP_SPEC)
assert PREP_SPEC.loader is not None
PREP_SPEC.loader.exec_module(prepare)
EVAL_SPEC = importlib.util.spec_from_file_location(
    "evaluate_sprof_go", MODULE_PATH.parent / "evaluate_sprof_go.py"
)
evaluate = importlib.util.module_from_spec(EVAL_SPEC)
assert EVAL_SPEC.loader is not None
EVAL_SPEC.loader.exec_module(evaluate)


class SprofGoPickleCompatibilityTests(unittest.TestCase):
    def test_numpy_2_private_core_namespace_is_remapped(self):
        self.assertEqual(
            pickle_compat.NumpyCompatUnpickler.compatible_module(
                "numpy._core.numeric"
            ),
            "numpy.core.numeric",
        )

    def test_non_numpy_pickle_globals_are_not_rewritten(self):
        self.assertEqual(
            pickle_compat.NumpyCompatUnpickler.compatible_module(
                "src.arc_dataset"
            ),
            "src.arc_dataset",
        )

    def test_list_of_record_dictionaries_exposes_ids(self):
        records = [
            {"id": "1ABC_A", "sequence": "MA"},
            {"protein_id": "2DEF_B", "sequence": "MV"},
        ]

        self.assertEqual(prepare.dataset_ids(records), ["1ABC_A", "2DEF_B"])

    def test_zero_scores_do_not_count_as_positive_at_zero_threshold(self):
        truth = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
        scores = np.zeros_like(truth, dtype=np.float32)

        result = evaluate.metrics(truth, scores)

        self.assertEqual(result["micro_fmax"], 0.0)
        self.assertEqual(result["macro_fmax"], 0.0)
