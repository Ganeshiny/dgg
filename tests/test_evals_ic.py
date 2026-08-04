from __future__ import annotations

import unittest

import numpy as np

from src.evals import compute_ic


class TrainingFrequencyInformationContentTests(unittest.TestCase):
    def test_unseen_terms_receive_one_count_floor(self):
        labels = np.asarray([
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
        ], dtype=np.uint8)

        observed = compute_ic(labels)
        expected = -np.log2(np.asarray([3, 1, 1], dtype=float) / 4)

        np.testing.assert_allclose(observed, expected)
        self.assertTrue(np.all(np.isfinite(observed)))

    def test_empty_training_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty training set"):
            compute_ic(np.empty((0, 3), dtype=np.uint8))

    def test_label_matrix_must_be_two_dimensional(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            compute_ic(np.asarray([1, 0, 1], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
