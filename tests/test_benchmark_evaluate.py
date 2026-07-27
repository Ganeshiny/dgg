import unittest

import numpy as np

from src.benchmark.evaluate import bootstrap_aupr, scalar_metrics


class BootstrapAuprTests(unittest.TestCase):
    def test_unit_weights_match_unweighted_point_estimate(self):
        truth = np.asarray(
            [[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]],
            dtype=np.uint8,
        )
        scores = np.asarray(
            [
                [0.9, 0.2, 0.1],
                [0.2, 0.8, 0.3],
                [0.7, 0.1, 0.8],
                [0.1, 0.7, 0.9],
            ],
            dtype=np.float32,
        )
        weights = np.ones((3, len(truth)), dtype=np.float32)

        micro, macro = bootstrap_aupr(truth, scores, weights, workers=2)
        point = scalar_metrics(truth, scores, "continuous")

        np.testing.assert_allclose(micro, point["micro_aupr"])
        np.testing.assert_allclose(macro, point["macro_aupr"])


if __name__ == "__main__":
    unittest.main()
