import unittest

import numpy as np

from src.benchmark.evaluate import (
    bootstrap_aupr,
    propagate_scores_to_ancestors,
    scalar_metrics,
    sequence_cluster_bootstrap_weights,
)


class BootstrapAuprTests(unittest.TestCase):
    def test_prediction_scores_are_propagated_to_all_represented_ancestors(self):
        scores = np.asarray([[0.1, 0.8, 0.2]], dtype=np.float32)
        terms = ["GO:ROOT", "GO:PARENT", "GO:CHILD"]
        parents = {
            "GO:CHILD": {"GO:PARENT"},
            "GO:PARENT": {"GO:ROOT"},
        }

        propagated = propagate_scores_to_ancestors(scores, terms, parents)

        np.testing.assert_allclose(propagated, [[0.8, 0.8, 0.2]])

    def test_identical_sequences_are_resampled_as_one_cluster(self):
        rng = np.random.default_rng(17)
        weights, cluster_count = sequence_cluster_bootstrap_weights(
            ["AAAA", "BBBB", "AAAA", "CCCC", "BBBB"],
            bootstraps=50,
            rng=rng,
        )

        self.assertEqual(cluster_count, 3)
        np.testing.assert_array_equal(weights[:, 0], weights[:, 2])
        np.testing.assert_array_equal(weights[:, 1], weights[:, 4])
        np.testing.assert_array_equal(
            weights[:, [0, 1, 3]].sum(axis=1),
            np.full(50, 3.0),
        )

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
