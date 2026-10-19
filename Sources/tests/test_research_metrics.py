import unittest

import torch

from ehsfp.research_metrics import (
    prototype_drift,
    prototype_fidelity_mmd,
    recovery_gap,
    recovery_rounds,
    rounds_to_convergence,
    trailing_window_stability,
)


class ResearchMetricsTests(unittest.TestCase):
    def test_trailing_stability_uses_population_variance(self):
        actual = trailing_window_stability([1.0, 2.0, 3.0, 4.0], window=3)
        self.assertEqual(actual[:2], [None, None])
        self.assertAlmostEqual(actual[2], 2.0 / 3.0)
        self.assertAlmostEqual(actual[3], 2.0 / 3.0)

    def test_rounds_to_convergence_clean(self):
        history = [0.2, 0.4, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9]
        self.assertEqual(rounds_to_convergence(history), 6)

    def test_rounds_to_convergence_never_converges(self):
        history = [0.1, 0.1, 1.0, 0.1, 0.1]
        self.assertEqual(rounds_to_convergence(history), "not_converged")

    def test_rounds_to_convergence_skips_failed_first_crossing(self):
        history = [0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9]
        self.assertEqual(rounds_to_convergence(history), 7)

    def test_prototype_drift_per_class_and_summaries(self):
        previous = {0: torch.tensor([0.0, 0.0]), 1: torch.tensor([1.0, 1.0])}
        current = {0: torch.tensor([3.0, 4.0]), 1: torch.tensor([1.0, 3.0])}
        drift = prototype_drift(previous, current)
        self.assertEqual(drift, {0: 5.0, 1: 2.0, "global": 3.5, "max": 5.0})

    def test_prototype_drift_round_one_is_undefined(self):
        self.assertIsNone(prototype_drift(None, {0: torch.tensor([1.0])}))
        self.assertIsNone(prototype_drift({}, {0: torch.tensor([1.0])}))

    def test_recovery_metrics(self):
        self.assertAlmostEqual(recovery_gap(0.91, 0.73), 0.18)
        self.assertEqual(recovery_rounds([0.6, 0.8, 0.86], 0.9), 3)
        self.assertIsNone(recovery_rounds([0.6, 0.8, 0.85], 0.9))

    def test_prototype_fidelity_wrapper_matches_identical_features(self):
        features = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertAlmostEqual(
            prototype_fidelity_mmd(features, features, sigmas=[torch.tensor(1.0)]),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
