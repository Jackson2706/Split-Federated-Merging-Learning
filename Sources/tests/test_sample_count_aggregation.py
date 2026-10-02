import unittest
from unittest import mock

import torch

from ehsfp import get_ehsfp_config, reliability_weighted_aggregate


class SampleCountAggregationTests(unittest.TestCase):
    def test_config_accepts_sample_count_weighted_mode(self):
        config = get_ehsfp_config({"aggregation_mode": "sample_count_weighted"})
        self.assertEqual(config["aggregation_mode"], "sample_count_weighted")

    def _aggregate(self, support_map, mode):
        outputs = {
            "low": ({0: torch.tensor([0.0])}, {0: torch.tensor([0.0])}),
            "high": ({0: torch.tensor([10.0])}, {0: torch.tensor([0.0])}),
        }
        # Zero noise exposes the aggregate mean directly while still exercising
        # the production synthetic-data path.
        with mock.patch("ehsfp.aggregation.torch.randn", side_effect=torch.zeros):
            features, labels = reliability_weighted_aggregate(
                outputs,
                num_samples_per_class=1,
                device=torch.device("cpu"),
                support_map=support_map,
                aggregation_mode=mode,
            )
        self.assertTrue(torch.equal(labels, torch.tensor([0])))
        return features.squeeze()

    def test_weighted_mean_favors_higher_support_and_matches_hand_calculation(self):
        weighted = self._aggregate(
            {"low": {0: 1}, "high": {0: 3}}, "sample_count_weighted",
        )
        average = self._aggregate(
            {"low": {0: 1}, "high": {0: 3}}, "average",
        )

        self.assertAlmostEqual(weighted.item(), 7.5)  # (1*0 + 3*10) / 4
        self.assertGreater(weighted.item(), average.item())
        self.assertAlmostEqual(average.item(), 5.0)

    def test_all_zero_support_falls_back_to_average(self):
        support = {"low": {0: 0}, "high": {0: 0}}
        weighted = self._aggregate(support, "sample_count_weighted")
        average = self._aggregate(support, "average")

        self.assertTrue(torch.isfinite(weighted))
        self.assertTrue(torch.equal(weighted, average))


if __name__ == "__main__":
    unittest.main()
