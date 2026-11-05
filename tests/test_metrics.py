import unittest

from arc_decision.metrics import paired_accuracy_difference


class MetricsTests(unittest.TestCase):
    def test_paired_difference_is_comparison_minus_baseline(self):
        result = paired_accuracy_difference([False, True], [True, True], samples=100, seed=3)
        self.assertEqual(result["sample_size"], 2)
        self.assertEqual(result["difference_pp"], 50.0)
        self.assertLessEqual(result["ci_low_pp"], result["difference_pp"])
        self.assertGreaterEqual(result["ci_high_pp"], result["difference_pp"])
