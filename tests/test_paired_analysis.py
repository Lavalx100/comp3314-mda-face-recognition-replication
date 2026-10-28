import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_paired_results import paired_differences  # noqa: E402


class PairedAnalysisTests(unittest.TestCase):
    def test_paired_difference_uses_same_split_and_percentage_points(self):
        rows = []
        for seed, reference, comparison in [(10, 0.8, 0.7), (11, 0.7, 0.6), (12, 0.9, 0.8)]:
            rows.extend([
                {"train_per_class": "2", "seed": str(seed), "method": "MDA/2-2", "accuracy": str(reference)},
                {"train_per_class": "2", "seed": str(seed), "method": "PCA", "accuracy": str(comparison)},
            ])
        result = paired_differences(rows, bootstrap_samples=100, seed=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["splits"], 3)
        self.assertAlmostEqual(result[0]["mean_difference_pp"], 10.0)
        self.assertEqual(result[0]["reference_win_fraction"], 1.0)

    def test_paired_difference_rejects_unmatched_method_rows(self):
        rows = [
            {"train_per_class": "2", "seed": "10", "method": "MDA/2-2", "accuracy": "0.8"},
            {"train_per_class": "2", "seed": "10", "method": "PCA", "accuracy": "0.7"},
            {"train_per_class": "2", "seed": "11", "method": "MDA/2-2", "accuracy": "0.9"},
        ]
        with self.assertRaises(ValueError):
            paired_differences(rows, bootstrap_samples=10)


if __name__ == "__main__":
    unittest.main()
