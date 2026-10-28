import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_experiment as experiment  # noqa: E402


class ExperimentTests(unittest.TestCase):
    def test_parse_sizes_deduplicates_and_sorts(self):
        self.assertEqual(experiment.parse_sizes("5, 2,3,2"), [2, 3, 5])

    def test_parse_sizes_rejects_out_of_range(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            experiment.parse_sizes("1,10")

    def test_stratified_split_is_repeatable_balanced_and_disjoint(self):
        labels = np.repeat(np.arange(4), 6)
        train, test = experiment.stratified_split(labels, train_per_class=2, seed=17)
        train_again, test_again = experiment.stratified_split(labels, train_per_class=2, seed=17)
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(test, test_again)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(train) + len(test), len(labels))
        self.assertTrue(np.all(np.bincount(labels[train], minlength=4) == 2))
        self.assertTrue(np.all(np.bincount(labels[test], minlength=4) == 4))

    def test_histogram_equalization_preserves_shape_and_range(self):
        images = np.stack([
            np.full((4, 4), 0.4),
            np.arange(16, dtype=float).reshape(4, 4) / 15.0,
        ])
        result = experiment.equalize_histogram(images)
        self.assertEqual(result.shape, images.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_generalized_eigenvectors_are_finite_and_have_requested_shape(self):
        between = np.diag([5.0, 2.0, 1.0])
        within = np.diag([2.0, 1.0, 0.5])
        projection = experiment.leading_generalized_eigenvectors(between, within, n_components=2)
        self.assertEqual(projection.shape, (3, 2))
        self.assertTrue(np.isfinite(projection).all())
        np.testing.assert_allclose(np.linalg.norm(projection, axis=0), 1.0)

    def test_nearest_neighbor_uses_nearest_training_label(self):
        train = np.array([[0.0], [10.0], [5.0]])
        labels = np.array([3, 9, 6])
        test = np.array([[0.2], [9.0], [5.4]])
        np.testing.assert_array_equal(
            experiment.predict_nearest_neighbor(train, labels, test),
            np.array([3, 9, 6]),
        )

    def test_mda_two_mode_projection_shapes(self):
        rng = np.random.default_rng(7)
        train_images = rng.normal(size=(9, 5, 4))
        test_images = rng.normal(size=(3, 5, 4))
        train_labels = np.repeat(np.arange(3), 3)
        train_features, test_features, objective = experiment.fit_mda(
            train_images, train_labels, test_images,
            components=(2, 2), max_iter=2, tolerance=0.0,
            projection_normalization="unit_l2",
        )
        self.assertEqual(train_features.shape, (9, 4))
        self.assertEqual(test_features.shape, (3, 4))
        self.assertGreaterEqual(len(objective), 1)
        self.assertTrue(np.isfinite(objective).all())


if __name__ == "__main__":
    unittest.main()
