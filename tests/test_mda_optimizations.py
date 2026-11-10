import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_experiment as experiment  # noqa: E402
from orl_data import read_pgm  # noqa: E402


def reference_mode_scatter(projected: np.ndarray, labels: np.ndarray, mode: int):
    """Small, transparent loop version of the paper's matching-coordinate sums."""
    if mode == 0:
        coordinate_count, feature_count = projected.shape[2], projected.shape[1]
        vectors_for_coordinate = lambda k: projected[:, :, k]
    else:
        coordinate_count, feature_count = projected.shape[1], projected.shape[2]
        vectors_for_coordinate = lambda k: projected[:, k, :]

    between = np.zeros((feature_count, feature_count), dtype=np.float64)
    within = np.zeros_like(between)
    for coordinate in range(coordinate_count):
        vectors = vectors_for_coordinate(coordinate)
        overall = vectors.mean(axis=0)
        for label in np.unique(labels):
            group = vectors[labels == label]
            class_mean = group.mean(axis=0)
            delta = class_mean - overall
            between += len(group) * np.outer(delta, delta)
            centered = group - class_mean
            within += centered.T @ centered
    return between, within


class MDAOptimizationTests(unittest.TestCase):
    def test_mode_scatter_matches_loop_equations_for_both_modes(self):
        rng = np.random.default_rng(44)
        projected = rng.normal(size=(15, 6, 5))
        labels = np.repeat(np.array([3, 7, 11]), 5)
        grouping = experiment.make_class_grouping(labels)
        for mode in (0, 1):
            with self.subTest(mode=mode):
                expected_between, expected_within = reference_mode_scatter(
                    projected, labels, mode
                )
                actual_between, actual_within = experiment.mode_scatter(
                    projected, labels, mode, grouping=grouping
                )
                np.testing.assert_allclose(
                    actual_between, expected_between, rtol=1e-12, atol=1e-12
                )
                np.testing.assert_allclose(
                    actual_within, expected_within, rtol=1e-12, atol=1e-12
                )

    def test_first_iteration_cache_keeps_direct_identity_contraction(self):
        rng = np.random.default_rng(441)
        train_images = rng.normal(size=(12, 7, 5))
        labels = np.repeat(np.array([0, 1, 2]), 4)
        identity_col = np.eye(train_images.shape[2], dtype=np.float64)
        expected = np.einsum(
            "nhw,wb->nhb", train_images, identity_col, optimize=True
        )
        original_mode_scatter = experiment.mode_scatter

        with patch.object(
            experiment, "mode_scatter", wraps=original_mode_scatter
        ) as scatter_spy:
            experiment.precompute_mda22_first_iteration(
                train_images,
                labels,
                [(2, 2), (3, 1)],
                projection_normalization="unit_l2",
            )

        first_row_input = scatter_spy.call_args_list[0].args[0]
        np.testing.assert_array_equal(first_row_input, expected)

    def test_reused_first_iteration_preserves_learned_projections(self):
        rng = np.random.default_rng(45)
        train_images = rng.normal(size=(15, 6, 5))
        labels = np.repeat(np.array([0, 1, 2]), 5)
        grouping = experiment.make_class_grouping(labels)
        pairs = [(2, 1), (2, 2), (3, 2)]
        probes = rng.normal(size=(5, 6, 5))

        for normalization in ("unit_l2", "within_scatter"):
            cache = experiment.precompute_mda22_first_iteration(
                train_images,
                labels,
                pairs,
                projection_normalization=normalization,
                scatter_grouping=grouping,
            )
            for row_dim, col_dim in pairs:
                with self.subTest(normalization=normalization, pair=(row_dim, col_dim)):
                    direct = experiment.fit_mda_projections(
                        train_images,
                        labels,
                        components=(row_dim, col_dim),
                        max_iter=5,
                        tolerance=0.0,
                        projection_normalization=normalization,
                        stopping_rule="paper_figure",
                        mode_order="row_first",
                        paper_epsilon=1e-4,
                        track_objective_history=False,
                        scatter_grouping=grouping,
                    )
                    initial_row, col_eigensystem = cache[row_dim]
                    initial_col = experiment.project_generalized_eigensystem(
                        col_eigensystem,
                        col_dim,
                        projection_normalization=normalization,
                    )
                    reused = experiment.fit_mda_projections(
                        train_images,
                        labels,
                        components=(row_dim, col_dim),
                        max_iter=5,
                        tolerance=0.0,
                        projection_normalization=normalization,
                        stopping_rule="paper_figure",
                        mode_order="row_first",
                        paper_epsilon=1e-4,
                        track_objective_history=False,
                        scatter_grouping=grouping,
                        initial_projections=(initial_row, initial_col),
                        initial_iterations=1,
                        initial_history=[float("nan")],
                    )
                    # Eigenvector signs are arbitrary; align each axis before
                    # comparing the learned basis. The tolerance also allows
                    # normal float64 reduction-order differences in the cache.
                    for direct_basis, cached_basis in zip(direct[:2], reused[:2]):
                        signs = np.sign(np.sum(direct_basis * cached_basis, axis=0))
                        signs[signs == 0] = 1.0
                        np.testing.assert_allclose(
                            direct_basis,
                            cached_basis * signs[None, :],
                            rtol=1e-7,
                            atol=1e-8,
                        )
                    self.assertEqual(len(direct[2]), len(reused[2]))
                    direct_train = experiment.apply_mda(train_images, *direct[:2])
                    reused_train = experiment.apply_mda(train_images, *reused[:2])
                    direct_probe = experiment.apply_mda(probes, *direct[:2])
                    reused_probe = experiment.apply_mda(probes, *reused[:2])

                    def squared_distances(train_features, probe_features):
                        train_sq = np.sum(train_features * train_features, axis=1)
                        probe_sq = np.sum(probe_features * probe_features, axis=1)
                        return np.maximum(
                            probe_sq[:, None]
                            + train_sq[None, :]
                            - 2.0 * probe_features @ train_features.T,
                            0.0,
                        )

                    np.testing.assert_allclose(
                        squared_distances(direct_train, direct_probe),
                        squared_distances(reused_train, reused_probe),
                        rtol=1e-7,
                        atol=1e-8,
                    )
                    self.assertTrue(
                        np.array_equal(
                            experiment.predict_nearest_neighbor(
                                direct_train, labels, direct_probe
                            ),
                            experiment.predict_nearest_neighbor(
                                reused_train, labels, reused_probe
                            ),
                        )
                    )
                    self.assertAlmostEqual(
                        experiment.fisher_trace_ratio(direct_train, labels),
                        experiment.fisher_trace_ratio(reused_train, labels),
                        delta=1e-8,
                    )

    def test_probe_images_do_not_change_training_fit(self):
        rng = np.random.default_rng(46)
        train_images = rng.normal(size=(15, 6, 5))
        labels = np.repeat(np.array([0, 1, 2]), 5)
        probe_a = rng.normal(size=(4, 6, 5))
        probe_b = rng.normal(size=(4, 6, 5)) + 10.0
        first = experiment.fit_mda(
            train_images, labels, probe_a, (2, 2), 4, 0.0, "unit_l2",
            stopping_rule="paper_figure", track_objective_history=False,
        )
        second = experiment.fit_mda(
            train_images, labels, probe_b, (2, 2), 4, 0.0, "unit_l2",
            stopping_rule="paper_figure", track_objective_history=False,
        )
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(len(first[2]), len(second[2]))
        self.assertTrue(np.isnan(first[2][:-1]).all())
        self.assertEqual(first[2][-1], second[2][-1])
        self.assertFalse(np.array_equal(first[1], second[1]))

    def test_pgm_reader_handles_p2_p5_and_sixteen_bit_rasters(self):
        p2 = b"P2\n# small fixture\n3 2\n255\n0 10 20\n30 40 50\n"
        np.testing.assert_array_equal(
            read_pgm(p2), np.array([[0, 10, 20], [30, 40, 50]], dtype=np.uint8)
        )

        p5 = b"P5\n2 2\n255\n" + bytes([0, 128, 254, 255])
        np.testing.assert_array_equal(
            read_pgm(p5), np.array([[0, 128], [254, 255]], dtype=np.uint8)
        )

        p5_16 = b"P5\n2 1\n1023\n" + bytes([0, 0, 1, 0])
        np.testing.assert_array_equal(
            read_pgm(p5_16), np.array([[0, 64]], dtype=np.uint8)
        )


if __name__ == "__main__":
    unittest.main()
