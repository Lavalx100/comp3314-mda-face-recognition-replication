#!/usr/bin/env python3
"""Reproducible ORL face-recognition study of MDA and standard baselines.

The implementation follows the alternating k-mode procedure in Yan et al. (2007)
and supports both scikit-learn's historical 64x64 copy and the original 112x92
Cambridge PGM archive. The original-resolution runs are paper-style follow-ups,
not exact reproductions: the paper omits its split seeds and full dimension search.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

# On macOS, NumPy uses Accelerate; set vecLib before loading NumPy because
# threadpoolctl cannot currently inspect/control that backend.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from orl_data import ORL_ARCHIVE_URL, ORL_CITATION, load_original_orl


PAPER_URL = "https://mmlab.ie.cuhk.edu.hk/archive/2007/IP07_face02.pdf"
METHODS = ("Eigenfaces PCA", "Fisherfaces LDA", "MDA/2-1", "MDA/2-2")


def parse_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes or any(size < 2 or size > 9 for size in sizes):
        raise argparse.ArgumentTypeError("Use comma-separated train counts from 2 to 9.")
    return sorted(set(sizes))


def load_faces(data_home: Path, histogram_levels: int = 256) -> tuple[np.ndarray, np.ndarray]:
    bunch = fetch_olivetti_faces(
        data_home=str(data_home),
        shuffle=False,
        download_if_missing=True,
    )
    images = equalize_histogram(np.asarray(bunch.images, dtype=np.float64), histogram_levels)
    labels = np.asarray(bunch.target, dtype=np.int64)
    if images.ndim != 3 or images.shape[0] != 400 or len(np.unique(labels)) != 40:
        raise ValueError(f"Unexpected Olivetti shape/classes: {images.shape}")
    return images, labels


def load_experiment_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.dataset == "orl-original":
        images, labels = load_original_orl(args.original_orl_archive)
        return equalize_histogram(images, args.histogram_levels), labels
    return load_faces(args.data_home, args.histogram_levels)


def equalize_histogram(images: np.ndarray, output_levels: int = 256) -> np.ndarray:
    """Equalize each image independently and quantize to the requested levels.

    The paper only says that histogram equalization was applied. The 64-level
    option approximates MATLAB ``histeq(I)``'s documented default; 256 preserves
    the earlier project protocol. The article does not identify the software or
    equalization settings used by its authors.
    """
    if not 2 <= output_levels <= 256:
        raise ValueError("Histogram equalization output_levels must be between 2 and 256.")
    quantized = np.clip(np.rint(images * 255.0), 0, 255).astype(np.uint8)
    output = np.empty_like(quantized)
    pixel_count = quantized.shape[1] * quantized.shape[2]
    output_values = np.rint(np.linspace(0, 255, output_levels)).astype(np.uint8)
    for index, image in enumerate(quantized):
        histogram = np.bincount(image.ravel(), minlength=256)
        cdf = histogram.cumsum()
        occupied = np.flatnonzero(histogram)
        if len(occupied) <= 1:
            output[index] = image
            continue
        cdf_min = int(cdf[occupied[0]])
        denominator = max(pixel_count - cdf_min, 1)
        cdf_fraction = np.clip((cdf - cdf_min) / denominator, 0.0, 1.0)
        lookup = np.rint(cdf_fraction * (output_levels - 1)).astype(np.int64)
        output[index] = output_values[lookup[image]]
    return output.astype(np.float64) / 255.0


def stratified_split(labels: np.ndarray, train_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if train_per_class >= len(indices):
            raise ValueError(f"Class {label} has only {len(indices)} images.")
        shuffled = rng.permutation(indices)
        train.extend(shuffled[:train_per_class].tolist())
        test.extend(shuffled[train_per_class:].tolist())
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def scatter_matrices(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Class-weighted between-class and pooled within-class scatter matrices."""
    features = np.asarray(features, dtype=np.float64)
    overall = features.mean(axis=0)
    between = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    within = np.zeros_like(between)
    for label in np.unique(labels):
        group = features[labels == label]
        class_mean = group.mean(axis=0)
        delta = class_mean - overall
        between += len(group) * np.outer(delta, delta)
        centered = group - class_mean
        within += centered.T @ centered
    return between, within


def fisher_trace_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute tr(Sb)/tr(Sw) without constructing a feature-by-feature matrix."""
    features = np.asarray(features, dtype=np.float64)
    overall = features.mean(axis=0)
    between_trace = 0.0
    within_trace = 0.0
    for label in np.unique(labels):
        group = features[labels == label]
        class_mean = group.mean(axis=0)
        delta = class_mean - overall
        between_trace += len(group) * float(delta @ delta)
        centered = group - class_mean
        within_trace += float(np.sum(centered * centered))
    return between_trace / max(within_trace, 1e-12)


def prepare_generalized_eigensystem(
    between: np.ndarray,
    within: np.ndarray,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor the symmetric, ridge-regularized generalized eigenproblem."""
    between = (between + between.T) * 0.5
    within = (within + within.T) * 0.5
    dimension = within.shape[0]

    sw_values, sw_vectors = np.linalg.eigh(within)
    scale = max(float(np.trace(within)) / max(dimension, 1), 1e-12)
    floor = ridge * scale
    # Whiten a regularized within-class scatter before solving the symmetric
    # eigenproblem. Nearest-neighbour distances depend on each axis's scale,
    # so expose the generalized-eigenvector scaling convention.
    regularized_values = np.maximum(sw_values, 0.0) + floor
    whitening = sw_vectors * (1.0 / np.sqrt(regularized_values))[None, :]
    whitened_between = whitening.T @ between @ whitening
    values, vectors = np.linalg.eigh((whitened_between + whitened_between.T) * 0.5)
    order = np.argsort(values)[::-1]
    return whitening, vectors, order


def project_generalized_eigensystem(
    eigensystem: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_components: int,
    projection_normalization: str = "unit_l2",
) -> np.ndarray:
    """Return leading axes using the same shape-specific multiply as a direct solve."""
    whitening, vectors, order = eigensystem
    dimension = whitening.shape[0]
    n_components = min(int(n_components), dimension)
    if n_components < 1:
        raise ValueError("At least one projection component is required.")
    # Keep the column slice before this matrix multiply. Multiplying a full
    # basis and truncating later can change rounding in nearly singular cases.
    projection = whitening @ vectors[:, order[:n_components]]
    if projection_normalization == "unit_l2":
        norms = np.linalg.norm(projection, axis=0)
        projection /= np.maximum(norms, 1e-12)[None, :]
    elif projection_normalization != "within_scatter":
        raise ValueError(f"Unknown projection normalization: {projection_normalization}")
    return projection


def leading_generalized_eigenvectors(
    between: np.ndarray,
    within: np.ndarray,
    n_components: int,
    ridge: float = 1e-6,
    projection_normalization: str = "unit_l2",
) -> np.ndarray:
    """Solve Sb v = lambda Sw v with explicit projection-axis scaling."""
    eigensystem = prepare_generalized_eigensystem(between, within, ridge=ridge)
    return project_generalized_eigensystem(
        eigensystem, n_components, projection_normalization=projection_normalization
    )


@dataclass(frozen=True)
class ClassGrouping:
    """Label-derived arrays reused by every scatter update in one split."""

    class_labels: np.ndarray
    class_index: np.ndarray
    class_counts: np.ndarray
    membership: np.ndarray


def make_class_grouping(labels: np.ndarray) -> ClassGrouping:
    class_labels, class_index = np.unique(labels, return_inverse=True)
    class_counts = np.bincount(class_index).astype(np.float64)
    membership = np.eye(len(class_labels), dtype=np.float64)[class_index]
    return ClassGrouping(class_labels, class_index, class_counts, membership)


def mode_scatter(
    projected: np.ndarray,
    labels: np.ndarray,
    mode: int,
    grouping: ClassGrouping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool the LDA scatters of matching columns in a mode-unfolded tensor.

    For mode 0, each column of each image contributes a row-mode vector. For
    mode 1, each row after the mode-0 projection contributes a column-mode
    vector. Means are computed separately for each matching coordinate, as in
    the paper's k-mode derivation.
    """
    if mode == 0:
        coordinate_samples = np.transpose(projected, (2, 0, 1))
    elif mode == 1:
        coordinate_samples = np.transpose(projected, (1, 0, 2))
    else:
        raise ValueError("MDA supports image modes 0 and 1 in this implementation.")

    dimension = coordinate_samples.shape[-1]
    grouping = grouping or make_class_grouping(labels)
    class_labels = grouping.class_labels
    class_index = grouping.class_index
    class_counts = grouping.class_counts
    overall = coordinate_samples.mean(axis=1)
    grouped_samples = coordinate_samples.transpose(1, 0, 2).reshape(len(labels), -1)
    class_sums = (grouping.membership.T @ grouped_samples).reshape(
        len(class_labels), coordinate_samples.shape[0], dimension
    )
    class_means = class_sums / class_counts[:, None, None]

    mean_offsets = class_means - overall[None, :, :]
    between = np.einsum(
        "c,cki,ckj->ij", class_counts, mean_offsets, mean_offsets, optimize=True
    )
    centered = coordinate_samples - class_means[class_index].transpose(1, 0, 2)
    within = np.einsum("kni,knj->ij", centered, centered, optimize=True)
    return between, within


def apply_mda(images: np.ndarray, row_projection: np.ndarray, col_projection: np.ndarray) -> np.ndarray:
    projected = np.einsum("ah,nhw,wb->nab", row_projection.T, images, col_projection, optimize=True)
    return projected.reshape(len(images), -1)


def fit_mda_projections(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    components: tuple[int | None, int | None],
    max_iter: int,
    tolerance: float,
    projection_normalization: str,
    stopping_rule: str = "objective",
    mode_order: str = "row_first",
    paper_epsilon: float = 1e-4,
    track_objective_history: bool = True,
    scatter_grouping: ClassGrouping | None = None,
    initial_projections: tuple[np.ndarray, np.ndarray] | None = None,
    initial_iterations: int = 0,
    initial_history: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Learn MDA axes; optional initial state can reuse a validated prefix."""
    height, width = train_images.shape[1:]
    if initial_projections is None:
        row_projection = np.eye(height, dtype=np.float64)
        col_projection = np.eye(width, dtype=np.float64)
    else:
        row_projection, col_projection = initial_projections
        if row_projection.shape[0] != height or col_projection.shape[0] != width:
            raise ValueError("Initial MDA projections do not match the input image shape.")
    if not 0 <= initial_iterations <= max_iter:
        raise ValueError("initial_iterations must be between zero and max_iter.")
    history = list(initial_history) if initial_history is not None else [float("nan")] * initial_iterations
    if len(history) != initial_iterations:
        raise ValueError("initial_history must contain one entry for each reused iteration.")
    if mode_order not in {"row_first", "column_first"}:
        raise ValueError(f"Unknown MDA mode-update order: {mode_order}")
    if paper_epsilon <= 0:
        raise ValueError("paper_epsilon must be positive.")
    if stopping_rule == "objective" and not track_objective_history:
        raise ValueError("Objective-based stopping requires objective history tracking.")
    if stopping_rule == "objective" and initial_iterations and not np.isfinite(history[-1]):
        raise ValueError("Objective-based stopping needs the objective for the reused prefix.")
    scatter_grouping = scatter_grouping or make_class_grouping(train_labels)
    mode_sequence = (0, 1) if mode_order == "row_first" else (1, 0)

    for iteration in range(initial_iterations, max_iter):
        previous_row = row_projection.copy()
        previous_col = col_projection.copy()
        for mode in mode_sequence:
            if mode == 0 and components[0] is not None:
                partially_projected = np.einsum(
                    "nhw,wb->nhb", train_images, col_projection, optimize=True
                )
                between, within = mode_scatter(partially_projected, train_labels, mode=0, grouping=scatter_grouping)
                row_projection = leading_generalized_eigenvectors(
                    between, within, components[0], projection_normalization=projection_normalization
                )
            elif mode == 1 and components[1] is not None:
                partially_projected = np.einsum(
                    "ah,nhw->naw", row_projection.T, train_images, optimize=True
                )
                between, within = mode_scatter(partially_projected, train_labels, mode=1, grouping=scatter_grouping)
                col_projection = leading_generalized_eigenvectors(
                    between, within, components[1], projection_normalization=projection_normalization
                )

        if track_objective_history:
            embeddings = apply_mda(train_images, row_projection, col_projection)
            # The output may have thousands of features during a dimension sweep.
            # Only the scatter traces are needed for this diagnostic; allocating the
            # full square scatter matrices would be wasteful and scale quadratically.
            history.append(fisher_trace_ratio(embeddings, train_labels))
        else:
            # Projection-change stopping does not use the objective. Keep one
            # placeholder per iteration so len(history) remains the iteration count.
            history.append(float("nan"))
        if stopping_rule == "objective" and len(history) > 1:
            change = abs(history[-1] - history[-2]) / max(abs(history[-2]), 1e-12)
            if change < tolerance:
                break
        elif stopping_rule == "projection_change" and iteration + 1 > 2:
            changes = []
            if components[0] is not None:
                changes.append(
                    np.linalg.norm(row_projection - previous_row, ord="fro")
                    < max(height, 1) * tolerance
                )
            if components[1] is not None:
                changes.append(
                    np.linalg.norm(col_projection - previous_col, ord="fro")
                    < max(width, 1) * tolerance
                )
            if changes and all(changes):
                break
        elif stopping_rule == "paper_figure" and iteration + 1 > 2:
            # Fig. 3 uses ||U_k^t-U_k^(t-1)||_F < m'_k*m_k*epsilon.
            # The paper does not report epsilon; record the selected value.
            changes = []
            if components[0] is not None:
                changes.append(
                    np.linalg.norm(row_projection - previous_row, ord="fro")
                    < components[0] * height * paper_epsilon
                )
            if components[1] is not None:
                changes.append(
                    np.linalg.norm(col_projection - previous_col, ord="fro")
                    < components[1] * width * paper_epsilon
                )
            if changes and all(changes):
                break
        elif stopping_rule not in {"objective", "projection_change", "paper_figure"}:
            raise ValueError(f"Unknown MDA stopping rule: {stopping_rule}")

    return row_projection, col_projection, history


def fit_mda(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    components: tuple[int | None, int | None],
    max_iter: int,
    tolerance: float,
    projection_normalization: str,
    stopping_rule: str = "objective",
    mode_order: str = "row_first",
    paper_epsilon: float = 1e-4,
    track_objective_history: bool = True,
    scatter_grouping: ClassGrouping | None = None,
    initial_projections: tuple[np.ndarray, np.ndarray] | None = None,
    initial_iterations: int = 0,
    initial_history: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    row_projection, col_projection, history = fit_mda_projections(
        train_images,
        train_labels,
        components=components,
        max_iter=max_iter,
        tolerance=tolerance,
        projection_normalization=projection_normalization,
        stopping_rule=stopping_rule,
        mode_order=mode_order,
        paper_epsilon=paper_epsilon,
        track_objective_history=track_objective_history,
        scatter_grouping=scatter_grouping,
        initial_projections=initial_projections,
        initial_iterations=initial_iterations,
        initial_history=initial_history,
    )
    train_features = apply_mda(train_images, row_projection, col_projection)
    if not track_objective_history:
        # In full-grid runs this final diagnostic is recorded but never selects
        # dimensions or affects the paper's projection-change stopping rule.
        if not history:
            history.append(float("nan"))
        history[-1] = fisher_trace_ratio(train_features, train_labels)
    test_features = apply_mda(test_images, row_projection, col_projection)
    return train_features, test_features, history


def precompute_mda22_first_iteration(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    dimension_pairs: Iterable[tuple[int, int]],
    projection_normalization: str,
    scatter_grouping: ClassGrouping | None = None,
) -> dict[int, tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Reuse MDA's first row-then-column update for every feasible dimension pair.

    At identity initialization, the first row scatter is independent of the
    requested output dimensions. For each row dimension, the following column
    scatter is independent of requested column dimension. Cache each scatter's
    eigensystem, then form the requested number of axes with the same
    shape-specific matrix multiply as a direct fit. This avoids repeated
    scatter/eigensolver work without changing the first-step operation order.
    """
    pairs_by_row: dict[int, list[int]] = defaultdict(list)
    for row_dim, col_dim in dimension_pairs:
        pairs_by_row[row_dim].append(col_dim)
    if not pairs_by_row:
        return {}
    height, width = train_images.shape[1:]
    grouping = scatter_grouping or make_class_grouping(train_labels)
    # Match the direct solver's identity-initialized contraction exactly.
    # Skipping this algebraic no-op can change rounding in singular or nearly
    # singular scatter problems, which may rotate tied generalized eigenvectors
    # and alter the projection-change stopping iteration.
    identity_col = np.eye(width, dtype=np.float64)
    identity_projected = np.einsum(
        "nhw,wb->nhb", train_images, identity_col, optimize=True
    )
    row_between, row_within = mode_scatter(
        identity_projected, train_labels, mode=0, grouping=grouping
    )
    row_eigensystem = prepare_generalized_eigensystem(row_between, row_within)
    cache: dict[int, tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for row_dim, col_dims in pairs_by_row.items():
        row_projection = project_generalized_eigensystem(
            row_eigensystem,
            row_dim,
            projection_normalization=projection_normalization,
        )
        partially_projected = np.einsum(
            "ah,nhw->naw", row_projection.T, train_images, optimize=True
        )
        col_between, col_within = mode_scatter(
            partially_projected, train_labels, mode=1, grouping=grouping
        )
        col_eigensystem = prepare_generalized_eigensystem(col_between, col_within)
        cache[row_dim] = (row_projection, col_eigensystem)
    return cache


def fit_pca(train_images: np.ndarray, test_images: np.ndarray, n_components: int = 36) -> tuple[np.ndarray, np.ndarray]:
    train = train_images.reshape(len(train_images), -1)
    test = test_images.reshape(len(test_images), -1)
    mean = train.mean(axis=0)
    centered = train - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    dimensions = min(n_components, len(train) - 1, vt.shape[0])
    projection = vt[:dimensions].T
    return centered @ projection, (test - mean) @ projection


def fit_fisherfaces(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    n_components: int = 36,
    projection_normalization: str = "unit_l2",
    pca_components_max: int | None = 100,
) -> tuple[np.ndarray, np.ndarray]:
    train = train_images.reshape(len(train_images), -1)
    test = test_images.reshape(len(test_images), -1)
    mean = train.mean(axis=0)
    centered = train - mean
    classes = np.unique(train_labels)
    # Preserve the earlier study's 100-component ceiling by default. The
    # original-resolution sweep passes None to use the full N-C subspace.
    pca_dimensions = min(len(train) - len(classes), centered.shape[1])
    if pca_components_max is not None:
        pca_dimensions = min(pca_dimensions, pca_components_max)
    if pca_dimensions < 1:
        raise ValueError("Fisherfaces requires at least two training images per identity.")
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    pca_projection = vt[:pca_dimensions].T
    train_pca = centered @ pca_projection
    test_pca = (test - mean) @ pca_projection
    between, within = scatter_matrices(train_pca, train_labels)
    lda_dimensions = min(n_components, len(classes) - 1, pca_dimensions)
    lda_projection = leading_generalized_eigenvectors(
        between, within, lda_dimensions, projection_normalization=projection_normalization
    )
    return train_pca @ lda_projection, test_pca @ lda_projection


def predict_nearest_neighbor(train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    train_sq = np.sum(train_features * train_features, axis=1)
    test_sq = np.sum(test_features * test_features, axis=1)
    distances = test_sq[:, None] + train_sq[None, :] - 2.0 * test_features @ train_features.T
    nearest = np.argmin(np.maximum(distances, 0.0), axis=1)
    return train_labels[nearest]


def method_features(
    name: str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    mda_dim: int,
    max_iter: int,
    tolerance: float,
    projection_normalization: str,
    mda_stopping_rule: str = "objective",
    mda_mode_order: str = "row_first",
    paper_epsilon: float = 1e-4,
    pca_dim: int = 36,
    fisher_dim: int = 36,
    fisher_pca_max: int | None = 100,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if name == "Eigenfaces PCA":
        train, test = fit_pca(train_images, test_images, n_components=pca_dim)
        return train, test, []
    if name == "Fisherfaces LDA":
        train, test = fit_fisherfaces(
            train_images, train_labels, test_images,
            n_components=fisher_dim,
            projection_normalization=projection_normalization,
            pca_components_max=fisher_pca_max,
        )
        return train, test, []
    if name == "MDA/2-1":
        train, test, history = fit_mda(
            train_images, train_labels, test_images,
            components=(mda_dim, None), max_iter=max_iter, tolerance=tolerance,
            projection_normalization=projection_normalization, stopping_rule=mda_stopping_rule,
            mode_order=mda_mode_order, paper_epsilon=paper_epsilon,
        )
        return train, test, history
    if name == "MDA/2-2":
        train, test, history = fit_mda(
            train_images, train_labels, test_images,
            components=(mda_dim, mda_dim), max_iter=max_iter, tolerance=tolerance,
            projection_normalization=projection_normalization, stopping_rule=mda_stopping_rule,
            mode_order=mda_mode_order, paper_epsilon=paper_epsilon,
        )
        return train, test, history
    raise ValueError(f"Unknown method: {name}")


def write_svg(summary: list[dict], output: Path) -> None:
    width, height = 820, 520
    left, right, top, bottom = 78, 24, 32, 122
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = {
        "Eigenfaces PCA": "#537895",
        "Fisherfaces LDA": "#D67B43",
        "MDA/2-1": "#7C69A6",
        "MDA/2-2": "#16836D",
    }
    methods = list(METHODS)
    sizes = sorted({int(row["train_per_class"]) for row in summary})
    x_pos = {value: left + (i / max(len(sizes) - 1, 1)) * plot_w for i, value in enumerate(sizes)}
    y_pos = lambda value: top + (100.0 - value) / 100.0 * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Recognition accuracy by training images per person</title>',
        '<desc id="desc">Mean accuracy over repeated stratified ORL splits; whiskers show one standard deviation.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for value in range(0, 101, 20):
        y = y_pos(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e4ebef"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#51616d">{value}%</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#61717d"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#61717d"/>',
    ])
    for train_count, x in x_pos.items():
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="Arial" font-size="13" fill="#51616d">{train_count}</text>')

    for method in methods:
        points = []
        for train_count in sizes:
            row = next((r for r in summary if r["method"] == method and int(r["train_per_class"]) == train_count), None)
            if row is None:
                continue
            x, y = x_pos[train_count], y_pos(float(row["mean_accuracy"]) * 100.0)
            sd = float(row["std_accuracy"]) * 100.0
            y_hi, y_lo = y_pos(min(100.0, float(row["mean_accuracy"]) * 100.0 + sd)), y_pos(max(0.0, float(row["mean_accuracy"]) * 100.0 - sd))
            color = colors[method]
            parts.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{color}" stroke-width="1.6"/>')
            parts.append(f'<line x1="{x-5:.1f}" y1="{y_hi:.1f}" x2="{x+5:.1f}" y2="{y_hi:.1f}" stroke="{color}" stroke-width="1.6"/>')
            parts.append(f'<line x1="{x-5:.1f}" y1="{y_lo:.1f}" x2="{x+5:.1f}" y2="{y_lo:.1f}" stroke="{color}" stroke-width="1.6"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[method]}" stroke-width="2.7" stroke-linejoin="round" stroke-linecap="round"/>')

    legend_x, legend_y = left + 6, top + plot_h + 75
    for i, method in enumerate(methods):
        x = legend_x + (i % 2) * 260
        y = legend_y + (i // 2) * 21
        parts.append(f'<line x1="{x}" y1="{y-4}" x2="{x+22}" y2="{y-4}" stroke="{colors[method]}" stroke-width="3"/>')
        parts.append(f'<text x="{x+29}" y="{y}" font-family="Arial" font-size="12" fill="#263746">{method}</text>')

    parts.extend([
        f'<text x="{left+plot_w/2:.1f}" y="{top+plot_h+52}" text-anchor="middle" font-family="Arial" font-size="13" fill="#263746">Training images per identity</text>',
        '<text transform="translate(20 220) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13" fill="#263746">Test accuracy</text>',
        '</svg>',
    ])
    output.write_text("\n".join(parts), encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    images, labels = load_experiment_data(args)
    raw_rows: list[dict] = []
    objective_rows: list[dict] = []

    for train_count in args.train_per_class:
        for repeat in range(args.repeats):
            seed = args.seed + 1000 * train_count + repeat
            train_idx, test_idx = stratified_split(labels, train_count, seed)
            x_train, x_test = images[train_idx], images[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            for method in METHODS:
                start = time.perf_counter()
                # This path fits one fixed MDA/2-2 pair per split, so there is
                # no cross-dimension first-step work to reuse. The direct path
                # avoids cache setup; dimension sweeps use the validated cached
                # eigensystem path in scripts/run_orl_dimension_sweep.py.
                train_features, test_features, history = method_features(
                    method, x_train, y_train, x_test,
                    mda_dim=args.mda_dim,
                    max_iter=args.max_iter,
                    tolerance=args.tolerance,
                    projection_normalization=args.projection_normalization,
                    mda_stopping_rule=args.mda_stopping_rule,
                    mda_mode_order=args.mda_mode_order,
                    paper_epsilon=args.paper_epsilon,
                    pca_dim=args.pca_dim,
                    fisher_dim=args.fisher_dim,
                    fisher_pca_max=args.fisher_pca_max,
                )
                prediction = predict_nearest_neighbor(train_features, y_train, test_features)
                elapsed = time.perf_counter() - start
                accuracy = float(np.mean(prediction == y_test))
                raw_rows.append({
                    "train_per_class": train_count,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "method": method,
                    "accuracy": accuracy,
                    "fit_and_predict_seconds": elapsed,
                })
                if history:
                    for iteration, objective in enumerate(history, start=1):
                        objective_rows.append({
                            "train_per_class": train_count,
                            "repeat": repeat + 1,
                            "seed": seed,
                            "method": method,
                            "iteration": iteration,
                            "training_fisher_ratio": objective,
                        })
                print(f"train={train_count} repeat={repeat+1}/{args.repeats} {method}: {accuracy:.3f} ({elapsed:.2f}s)", flush=True)

    buckets: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in raw_rows:
        buckets[(int(row["train_per_class"]), str(row["method"]))].append(float(row["accuracy"]))
    summary = []
    for (train_count, method), values in sorted(buckets.items()):
        summary.append({
            "train_per_class": train_count,
            "method": method,
            "repeats": len(values),
            "mean_accuracy": float(np.mean(values)),
            "std_accuracy": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        })
    return raw_rows, summary, objective_rows


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("olivetti64", "orl-original"),
        default="olivetti64",
        help="Dataset copy; use orl-original for the Cambridge 112x92 PGM archive.",
    )
    parser.add_argument("--data-home", type=Path, default=Path(".cache/olivetti"), help="Local scikit-learn dataset cache.")
    parser.add_argument(
        "--original-orl-archive",
        type=Path,
        default=Path(".cache/orl-original/att_faces.zip"),
        help="Path to Cambridge's att_faces.zip; kept out of version control.",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory; defaults to a timestamped results/runs folder.")
    parser.add_argument("--train-per-class", type=parse_sizes, default=parse_sizes("2,3,4,5"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3314)
    parser.add_argument("--mda-dim", type=int, default=6, help="Projection dimension per retained MDA mode.")
    parser.add_argument("--pca-dim", type=int, default=36, help="Maximum retained Eigenfaces PCA dimension.")
    parser.add_argument("--fisher-dim", type=int, default=36, help="Maximum Fisherface LDA output dimension.")
    parser.add_argument("--fisher-pca-max", type=int, default=100, help="Fisherface PCA pre-reduction cap; use a large value for full N-C reduction.")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--mda-stopping-rule",
        choices=("objective", "projection_change", "paper_figure"),
        default="paper_figure",
        help="paper_figure uses the stopping threshold printed in Fig. 3; projection_change uses a stricter tolerance.",
    )
    parser.add_argument(
        "--mda-mode-order",
        choices=("row_first", "column_first"),
        default="row_first",
        help="Which image mode is optimized first in each MDA iteration.",
    )
    parser.add_argument(
        "--paper-epsilon",
        type=float,
        default=1e-4,
        help="Epsilon in the Fig. 3 projection-change rule; the article does not specify it.",
    )
    parser.add_argument(
        "--histogram-levels",
        type=int,
        default=64,
        help="Output intensity levels per image; 64 approximates MATLAB histeq's default.",
    )
    parser.add_argument(
        "--projection-normalization",
        choices=("unit_l2", "within_scatter"),
        default="unit_l2",
        help="Scaling convention for generalized-eigenvector axes; 1-NN is scale-sensitive.",
    )
    args = parser.parse_args()
    if min(args.repeats, args.mda_dim, args.pca_dim, args.fisher_dim, args.max_iter) < 1:
        parser.error("repeats and all feature dimensions/iteration limits must be positive")
    if not 2 <= args.histogram_levels <= 256:
        parser.error("histogram-levels must be between 2 and 256")
    if args.paper_epsilon <= 0:
        parser.error("paper-epsilon must be positive")

    if args.output_dir is None:
        timestamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d-%H%M%S")
        args.output_dir = Path("results/runs") / (
            f"{timestamp}-{args.dataset}-{args.projection_normalization}"
        )

    if args.dataset == "olivetti64":
        args.data_home.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, summary, objective_rows = run(args)
    write_csv(args.output_dir / "raw_results.csv", raw_rows, list(raw_rows[0].keys()))
    write_csv(args.output_dir / "summary.csv", summary, list(summary[0].keys()))
    if objective_rows:
        write_csv(args.output_dir / "mda_objectives.csv", objective_rows, list(objective_rows[0].keys()))
    original_orl = args.dataset == "orl-original"
    image_shape = [112, 92] if original_orl else [64, 64]
    metadata = {
        "paper": "Yan et al., Multilinear Discriminant Analysis for Face Recognition (IEEE TIP, 2007)",
        "paper_url": PAPER_URL,
        "dataset": "Original-resolution AT&T/ORL Database of Faces" if original_orl else "Olivetti faces (AT&T ORL copy fetched by scikit-learn)",
        "dataset_archive_url": ORL_ARCHIVE_URL if original_orl else "https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset",
        "dataset_credit": ORL_CITATION if original_orl else "AT&T Laboratories Cambridge; scikit-learn loader",
        "dataset_archive_sha256": hashlib.sha256(args.original_orl_archive.read_bytes()).hexdigest() if original_orl else "",
        "images": 400,
        "identities": 40,
        "image_shape_rows_columns": image_shape,
        "original_orl_shape_in_paper": [112, 92],
        "protocol": {
            "train_images_per_identity": args.train_per_class,
            "repeated_stratified_splits": args.repeats,
            "seed_rule": "base seed + 1000 * train_images_per_identity + repeat_index",
            "classification": "Euclidean 1-nearest-neighbour in the learned feature space",
            "preprocessing": f"Per-image 8-bit histogram equalization to {args.histogram_levels} output levels. Original PGM data are used." if original_orl else f"Per-image 8-bit histogram equalization to {args.histogram_levels} output levels on scikit-learn's 64x64 copy. No test data are used to fit projections.",
        },
        "models": [
            {"name": "Eigenfaces PCA", "feature_dimensions": args.pca_dim},
            {"name": "Fisherfaces LDA", "feature_dimensions": f"up to {args.fisher_dim} after PCA pre-reduction capped at {args.fisher_pca_max}"},
            {"name": "MDA/2-1", "feature_dimensions": f"{args.mda_dim} x {image_shape[1]}"},
            {"name": "MDA/2-2", "feature_dimensions": f"{args.mda_dim} x {args.mda_dim}"},
        ],
        "mda": {
            "initialization": "identity projections, as in the paper's algorithm figure",
            "max_iterations": args.max_iter,
            "stopping_rule": args.mda_stopping_rule,
            "mode_update_order": args.mda_mode_order,
            "tolerance": args.tolerance,
            "paper_epsilon": args.paper_epsilon,
            "ridge_for_singular_scatter": 1e-6,
            "projection_normalization": args.projection_normalization,
        },
        "results_status": (
            "Original-resolution follow-up: uses the paper's ORL image dimensions and stated preprocessing, "
            "but random seeds and complete feature-selection settings are not specified in the paper; this is "
            "not an exact reproduction of its best-tuned Table I."
            if original_orl else
            "Group-computed result; partial reproduction, not a direct reproduction of the paper's original preprocessing or best-tuned table."
        ),
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_svg(summary, args.output_dir / "accuracy_vs_training_size.svg")

    print("\nMean accuracy ± sample SD across repeated splits:")
    for row in summary:
        print(f"{row['method']:18} train={row['train_per_class']}: {row['mean_accuracy']*100:5.1f}% ± {row['std_accuracy']*100:4.1f}%")
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
