#!/usr/bin/env python3
"""Small, reproducible ORL-style study of MDA against face-recognition baselines.

The implementation follows the alternating k-mode procedure in Yan et al. (2007).
It uses the 64x64 Olivetti copy supplied by scikit-learn, not the paper's original
112x92 preprocessing, so its results are a partial reproduction rather than an
exact replication of the published table.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.datasets import fetch_olivetti_faces


PAPER_URL = "https://mmlab.ie.cuhk.edu.hk/archive/2007/IP07_face02.pdf"
METHODS = ("Eigenfaces PCA", "Fisherfaces LDA", "MDA/2-1", "MDA/2-2")


def parse_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes or any(size < 2 or size > 9 for size in sizes):
        raise argparse.ArgumentTypeError("Use comma-separated train counts from 2 to 9.")
    return sorted(set(sizes))


def load_faces(data_home: Path) -> tuple[np.ndarray, np.ndarray]:
    bunch = fetch_olivetti_faces(
        data_home=str(data_home),
        shuffle=False,
        download_if_missing=True,
    )
    images = equalize_histogram(np.asarray(bunch.images, dtype=np.float64))
    labels = np.asarray(bunch.target, dtype=np.int64)
    if images.ndim != 3 or images.shape[0] != 400 or len(np.unique(labels)) != 40:
        raise ValueError(f"Unexpected Olivetti shape/classes: {images.shape}")
    return images, labels


def equalize_histogram(images: np.ndarray) -> np.ndarray:
    """Apply per-image 8-bit histogram equalization as stated in the paper."""
    quantized = np.clip(np.rint(images * 255.0), 0, 255).astype(np.uint8)
    output = np.empty_like(quantized)
    pixel_count = quantized.shape[1] * quantized.shape[2]
    for index, image in enumerate(quantized):
        histogram = np.bincount(image.ravel(), minlength=256)
        cdf = histogram.cumsum()
        occupied = np.flatnonzero(histogram)
        if len(occupied) <= 1:
            output[index] = image
            continue
        cdf_min = int(cdf[occupied[0]])
        denominator = max(pixel_count - cdf_min, 1)
        lookup = np.clip(np.rint((cdf - cdf_min) * 255.0 / denominator), 0, 255).astype(np.uint8)
        output[index] = lookup[image]
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


def leading_generalized_eigenvectors(
    between: np.ndarray,
    within: np.ndarray,
    n_components: int,
    ridge: float = 1e-6,
    projection_normalization: str = "unit_l2",
) -> np.ndarray:
    """Solve Sb v = lambda Sw v with explicit projection-axis scaling."""
    between = (between + between.T) * 0.5
    within = (within + within.T) * 0.5
    dimension = within.shape[0]
    n_components = min(int(n_components), dimension)
    if n_components < 1:
        raise ValueError("At least one projection component is required.")

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
    order = np.argsort(values)[::-1][:n_components]
    projection = whitening @ vectors[:, order]
    if projection_normalization == "unit_l2":
        norms = np.linalg.norm(projection, axis=0)
        projection /= np.maximum(norms, 1e-12)[None, :]
    elif projection_normalization != "within_scatter":
        raise ValueError(f"Unknown projection normalization: {projection_normalization}")
    return projection


def mode_scatter(projected: np.ndarray, labels: np.ndarray, mode: int) -> tuple[np.ndarray, np.ndarray]:
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
    between = np.zeros((dimension, dimension), dtype=np.float64)
    within = np.zeros_like(between)
    class_labels = np.unique(labels)

    for vectors in coordinate_samples:
        overall = vectors.mean(axis=0)
        for label in class_labels:
            group = vectors[labels == label]
            class_mean = group.mean(axis=0)
            delta = class_mean - overall
            between += len(group) * np.outer(delta, delta)
            centered = group - class_mean
            within += centered.T @ centered
    return between, within


def apply_mda(images: np.ndarray, row_projection: np.ndarray, col_projection: np.ndarray) -> np.ndarray:
    projected = np.einsum("ah,nhw,wb->nab", row_projection.T, images, col_projection, optimize=True)
    return projected.reshape(len(images), -1)


def fit_mda(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    components: tuple[int | None, int | None],
    max_iter: int,
    tolerance: float,
    projection_normalization: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    height, width = train_images.shape[1:]
    row_projection = np.eye(height, dtype=np.float64)
    col_projection = np.eye(width, dtype=np.float64)
    history: list[float] = []

    for _iteration in range(max_iter):
        if components[0] is not None:
            partially_projected = np.einsum(
                "nhw,wb->nhb", train_images, col_projection, optimize=True
            )
            between, within = mode_scatter(partially_projected, train_labels, mode=0)
            row_projection = leading_generalized_eigenvectors(
                between, within, components[0], projection_normalization=projection_normalization
            )

        if components[1] is not None:
            partially_projected = np.einsum(
                "ah,nhw->naw", row_projection.T, train_images, optimize=True
            )
            between, within = mode_scatter(partially_projected, train_labels, mode=1)
            col_projection = leading_generalized_eigenvectors(
                between, within, components[1], projection_normalization=projection_normalization
            )

        embeddings = apply_mda(train_images, row_projection, col_projection)
        between, within = scatter_matrices(embeddings, train_labels)
        objective = float(np.trace(between) / max(float(np.trace(within)), 1e-12))
        history.append(objective)
        if len(history) > 1:
            change = abs(history[-1] - history[-2]) / max(abs(history[-2]), 1e-12)
            if change < tolerance:
                break

    return (
        apply_mda(train_images, row_projection, col_projection),
        apply_mda(test_images, row_projection, col_projection),
        history,
    )


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
) -> tuple[np.ndarray, np.ndarray]:
    train = train_images.reshape(len(train_images), -1)
    test = test_images.reshape(len(test_images), -1)
    mean = train.mean(axis=0)
    centered = train - mean
    classes = np.unique(train_labels)
    pca_dimensions = min(100, len(train) - len(classes), centered.shape[1])
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
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if name == "Eigenfaces PCA":
        train, test = fit_pca(train_images, test_images)
        return train, test, []
    if name == "Fisherfaces LDA":
        train, test = fit_fisherfaces(
            train_images, train_labels, test_images,
            projection_normalization=projection_normalization,
        )
        return train, test, []
    if name == "MDA/2-1":
        train, test, history = fit_mda(
            train_images, train_labels, test_images,
            components=(mda_dim, None), max_iter=max_iter, tolerance=tolerance,
            projection_normalization=projection_normalization,
        )
        return train, test, history
    if name == "MDA/2-2":
        train, test, history = fit_mda(
            train_images, train_labels, test_images,
            components=(mda_dim, mda_dim), max_iter=max_iter, tolerance=tolerance,
            projection_normalization=projection_normalization,
        )
        return train, test, history
    raise ValueError(f"Unknown method: {name}")


def write_svg(summary: list[dict], output: Path) -> None:
    width, height = 820, 470
    left, right, top, bottom = 78, 24, 32, 72
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

    legend_x, legend_y = left + 6, height - 27
    for i, method in enumerate(methods):
        x = legend_x + (i % 2) * 260
        y = legend_y + (i // 2) * 21
        parts.append(f'<line x1="{x}" y1="{y-4}" x2="{x+22}" y2="{y-4}" stroke="{colors[method]}" stroke-width="3"/>')
        parts.append(f'<text x="{x+29}" y="{y}" font-family="Arial" font-size="12" fill="#263746">{method}</text>')

    parts.extend([
        f'<text x="{left+plot_w/2:.1f}" y="{height-8}" text-anchor="middle" font-family="Arial" font-size="13" fill="#263746">Training images per identity</text>',
        '<text transform="translate(20 220) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13" fill="#263746">Test accuracy</text>',
        '</svg>',
    ])
    output.write_text("\n".join(parts), encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    images, labels = load_faces(args.data_home)
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
                train_features, test_features, history = method_features(
                    method, x_train, y_train, x_test,
                    mda_dim=args.mda_dim,
                    max_iter=args.max_iter,
                    tolerance=args.tolerance,
                    projection_normalization=args.projection_normalization,
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=Path(".cache/olivetti"), help="Local scikit-learn dataset cache.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for CSV, JSON, and SVG results.")
    parser.add_argument("--train-per-class", type=parse_sizes, default=parse_sizes("2,3,4,5"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3314)
    parser.add_argument("--mda-dim", type=int, default=6, help="Projection dimension per retained MDA mode.")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--projection-normalization",
        choices=("unit_l2", "within_scatter"),
        default="unit_l2",
        help="Scaling convention for generalized-eigenvector axes; 1-NN is scale-sensitive.",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.mda_dim < 1 or args.max_iter < 1:
        parser.error("repeats, mda-dim, and max-iter must be positive")

    args.data_home.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, summary, objective_rows = run(args)
    write_csv(args.output_dir / "raw_results.csv", raw_rows, list(raw_rows[0].keys()))
    write_csv(args.output_dir / "summary.csv", summary, list(summary[0].keys()))
    if objective_rows:
        write_csv(args.output_dir / "mda_objectives.csv", objective_rows, list(objective_rows[0].keys()))
    metadata = {
        "paper": "Yan et al., Multilinear Discriminant Analysis for Face Recognition (IEEE TIP, 2007)",
        "paper_url": PAPER_URL,
        "dataset": "Olivetti faces (AT&T ORL copy fetched by scikit-learn)",
        "images": 400,
        "identities": 40,
        "image_shape": [64, 64],
        "original_orl_shape_in_paper": [112, 92],
        "protocol": {
            "train_images_per_identity": args.train_per_class,
            "repeated_stratified_splits": args.repeats,
            "seed_rule": "base seed + 1000 * train_images_per_identity + repeat_index",
            "classification": "Euclidean 1-nearest-neighbour in the learned feature space",
            "preprocessing": "Per-image 8-bit histogram equalization, matching the paper's stated preprocessing; scikit-learn 64x64 images are used. No test data are used to fit projections.",
        },
        "models": [
            {"name": "Eigenfaces PCA", "feature_dimensions": 36},
            {"name": "Fisherfaces LDA", "feature_dimensions": "up to 36 after PCA pre-reduction"},
            {"name": "MDA/2-1", "feature_dimensions": f"{args.mda_dim} x 64"},
            {"name": "MDA/2-2", "feature_dimensions": f"{args.mda_dim} x {args.mda_dim}"},
        ],
        "mda": {
            "initialization": "identity projections, as in the paper's algorithm figure",
            "max_iterations": args.max_iter,
            "relative_objective_tolerance": args.tolerance,
            "ridge_for_singular_scatter": 1e-6,
            "projection_normalization": args.projection_normalization,
        },
        "results_status": "Group-computed result; partial reproduction, not a direct reproduction of the paper's original preprocessing or best-tuned table.",
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
