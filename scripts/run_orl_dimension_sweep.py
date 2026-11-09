#!/usr/bin/env python3
"""Run a paper-style feature-dimension sweep on the original 112x92 ORL faces.

The article does not publish its random seeds or complete sweep ranges. This
script therefore records a reproducible follow-up, not an exact reconstruction
of the authors' original run. Per-split probe scores are retained so that
dimension selection and its optimistic bias remain visible.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# NumPy on Apple Silicon uses Accelerate, which threadpoolctl does not expose.
# Set vecLib's default before NumPy loads; callers can override this in the
# process environment when benchmarking a different BLAS thread count.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orl_data import ORL_ARCHIVE_URL, ORL_CITATION, load_original_orl  # noqa: E402
from run_experiment import (  # noqa: E402
    PAPER_URL,
    equalize_histogram,
    fit_fisherfaces,
    fit_mda,
    fit_pca,
    make_class_grouping,
    precompute_mda22_first_iteration,
    predict_nearest_neighbor,
    stratified_split,
    write_csv,
)


PUBLISHED_TABLE = [
    {"paper_method": "Eigenface", "gallery_per_identity": 5, "probe_per_identity": 5, "accuracy_percent": 97.5},
    {"paper_method": "Eigenface", "gallery_per_identity": 4, "probe_per_identity": 6, "accuracy_percent": 91.3},
    {"paper_method": "Eigenface", "gallery_per_identity": 3, "probe_per_identity": 7, "accuracy_percent": 87.7},
    {"paper_method": "Eigenface", "gallery_per_identity": 2, "probe_per_identity": 8, "accuracy_percent": 81.6},
    {"paper_method": "Fisherface", "gallery_per_identity": 5, "probe_per_identity": 5, "accuracy_percent": 92.0},
    {"paper_method": "Fisherface", "gallery_per_identity": 4, "probe_per_identity": 6, "accuracy_percent": 85.4},
    {"paper_method": "Fisherface", "gallery_per_identity": 3, "probe_per_identity": 7, "accuracy_percent": 87.5},
    {"paper_method": "Fisherface", "gallery_per_identity": 2, "probe_per_identity": 8, "accuracy_percent": 74.7},
    {"paper_method": "Fisherface/O", "gallery_per_identity": 5, "probe_per_identity": 5, "accuracy_percent": 92.0},
    {"paper_method": "Fisherface/O", "gallery_per_identity": 4, "probe_per_identity": 6, "accuracy_percent": 85.8},
    {"paper_method": "Fisherface/O", "gallery_per_identity": 3, "probe_per_identity": 7, "accuracy_percent": 87.5},
    {"paper_method": "Fisherface/O", "gallery_per_identity": 2, "probe_per_identity": 8, "accuracy_percent": 79.7},
    {"paper_method": "MDA/2-1", "gallery_per_identity": 5, "probe_per_identity": 5, "accuracy_percent": 98.5},
    {"paper_method": "MDA/2-1", "gallery_per_identity": 4, "probe_per_identity": 6, "accuracy_percent": 96.3},
    {"paper_method": "MDA/2-1", "gallery_per_identity": 3, "probe_per_identity": 7, "accuracy_percent": 94.3},
    {"paper_method": "MDA/2-1", "gallery_per_identity": 2, "probe_per_identity": 8, "accuracy_percent": 88.1},
    {"paper_method": "MDA/2-2", "gallery_per_identity": 5, "probe_per_identity": 5, "accuracy_percent": 99.0},
    {"paper_method": "MDA/2-2", "gallery_per_identity": 4, "probe_per_identity": 6, "accuracy_percent": 98.3},
    {"paper_method": "MDA/2-2", "gallery_per_identity": 3, "probe_per_identity": 7, "accuracy_percent": 95.0},
    {"paper_method": "MDA/2-2", "gallery_per_identity": 2, "probe_per_identity": 8, "accuracy_percent": 90.1},
]


def parse_int_list(value: str) -> list[int]:
    try:
        values = sorted(set(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers.") from exc
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers.")
    return values


def accuracy_for_dimension(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    dimension: int,
) -> float:
    prediction = predict_nearest_neighbor(
        train_features[:, :dimension], train_labels, test_features[:, :dimension]
    )
    return float(np.mean(prediction == test_labels))


def add_row(
    rows: list[dict],
    *,
    train_count: int,
    repeat: int,
    seed: int,
    method: str,
    row_dim: int | str,
    col_dim: int | str,
    feature_count: int,
    accuracy: float,
    elapsed: float,
    classification_elapsed: float = 0.0,
    iterations: int = 0,
    objective: float | str = "",
) -> None:
    rows.append({
        "train_per_class": train_count,
        "repeat": repeat,
        "seed": seed,
        "method": method,
        "row_dim": row_dim,
        "col_dim": col_dim,
        "feature_count": feature_count,
        "accuracy": accuracy,
        "fit_seconds": elapsed,
        "classification_seconds": classification_elapsed,
        "fit_and_predict_seconds": elapsed + classification_elapsed,
        "mda_iterations": iterations,
        "final_training_fisher_ratio": objective,
    })


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    archive_sha256 = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    checkpoint_rows_path = args.output_dir / "raw_sweep_checkpoint.csv"
    checkpoint_objectives_path = args.output_dir / "mda_objectives_checkpoint.csv"
    checkpoint_config_path = args.output_dir / "sweep_checkpoint_config.json"
    checkpoint_config = {
        "archive_sha256": archive_sha256,
        "train_per_class": args.train_per_class,
        "mda_dimensions": args.mda_dimensions,
        "full_mda_grid": args.full_mda_grid,
        "repeats": args.repeats,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
        "paper_epsilon": args.paper_epsilon,
        "histogram_levels": args.histogram_levels,
        "workers": args.workers,
        "scatter_aggregation": "class-membership-matrix",
        "scatter_label_grouping": "cached-per-split",
        "first_iteration_cache": (
            "row-first identity-initialized prefix reuse"
            if args.mda_order == "row_first" and args.mda_stopping == "paper_figure"
            else "disabled for this update order/stopping rule"
        ),
        "mda22_objective_log": "final-only",
        "projection_normalization": args.projection_normalization,
        "mda_order": args.mda_order,
        "mda_stopping": args.mda_stopping,
    }
    if args.resume and checkpoint_rows_path.exists():
        if not checkpoint_config_path.is_file():
            raise ValueError("Cannot resume: checkpoint configuration is missing.")
        saved_config = json.loads(checkpoint_config_path.read_text(encoding="utf-8"))
        if saved_config != checkpoint_config:
            raise ValueError("Cannot resume: the requested sweep configuration differs from the checkpoint.")
        with checkpoint_rows_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if checkpoint_objectives_path.is_file():
            with checkpoint_objectives_path.open(newline="", encoding="utf-8") as handle:
                objective_rows = [
                    row for row in csv.DictReader(handle)
                    if row["method"] != "MDA/2-2"
                ]
        else:
            objective_rows = []
        completed_splits = {
            (int(row["train_per_class"]), int(row["repeat"])) for row in rows
        }
        print(f"Resuming from {len(completed_splits)} completed gallery/repeat splits.", flush=True)
    else:
        rows = []
        objective_rows = []
        completed_splits = set()
        checkpoint_config_path.write_text(json.dumps(checkpoint_config, indent=2) + "\n", encoding="utf-8")

    images, labels = load_original_orl(args.archive)
    images = equalize_histogram(images, args.histogram_levels)
    height, width = images.shape[1:]
    dimensions = list(range(1, max(height, width) + 1)) if args.full_mda_grid else args.mda_dimensions
    row_dimensions = [dimension for dimension in dimensions if dimension <= height]
    col_dimensions = [dimension for dimension in dimensions if dimension <= width]
    class_count = len(np.unique(labels))
    mda22_dimensions = [
        (row_dim, col_dim)
        for row_dim in row_dimensions
        for col_dim in col_dimensions
        if row_dim <= (class_count - 1) * col_dim
        and col_dim <= (class_count - 1) * row_dim
    ]

    for train_count in args.train_per_class:
        for repeat_index in range(args.repeats):
            repeat_number = repeat_index + 1
            if (train_count, repeat_number) in completed_splits:
                continue
            seed = args.seed + 1000 * train_count + repeat_index
            train_idx, test_idx = stratified_split(labels, train_count, seed)
            x_train, x_test = images[train_idx], images[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            scatter_grouping = make_class_grouping(y_train)
            max_eigenfaces = min(len(train_idx) - 1, height * width)
            max_fisherfaces = min(len(np.unique(y_train)) - 1, 39)

            started = time.perf_counter()
            pca_train, pca_test = fit_pca(x_train, x_test, max_eigenfaces)
            elapsed = time.perf_counter() - started
            for dimension in range(1, pca_train.shape[1] + 1):
                classification_started = time.perf_counter()
                accuracy = accuracy_for_dimension(
                    pca_train, y_train, pca_test, y_test, dimension
                )
                classification_elapsed = time.perf_counter() - classification_started
                add_row(
                    rows, train_count=train_count, repeat=repeat_index + 1, seed=seed,
                    method="Eigenface", row_dim="", col_dim="", feature_count=dimension,
                    accuracy=accuracy, elapsed=elapsed,
                    classification_elapsed=classification_elapsed,
                )

            started = time.perf_counter()
            fisher_train, fisher_test = fit_fisherfaces(
                x_train, y_train, x_test, n_components=max_fisherfaces,
                projection_normalization=args.projection_normalization,
                pca_components_max=None,
            )
            elapsed = time.perf_counter() - started
            for dimension in range(1, fisher_train.shape[1] + 1):
                classification_started = time.perf_counter()
                accuracy = accuracy_for_dimension(
                    fisher_train, y_train, fisher_test, y_test, dimension
                )
                classification_elapsed = time.perf_counter() - classification_started
                add_row(
                    rows, train_count=train_count, repeat=repeat_index + 1, seed=seed,
                    method="Fisherface", row_dim="", col_dim="", feature_count=dimension,
                    accuracy=accuracy, elapsed=elapsed,
                    classification_elapsed=classification_elapsed,
                )

            max_row_dim = min(height, (len(np.unique(y_train)) - 1) * width)
            started = time.perf_counter()
            mda1_train, mda1_test, mda1_history = fit_mda(
                x_train, y_train, x_test, components=(max_row_dim, None),
                max_iter=args.max_iter, tolerance=args.tolerance,
                projection_normalization=args.projection_normalization,
                stopping_rule=args.mda_stopping,
                mode_order=args.mda_order,
                paper_epsilon=args.paper_epsilon,
                scatter_grouping=scatter_grouping,
            )
            elapsed = time.perf_counter() - started
            for dimension in range(1, mda1_train.shape[1] // width + 1):
                feature_dim = dimension * width
                classification_started = time.perf_counter()
                accuracy = accuracy_for_dimension(
                    mda1_train, y_train, mda1_test, y_test, feature_dim
                )
                classification_elapsed = time.perf_counter() - classification_started
                add_row(
                    rows, train_count=train_count, repeat=repeat_index + 1, seed=seed,
                    method="MDA/2-1 (row mode)", row_dim=dimension, col_dim=width,
                    feature_count=feature_dim,
                    elapsed=elapsed, iterations=len(mda1_history), objective=mda1_history[-1],
                    accuracy=accuracy, classification_elapsed=classification_elapsed,
                )
            for iteration, objective in enumerate(mda1_history, start=1):
                objective_rows.append({
                    "train_per_class": train_count, "repeat": repeat_index + 1,
                    "seed": seed, "method": "MDA/2-1 (row mode)",
                    "row_dim": max_row_dim, "col_dim": width,
                    "iteration": iteration, "training_fisher_ratio": objective,
                })

            initial_projection_cache = None
            if args.mda_order == "row_first" and args.mda_stopping == "paper_figure":
                cache_started = time.perf_counter()
                initial_projection_cache = precompute_mda22_first_iteration(
                    x_train,
                    y_train,
                    mda22_dimensions,
                    projection_normalization=args.projection_normalization,
                    scatter_grouping=scatter_grouping,
                )
                print(
                    f"gallery={train_count} repeat={repeat_number}/{args.repeats} "
                    f"cached first MDA iteration in {time.perf_counter() - cache_started:.2f}s",
                    flush=True,
                )

            def evaluate_mda22(pair: tuple[int, int]) -> tuple[dict, list[dict]]:
                row_dim, col_dim = pair
                started = time.perf_counter()
                initial_projections = None
                initial_iterations = 0
                initial_history = None
                if initial_projection_cache is not None:
                    initial_row, full_initial_col = initial_projection_cache[row_dim]
                    initial_projections = (initial_row, full_initial_col[:, :col_dim])
                    initial_iterations = 1
                    initial_history = [float("nan")]
                mda_train, mda_test, history = fit_mda(
                    x_train, y_train, x_test, components=(row_dim, col_dim),
                    max_iter=args.max_iter, tolerance=args.tolerance,
                    projection_normalization=args.projection_normalization,
                    stopping_rule=args.mda_stopping,
                    mode_order=args.mda_order,
                    paper_epsilon=args.paper_epsilon,
                    track_objective_history=False,
                    scatter_grouping=scatter_grouping,
                    initial_projections=initial_projections,
                    initial_iterations=initial_iterations,
                    initial_history=initial_history,
                )
                fit_elapsed = time.perf_counter() - started
                classification_started = time.perf_counter()
                prediction = predict_nearest_neighbor(mda_train, y_train, mda_test)
                classification_elapsed = time.perf_counter() - classification_started
                accuracy = float(np.mean(prediction == y_test))
                method = "MDA/2-2"
                pair_rows: list[dict] = []
                add_row(
                    pair_rows, train_count=train_count, repeat=repeat_index + 1, seed=seed,
                    method=method, row_dim=row_dim, col_dim=col_dim,
                    feature_count=row_dim * col_dim, accuracy=accuracy, elapsed=fit_elapsed,
                    classification_elapsed=classification_elapsed,
                    iterations=len(history), objective=history[-1],
                )
                pair_objectives = []
                pair_objectives.append({
                    "train_per_class": train_count, "repeat": repeat_index + 1,
                    "seed": seed, "method": method,
                    "row_dim": row_dim, "col_dim": col_dim,
                    "iteration": len(history), "training_fisher_ratio": history[-1],
                })
                return pair_rows[0], pair_objectives

            with threadpool_limits(limits=1):
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    for pair_number, (pair_row, pair_objectives) in enumerate(
                        executor.map(evaluate_mda22, mda22_dimensions), start=1
                    ):
                        rows.append(pair_row)
                        objective_rows.extend(pair_objectives)
                        if pair_number % 1000 == 0:
                            print(
                                f"gallery={train_count} repeat={repeat_number}/{args.repeats} "
                                f"MDA/2-2={pair_number}/{len(mda22_dimensions)} pairs",
                                flush=True,
                            )

            print(
                f"gallery={train_count} repeat={repeat_number}/{args.repeats} "
                f"rows={len(rows)} MDA/2-2={len(mda22_dimensions)} pairs complete",
                flush=True,
            )
            write_csv(checkpoint_rows_path, rows, list(rows[0]))
            if objective_rows:
                write_csv(
                    checkpoint_objectives_path,
                    objective_rows,
                    ["train_per_class", "repeat", "seed", "method", "row_dim", "col_dim", "iteration", "training_fisher_ratio"],
                )

    grouped: dict[tuple[int, str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["train_per_class"]), str(row["method"]), str(row["row_dim"]),
            str(row["col_dim"]), int(row["feature_count"]),
        )
        grouped[key].append(float(row["accuracy"]))
    summary = []
    for (train_count, method, row_dim, col_dim, feature_count), values in sorted(grouped.items()):
        summary.append({
            "train_per_class": train_count,
            "method": method,
            "row_dim": row_dim,
            "col_dim": col_dim,
            "feature_count": feature_count,
            "repeats": len(values),
            "mean_accuracy": float(np.mean(values)),
            "std_accuracy": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        })

    metadata = [
        {"key": "paper", "value": "Yan et al. (2007), IEEE TIP 16(1), 212-220"},
        {"key": "paper_url", "value": PAPER_URL},
        {"key": "dataset", "value": "Original-resolution AT&T/ORL Database of Faces"},
        {"key": "dataset_url", "value": ORL_ARCHIVE_URL},
        {"key": "dataset_credit", "value": ORL_CITATION},
        {"key": "dataset_archive_sha256", "value": archive_sha256},
        {"key": "dataset_shape_rows_columns", "value": "112x92"},
        {"key": "images_and_identities", "value": "400 images; 40 identities; 10 images each"},
        {"key": "preprocessing", "value": f"Per-image 8-bit histogram equalization to {args.histogram_levels} output levels; converted from original 8-bit PGM"},
        {"key": "histogram_output_levels", "value": str(args.histogram_levels)},
        {"key": "gallery_counts", "value": ",".join(map(str, args.train_per_class))},
        {"key": "repeats", "value": str(args.repeats)},
        {"key": "seed_rule", "value": f"{args.seed} + 1000 * gallery_count + zero_based_repeat"},
        {"key": "classifier", "value": "Euclidean 1-nearest-neighbour; matrix embeddings flattened (Frobenius distance)"},
        {"key": "projection_normalization", "value": args.projection_normalization},
        {"key": "mda_update_order", "value": args.mda_order},
        {"key": "mda_ridge_relative_to_mean_within_scatter", "value": "1e-6"},
        {"key": "mda_stopping", "value": (
            f"paper Fig. 3: projection Frobenius change < output_dim * input_dim * epsilon after iteration 2; epsilon={args.paper_epsilon}; max {args.max_iter}"
            if args.mda_stopping == "paper_figure"
            else f"{args.mda_stopping}; tolerance={args.tolerance}; max {args.max_iter}"
        )},
        {"key": "mda2_2_dimension_grid", "value": ("all theorem-feasible integer dimension pairs" if args.full_mda_grid else ",".join(map(str, row_dimensions)) + " x " + ",".join(map(str, col_dimensions)) + " filtered by Theorem 2 rank bounds")},
        {"key": "mda22_feasible_pair_count", "value": str(len(mda22_dimensions))},
        {"key": "workers", "value": str(args.workers)},
        {"key": "linear_algebra_threads", "value": os.environ.get("VECLIB_MAXIMUM_THREADS", "1 (threadpoolctl on non-Accelerate backends)")},
        {"key": "mode_scatter_aggregation", "value": "class membership matrix; same class-wise scatter equations"},
        {"key": "scatter_label_grouping", "value": "class index/counts/one-hot membership cached once per split"},
        {"key": "first_iteration_cache", "value": checkpoint_config["first_iteration_cache"]},
        {"key": "mda22_objective_log", "value": "final objective per dimension/split; stopping follows projection change"},
        {"key": "timing_columns", "value": "fit_seconds and classification_seconds are timed separately; fit_and_predict_seconds is their sum"},
        {"key": "selection_warning", "value": "Best mean probe accuracy over this sweep is an optimistic, test-selected diagnostic; do not present it as an unbiased estimate."},
        {"key": "replication_scope", "value": "Original-resolution, paper-style follow-up. The article does not report split seeds or exact full feature sweep, so exact Table I recreation is not possible."},
    ]

    # Best-over-grid rows mimic the paper's published best-dimension presentation.
    # The per-dimension summary and raw rows remain alongside these selected values.
    best = []
    by_method_gallery: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in summary:
        by_method_gallery[(int(row["train_per_class"]), str(row["method"]))].append(row)
    for (train_count, method), candidates in sorted(by_method_gallery.items()):
        selected = max(candidates, key=lambda item: float(item["mean_accuracy"]))
        best.append({
            **selected,
            "selection_basis": "highest mean probe accuracy over the predeclared sweep; optimistic, not unbiased",
        })
    return rows, summary, best, objective_rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(".cache/orl-original/att_faces.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-10-28-orl-original-dimension-sweep"))
    parser.add_argument("--train-per-class", type=parse_int_list, default=parse_int_list("2,3,4,5"))
    parser.add_argument("--mda-dimensions", type=parse_int_list, default=parse_int_list("2,4,6,8,12"))
    parser.add_argument("--full-mda-grid", action="store_true", help="Search every feasible integer row/column dimension on the 112x92 ORL images.")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3314)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--paper-epsilon", type=float, default=1e-4, help="Epsilon in Fig. 3; the article does not report its value.")
    parser.add_argument("--histogram-levels", type=int, default=256, help="Per-image equalization output levels; 64 approximates MATLAB histeq's default.")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent independent MDA/2-2 fits.")
    parser.add_argument(
        "--mda-stopping",
        choices=("paper_figure", "projection_change", "objective"),
        default="paper_figure",
    )
    parser.add_argument("--projection-normalization", choices=("unit_l2", "within_scatter"), default="unit_l2")
    parser.add_argument("--mda-order", choices=("row_first", "column_first"), default="row_first")
    parser.add_argument("--resume", action="store_true", help="Resume completed splits from a compatible checkpoint.")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_iter < 1 or args.workers < 1:
        parser.error("repeats, max-iter, and workers must be positive")
    if not 2 <= args.histogram_levels <= 256 or args.paper_epsilon <= 0:
        parser.error("histogram-levels must be 2..256 and paper-epsilon must be positive")
    if not args.archive.is_file():
        parser.error(f"ORL archive not found: {args.archive}; see docs/data-and-ethics.md")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary, best, objectives, metadata = run(args)
    write_csv(args.output_dir / "raw_sweep.csv", rows, list(rows[0]))
    write_csv(args.output_dir / "dimension_summary.csv", summary, list(summary[0]))
    write_csv(args.output_dir / "best_by_gallery.csv", best, list(best[0]))
    if objectives:
        write_csv(
            args.output_dir / "mda_objectives.csv",
            objectives,
            ["train_per_class", "repeat", "seed", "method", "row_dim", "col_dim", "iteration", "training_fisher_ratio"],
        )
    write_csv(args.output_dir / "metadata.csv", metadata, ["key", "value"])
    write_csv(args.output_dir / "published_table_i.csv", PUBLISHED_TABLE, list(PUBLISHED_TABLE[0]))
    print(f"\nSaved {len(rows)} split/configuration rows to {args.output_dir.resolve()}")
    print("Best-by-gallery values are test-selected diagnostics; cite dimension_summary.csv for the full curves.")
    for checkpoint in (
        args.output_dir / "raw_sweep_checkpoint.csv",
        args.output_dir / "mda_objectives_checkpoint.csv",
        args.output_dir / "sweep_checkpoint_config.json",
    ):
        if checkpoint.exists():
            checkpoint.unlink()


if __name__ == "__main__":
    main()
