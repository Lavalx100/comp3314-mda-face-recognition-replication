#!/usr/bin/env python3
"""Profile and benchmark the ORL MDA sweep without running the exhaustive grid.

The parent process launches isolated children so Accelerate/OpenBLAS thread
settings are applied before NumPy loads. The benchmark uses one fixed ORL split
and a deterministic, dimension-spread set of feasible MDA pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = [
    (2, 2), (2, 8), (4, 4), (8, 8), (16, 12), (24, 16),
    (32, 24), (48, 24), (56, 46), (64, 46), (92, 46), (112, 92),
]


def parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def run_child(args: argparse.Namespace) -> dict:
    # Set these before importing NumPy or any module that imports it.
    thread_count = int(args.blas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
    os.environ["MKL_NUM_THREADS"] = str(thread_count)
    os.environ["BLIS_NUM_THREADS"] = str(thread_count)
    os.environ["OMP_NUM_THREADS"] = str(thread_count)

    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits

    sys.path.insert(0, str(ROOT / "src"))
    import run_experiment as experiment
    from orl_data import load_original_orl

    archive = args.archive
    started = time.perf_counter()
    images, labels = load_original_orl(archive)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    images = experiment.equalize_histogram(images, args.histogram_levels)
    histogram_seconds = time.perf_counter() - started
    started = time.perf_counter()
    train_idx, test_idx = experiment.stratified_split(
        labels, args.gallery_size, args.seed + 1000 * args.gallery_size
    )
    x_train, x_test = images[train_idx], images[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    split_seconds = time.perf_counter() - started
    grouping = experiment.make_class_grouping(y_train)
    pairs = [tuple(pair) for pair in DEFAULT_PAIRS]

    # Stage profiling runs the ordinary implementation and instruments its
    # internals; it does not use a warm-start cache.
    mode = "direct" if args.child_mode == "stages" else args.child_mode

    cache_seconds = 0.0
    initial_cache = None
    if mode == "cached":
        started = time.perf_counter()
        initial_cache = experiment.precompute_mda22_first_iteration(
            x_train,
            y_train,
            pairs,
            projection_normalization=args.normalization,
            scatter_grouping=grouping,
        )
        cache_seconds = time.perf_counter() - started

    def evaluate(pair: tuple[int, int]) -> dict:
        row_dim, col_dim = pair
        initial_projections = None
        initial_iterations = 0
        initial_history = None
        if initial_cache is not None:
            row_basis, col_eigensystem = initial_cache[row_dim]
            col_basis = experiment.project_generalized_eigensystem(
                col_eigensystem,
                col_dim,
                projection_normalization=args.normalization,
            )
            initial_projections = (row_basis, col_basis)
            initial_iterations = 1
            initial_history = [float("nan")]
        started_fit = time.perf_counter()
        train_features, test_features, history = experiment.fit_mda(
            x_train,
            y_train,
            x_test,
            components=pair,
            max_iter=args.max_iter,
            tolerance=1e-4,
            projection_normalization=args.normalization,
            stopping_rule="paper_figure",
            mode_order="row_first",
            paper_epsilon=args.paper_epsilon,
            track_objective_history=False,
            scatter_grouping=grouping,
            initial_projections=initial_projections,
            initial_iterations=initial_iterations,
            initial_history=initial_history,
        )
        fit_seconds = time.perf_counter() - started_fit
        started_classification = time.perf_counter()
        prediction = experiment.predict_nearest_neighbor(
            train_features, y_train, test_features
        )
        classification_seconds = time.perf_counter() - started_classification
        return {
            "pair": pair,
            "fit_seconds": fit_seconds,
            "classification_seconds": classification_seconds,
            "accuracy": float(np.mean(prediction == y_test)),
            "iterations": len(history),
        }

    if args.child_mode == "stages":
        timer = {"scatter_seconds": 0.0, "eigendecomposition_seconds": 0.0}
        original_scatter = experiment.mode_scatter
        original_eigh = np.linalg.eigh

        def timed_scatter(*call_args, **call_kwargs):
            start = time.perf_counter()
            try:
                return original_scatter(*call_args, **call_kwargs)
            finally:
                timer["scatter_seconds"] += time.perf_counter() - start

        def timed_eigh(*call_args, **call_kwargs):
            start = time.perf_counter()
            try:
                return original_eigh(*call_args, **call_kwargs)
            finally:
                timer["eigendecomposition_seconds"] += time.perf_counter() - start

        experiment.mode_scatter = timed_scatter
        np.linalg.eigh = timed_eigh
        with threadpool_limits(limits=thread_count):
            profiles = [evaluate(pair) for pair in DEFAULT_PAIRS]
        np.linalg.eigh = original_eigh
        experiment.mode_scatter = original_scatter
        total_fit = sum(row["fit_seconds"] for row in profiles)
        total_iterations = sum(row["iterations"] for row in profiles)
        result = {
            "mode": "direct-stage-profile",
            "normalization": args.normalization,
            "gallery_size": args.gallery_size,
            "data_load_seconds": load_seconds,
            "histogram_equalization_seconds": histogram_seconds,
            "split_seconds": split_seconds,
            **timer,
            "fit_seconds": total_fit,
            "mda_iteration_seconds_per_iteration": total_fit / max(total_iterations, 1),
            "other_iteration_projection_seconds": max(
                total_fit - timer["scatter_seconds"] - timer["eigendecomposition_seconds"],
                0.0,
            ),
            "classification_seconds": sum(
                row["classification_seconds"] for row in profiles
            ),
            "profiles": profiles,
        }
    else:
        jobs = pairs * args.pair_repeats
        started = time.perf_counter()
        with threadpool_limits(limits=thread_count):
            if args.workers == 1:
                results = [evaluate(pair) for pair in jobs]
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    results = list(executor.map(evaluate, jobs))
        wall_seconds = time.perf_counter() - started
        fits = len(results)
        result = {
            "mode": mode,
            "normalization": args.normalization,
            "gallery_size": args.gallery_size,
            "workers": args.workers,
            "blas_threads": thread_count,
            "fits": fits,
            "cache_build_seconds": cache_seconds,
            "wall_seconds": wall_seconds,
            "fits_per_second": fits / wall_seconds,
            "fit_seconds_sum": sum(row["fit_seconds"] for row in results),
            "classification_seconds_sum": sum(
                row["classification_seconds"] for row in results
            ),
            "mean_accuracy": float(np.mean([row["accuracy"] for row in results])),
            "iterations_mean": float(np.mean([row["iterations"] for row in results])),
            "peak_rss_gib": (
                __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss
                / (1024**3 if sys.platform == "darwin" else 1024**2)
            ),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "threadpool_info": threadpool_info(),
        }
    return result


def parent_main(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        timestamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d-%H%M%S")
        args.output_dir = Path("results/runs") / f"{timestamp}-mda-performance-benchmark"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_counts = parse_ints(args.worker_counts)
    blas_counts = parse_ints(args.blas_counts)
    # Sweep the complete worker curve at the first BLAS setting, then vary
    # BLAS threads at a representative subset of worker counts. This limits
    # redundant configurations while still exposing oversubscription.
    combinations = []
    for mode in ("direct", "cached"):
        combinations.extend((mode, workers, blas_counts[0]) for workers in worker_counts)
        representative_workers = [worker for worker in (1, 4, 8, 16) if worker in worker_counts]
        for blas in blas_counts[1:]:
            combinations.extend((mode, workers, blas) for workers in representative_workers)
    with tempfile.TemporaryDirectory(prefix="mda-efficiency-") as temp_dir:
        records = []
        for index, (mode, workers, blas) in enumerate(combinations, start=1):
            output_path = Path(temp_dir) / f"{index}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child-mode", mode,
                "--workers", str(workers),
                "--blas-threads", str(blas),
                "--pair-repeats", str(args.pair_repeats),
                "--gallery-size", str(args.gallery_size),
                "--seed", str(args.seed),
                "--normalization", args.normalization,
                "--archive", str(args.archive.resolve()),
                "--output-json", str(output_path),
            ]
            subprocess.run(command, check=True, cwd=ROOT)
            record = json.loads(output_path.read_text(encoding="utf-8"))
            records.append(record)
            print(
                f"{mode:6s} workers={workers:2d} BLAS={blas}: "
                f"{record['fits_per_second']:.2f} fits/s, "
                f"{record['wall_seconds']:.2f}s, "
                f"cache={record['cache_build_seconds']:.2f}s",
                flush=True,
            )

        csv_path = args.output_dir / "worker_scaling.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "mode", "normalization", "gallery_size", "workers", "blas_threads",
                "fits", "cache_build_seconds", "wall_seconds", "fits_per_second",
                "fit_seconds_sum", "classification_seconds_sum", "mean_accuracy",
                "iterations_mean", "peak_rss_gib", "numpy_version", "scipy_version",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow({key: row.get(key, "") for key in fields})

        profile_json = Path(temp_dir) / "stages.json"
        subprocess.run([
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-mode", "stages",
            "--workers", "1",
            "--blas-threads", str(blas_counts[0]),
            "--gallery-size", str(args.gallery_size),
            "--normalization", args.normalization,
            "--archive", str(args.archive.resolve()),
            "--output-json", str(profile_json),
        ], check=True, cwd=ROOT)
        stage_result = json.loads(profile_json.read_text(encoding="utf-8"))
        (args.output_dir / "stage_timings.json").write_text(
            json.dumps(stage_result, indent=2) + "\n", encoding="utf-8"
        )

    metadata = {
        "host_platform": platform.platform(),
        "machine": platform.machine(),
        "source": "scripts/benchmark_mda_efficiency.py",
        "representative_pairs": [list(pair) for pair in DEFAULT_PAIRS],
        "pair_repeats": args.pair_repeats,
        "normalization": args.normalization,
        "gallery_size": args.gallery_size,
        "base_seed": args.seed,
        "split_seed": args.seed + 1000 * args.gallery_size,
        "worker_counts": worker_counts,
        "blas_counts": blas_counts,
        "note": "Representative configuration benchmark only; no exhaustive grid is run.",
    }
    (args.output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved benchmark outputs to {args.output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(".cache/orl-original/att_faces.zip"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory; defaults to a timestamped results/runs folder.")
    parser.add_argument("--worker-counts", default="1,2,4,8,10,12,14,16")
    parser.add_argument("--blas-counts", default="1,2,4")
    parser.add_argument("--pair-repeats", type=int, default=2)
    parser.add_argument("--gallery-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3314)
    parser.add_argument("--normalization", choices=("unit_l2", "within_scatter"), default="unit_l2")
    parser.add_argument("--histogram-levels", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--paper-epsilon", type=float, default=1e-4)
    parser.add_argument("--child-mode", choices=("direct", "cached", "stages"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.child_mode:
        if args.output_json is None:
            parser.error("Child mode requires --output-json.")
        result = run_child(args)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        if not args.archive.is_file():
            parser.error(f"ORL archive not found: {args.archive}")
        parent_main(args)


if __name__ == "__main__":
    main()
