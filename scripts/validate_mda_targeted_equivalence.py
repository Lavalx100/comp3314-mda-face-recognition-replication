#!/usr/bin/env python3
"""Compare direct and cached MDA fits on a supplied set of saved configurations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orl_data import load_original_orl  # noqa: E402
from run_experiment import (  # noqa: E402
    apply_mda,
    equalize_histogram,
    fisher_trace_ratio,
    make_class_grouping,
    predict_nearest_neighbor,
    precompute_mda22_first_iteration,
    project_generalized_eigensystem,
    fit_mda_projections,
    stratified_split,
)


def squared_distances(train: np.ndarray, probes: np.ndarray) -> np.ndarray:
    train_sq = np.sum(train * train, axis=1)
    probe_sq = np.sum(probes * probes, axis=1)
    return np.maximum(probe_sq[:, None] + train_sq[None, :] - 2.0 * probes @ train.T, 0.0)


def aligned_relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    signs = np.sign(np.sum(left * right, axis=0))
    signs[signs == 0] = 1.0
    return float(np.linalg.norm(left - right * signs[None, :]) / max(np.linalg.norm(left), 1e-300))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(".cache/orl-original/att_faces.zip"))
    parser.add_argument("--baseline-grid", type=Path, required=True)
    parser.add_argument("--target-configurations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization", choices=("unit_l2", "within_scatter"), required=True)
    parser.add_argument("--histogram-levels", type=int, default=64)
    args = parser.parse_args()

    with (args.baseline_grid / "raw_sweep.csv").open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    with args.target_configurations.open(newline="", encoding="utf-8") as handle:
        requested_rows = list(csv.DictReader(handle))
    baseline = {
        (int(row["train_per_class"]), int(row["repeat"]), int(row["seed"]), int(row["row_dim"]), int(row["col_dim"])): row
        for row in baseline_rows
        if row["method"] == "MDA/2-2"
    }
    requested = sorted({
        (int(row["train_per_class"]), int(row["repeat"]), int(row["seed"]), int(row["row_dim"]), int(row["col_dim"]))
        for row in requested_rows
    })
    missing = [key for key in requested if key not in baseline]
    if missing:
        raise ValueError(f"{len(missing)} requested configurations are absent from the baseline grid.")

    images, labels = load_original_orl(args.archive)
    images = equalize_histogram(images, args.histogram_levels)
    targets_by_seed: dict[tuple[int, int], list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for key in requested:
        gallery_size, repeat, seed, row_dim, col_dim = key
        targets_by_seed[(gallery_size, seed)].append(key)

    records: list[dict[str, object]] = []
    for (gallery_size, seed), targets in sorted(targets_by_seed.items()):
        train_indices, probe_indices = stratified_split(labels, gallery_size, seed)
        train_images, probe_images = images[train_indices], images[probe_indices]
        train_labels, probe_labels = labels[train_indices], labels[probe_indices]
        grouping = make_class_grouping(train_labels)
        pairs = sorted({(key[3], key[4]) for key in targets})
        first_iteration = precompute_mda22_first_iteration(
            train_images,
            train_labels,
            pairs,
            projection_normalization=args.normalization,
            scatter_grouping=grouping,
        )

        for key in targets:
            _, repeat, _, row_dim, col_dim = key
            original = baseline[key]
            fit_options = dict(
                components=(row_dim, col_dim),
                max_iter=10,
                tolerance=1e-4,
                projection_normalization=args.normalization,
                stopping_rule="paper_figure",
                mode_order="row_first",
                paper_epsilon=1e-4,
                track_objective_history=False,
                scatter_grouping=grouping,
            )
            direct_row, direct_col, direct_history = fit_mda_projections(
                train_images, train_labels, **fit_options
            )
            cached_row, cached_col_system = first_iteration[row_dim]
            cached_col = project_generalized_eigensystem(
                cached_col_system,
                col_dim,
                projection_normalization=args.normalization,
            )
            cached_row, cached_col, cached_history = fit_mda_projections(
                train_images,
                train_labels,
                **fit_options,
                initial_projections=(cached_row, cached_col),
                initial_iterations=1,
                initial_history=[float("nan")],
            )

            direct_train = apply_mda(train_images, direct_row, direct_col)
            direct_probe = apply_mda(probe_images, direct_row, direct_col)
            cached_train = apply_mda(train_images, cached_row, cached_col)
            cached_probe = apply_mda(probe_images, cached_row, cached_col)
            direct_predictions = predict_nearest_neighbor(direct_train, train_labels, direct_probe)
            cached_predictions = predict_nearest_neighbor(cached_train, train_labels, cached_probe)
            direct_accuracy = float(np.mean(direct_predictions == probe_labels))
            cached_accuracy = float(np.mean(cached_predictions == probe_labels))
            direct_objective = fisher_trace_ratio(direct_train, train_labels)
            cached_objective = fisher_trace_ratio(cached_train, train_labels)
            direct_distances = squared_distances(direct_train, direct_probe)
            cached_distances = squared_distances(cached_train, cached_probe)
            records.append({
                "gallery_size": gallery_size,
                "repeat": repeat,
                "seed": seed,
                "row_dim": row_dim,
                "col_dim": col_dim,
                "baseline_accuracy": float(original["accuracy"]),
                "direct_accuracy": direct_accuracy,
                "cached_accuracy": cached_accuracy,
                "baseline_iterations": int(original["mda_iterations"]),
                "direct_iterations": len(direct_history),
                "cached_iterations": len(cached_history),
                "prediction_mismatches": int(np.count_nonzero(direct_predictions != cached_predictions)),
                "row_projection_relative_difference": aligned_relative_difference(direct_row, cached_row),
                "col_projection_relative_difference": aligned_relative_difference(direct_col, cached_col),
                "squared_distance_relative_difference": float(
                    np.linalg.norm(direct_distances - cached_distances)
                    / max(np.linalg.norm(direct_distances), 1e-300)
                ),
                "direct_objective": direct_objective,
                "cached_objective": cached_objective,
                "objective_absolute_difference": abs(direct_objective - cached_objective),
                "direct_matches_baseline_accuracy": direct_accuracy == float(original["accuracy"]),
                "direct_matches_baseline_iterations": len(direct_history) == int(original["mda_iterations"]),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "within_scatter_targeted_equivalence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "normalization": args.normalization,
        "baseline_grid": str(args.baseline_grid),
        "target_count": len(records),
        "prediction_mismatches": sum(int(r["prediction_mismatches"]) for r in records),
        "direct_vs_baseline_accuracy_mismatches": sum(not bool(r["direct_matches_baseline_accuracy"]) for r in records),
        "direct_vs_baseline_iteration_mismatches": sum(not bool(r["direct_matches_baseline_iterations"]) for r in records),
        "max_row_projection_relative_difference": max(float(r["row_projection_relative_difference"]) for r in records),
        "max_col_projection_relative_difference": max(float(r["col_projection_relative_difference"]) for r in records),
        "max_squared_distance_relative_difference": max(float(r["squared_distance_relative_difference"]) for r in records),
        "max_objective_absolute_difference": max(float(r["objective_absolute_difference"]) for r in records),
        "records_csv": csv_path.name,
    }
    (args.output_dir / "within_scatter_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
