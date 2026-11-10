#!/usr/bin/env python3
"""Recheck only cached/direct MDA configurations that differ in saved outputs.

This diagnostic joins a complete direct grid with a partial cached checkpoint,
then reruns only matching MDA/2-2 rows whose accuracy, iteration count, or
final objective differs beyond the selected float64 screening tolerance.
"""

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


KEY_FIELDS = (
    "train_per_class", "repeat", "seed", "method", "row_dim", "col_dim",
    "feature_count",
)
OBJECTIVE_SCREENING_REL_TOL = 1e-7


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def squared_distances(train: np.ndarray, probes: np.ndarray) -> np.ndarray:
    train_sq = np.sum(train * train, axis=1)
    probe_sq = np.sum(probes * probes, axis=1)
    return np.maximum(probe_sq[:, None] + train_sq[None, :] - 2.0 * probes @ train.T, 0.0)


def aligned_projection_relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    signs = np.sign(np.sum(left * right, axis=0))
    signs[signs == 0] = 1.0
    return float(np.linalg.norm(left - right * signs[None, :]) / max(np.linalg.norm(left), 1e-300))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(".cache/orl-original/att_faces.zip"))
    parser.add_argument("--baseline-grid", type=Path, required=True)
    parser.add_argument("--cached-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization", choices=("unit_l2", "within_scatter"), required=True)
    parser.add_argument("--histogram-levels", type=int, default=64)
    args = parser.parse_args()

    baseline_rows = read_csv(args.baseline_grid / "raw_sweep.csv")
    cached_rows = read_csv(args.cached_checkpoint)
    baseline = {row_key(row): row for row in baseline_rows}
    cached = {row_key(row): row for row in cached_rows}
    shared = baseline.keys() & cached.keys()
    screened = []
    partial_accuracy_mismatches = 0
    partial_iteration_mismatches = 0
    partial_objective_relative_differences = []
    partial_differences = []
    for key in shared:
        old, new = baseline[key], cached[key]
        if old["method"] != "MDA/2-2":
            continue
        accuracy_mismatch = old["accuracy"] != new["accuracy"]
        iteration_mismatch = old["mda_iterations"] != new["mda_iterations"]
        old_objective = float(old["final_training_fisher_ratio"])
        new_objective = float(new["final_training_fisher_ratio"])
        relative_objective_difference = abs(old_objective - new_objective) / max(
            abs(old_objective), abs(new_objective), 1e-300
        )
        partial_accuracy_mismatches += accuracy_mismatch
        partial_iteration_mismatches += iteration_mismatch
        partial_objective_relative_differences.append(relative_objective_difference)
        if accuracy_mismatch or iteration_mismatch or relative_objective_difference > OBJECTIVE_SCREENING_REL_TOL:
            partial_differences.append({
                "train_per_class": int(key[0]),
                "repeat": int(key[1]),
                "seed": int(key[2]),
                "row_dim": int(key[4]),
                "col_dim": int(key[5]),
                "original_accuracy": old["accuracy"],
                "partial_cached_accuracy": new["accuracy"],
                "original_iterations": old["mda_iterations"],
                "partial_cached_iterations": new["mda_iterations"],
                "objective_relative_difference": relative_objective_difference,
            })
        if (
            accuracy_mismatch
            or iteration_mismatch
            or relative_objective_difference > OBJECTIVE_SCREENING_REL_TOL
        ):
            screened.append((key, old, new, relative_objective_difference))

    images, labels = load_original_orl(args.archive)
    images = equalize_histogram(images, args.histogram_levels)
    height, width = images.shape[1:]
    class_count = len(np.unique(labels))
    all_pairs = [
        (row_dim, col_dim)
        for row_dim in range(1, height + 1)
        for col_dim in range(1, width + 1)
        if row_dim <= (class_count - 1) * col_dim
        and col_dim <= (class_count - 1) * row_dim
    ]

    targets_by_seed: dict[int, list[tuple[tuple[str, ...], dict, dict, float]]] = defaultdict(list)
    for item in screened:
        targets_by_seed[int(item[0][2])].append(item)

    records = []
    for seed, targets in sorted(targets_by_seed.items()):
        gallery_size = int(targets[0][1]["train_per_class"])
        train_indices, probe_indices = stratified_split(labels, gallery_size, seed)
        train_images, probe_images = images[train_indices], images[probe_indices]
        train_labels, probe_labels = labels[train_indices], labels[probe_indices]
        grouping = make_class_grouping(train_labels)
        first_iteration = precompute_mda22_first_iteration(
            train_images,
            train_labels,
            all_pairs,
            projection_normalization=args.normalization,
            scatter_grouping=grouping,
        )

        for key, old, partial, screened_objective_difference in sorted(
            targets, key=lambda entry: (int(entry[0][4]), int(entry[0][5]))
        ):
            row_dim, col_dim = int(key[4]), int(key[5])
            common = dict(
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
                train_images, train_labels, **common
            )
            cached_row, cached_col_eigensystem = first_iteration[row_dim]
            cached_col = project_generalized_eigensystem(
                cached_col_eigensystem,
                col_dim,
                projection_normalization=args.normalization,
            )
            cached_row, cached_col, cached_history = fit_mda_projections(
                train_images,
                train_labels,
                **common,
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
            direct_distances = squared_distances(direct_train, direct_probe)
            cached_distances = squared_distances(cached_train, cached_probe)
            original_objective = float(old["final_training_fisher_ratio"])
            direct_objective = fisher_trace_ratio(direct_train, train_labels)
            cached_objective = fisher_trace_ratio(cached_train, train_labels)

            records.append({
                "gallery_size": gallery_size,
                "repeat": int(key[1]),
                "seed": seed,
                "row_dim": row_dim,
                "col_dim": col_dim,
                "original_accuracy": float(old["accuracy"]),
                "partial_cached_accuracy": float(partial["accuracy"]),
                "direct_accuracy": float(np.mean(direct_predictions == probe_labels)),
                "corrected_cached_accuracy": float(np.mean(cached_predictions == probe_labels)),
                "original_iterations": int(old["mda_iterations"]),
                "partial_cached_iterations": int(partial["mda_iterations"]),
                "direct_iterations": len(direct_history),
                "corrected_cached_iterations": len(cached_history),
                "prediction_mismatches_direct_vs_cached": int(np.count_nonzero(direct_predictions != cached_predictions)),
                "row_projection_relative_difference_sign_aligned": aligned_projection_relative_difference(direct_row, cached_row),
                "col_projection_relative_difference_sign_aligned": aligned_projection_relative_difference(direct_col, cached_col),
                "squared_distance_relative_difference": float(
                    np.linalg.norm(direct_distances - cached_distances)
                    / max(np.linalg.norm(direct_distances), 1e-300)
                ),
                "original_objective": original_objective,
                "direct_objective": direct_objective,
                "corrected_cached_objective": cached_objective,
                "objective_abs_difference_direct_vs_original": abs(direct_objective - original_objective),
                "objective_rel_difference_direct_vs_original": abs(direct_objective - original_objective) / max(abs(direct_objective), abs(original_objective), 1e-300),
                "objective_abs_difference_direct_vs_cached": abs(direct_objective - cached_objective),
                "screened_objective_relative_difference_original_vs_partial": screened_objective_difference,
                "direct_matches_original_accuracy": float(np.mean(direct_predictions == probe_labels)) == float(old["accuracy"]),
                "cached_matches_direct_accuracy": float(np.mean(cached_predictions == probe_labels)) == float(np.mean(direct_predictions == probe_labels)),
                "direct_matches_original_iterations": len(direct_history) == int(old["mda_iterations"]),
                "cached_matches_direct_iterations": len(cached_history) == len(direct_history),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_differences.sort(
        key=lambda row: (
            row.get("train_per_class", 0),
            row.get("repeat", 0),
            row.get("row_dim", 0),
            row.get("col_dim", 0),
        )
    )
    differences_path = args.output_dir / "partial_checkpoint_differences.csv"
    with differences_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(partial_differences[0]) if partial_differences else ["status"])
        writer.writeheader()
        if partial_differences:
            writer.writerows(partial_differences)
        else:
            writer.writerow({"status": "no material differences detected"})
    csv_path = args.output_dir / "targeted_equivalence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["status"])
        writer.writeheader()
        if records:
            writer.writerows(records)
        else:
            writer.writerow({"status": "no configurations required targeted recheck"})

    summary = {
        "normalization": args.normalization,
        "screened_shared_mda22_rows": sum(1 for key in shared if key[3] == "MDA/2-2"),
        "partial_checkpoint_vs_original_accuracy_mismatches": partial_accuracy_mismatches,
        "partial_checkpoint_vs_original_iteration_mismatches": partial_iteration_mismatches,
        "partial_checkpoint_vs_original_objective_relative_difference_max": max(partial_objective_relative_differences, default=0.0),
        "partial_checkpoint_objective_relative_differences_over_1e-8": sum(x > 1e-8 for x in partial_objective_relative_differences),
        "partial_checkpoint_objective_relative_differences_over_1e-7": sum(x > OBJECTIVE_SCREENING_REL_TOL for x in partial_objective_relative_differences),
        "targeted_configurations": len(records),
        "objective_screening_relative_tolerance": OBJECTIVE_SCREENING_REL_TOL,
        "prediction_mismatches_after_correction": sum(r["prediction_mismatches_direct_vs_cached"] for r in records),
        "accuracy_or_iteration_mismatches_direct_vs_original": sum(
            not r["direct_matches_original_accuracy"] or not r["direct_matches_original_iterations"]
            for r in records
        ),
        "accuracy_or_iteration_mismatches_corrected_cache_vs_direct": sum(
            not r["cached_matches_direct_accuracy"] or not r["cached_matches_direct_iterations"]
            for r in records
        ),
        "max_row_projection_relative_difference_sign_aligned": max(
            (r["row_projection_relative_difference_sign_aligned"] for r in records), default=0.0
        ),
        "max_col_projection_relative_difference_sign_aligned": max(
            (r["col_projection_relative_difference_sign_aligned"] for r in records), default=0.0
        ),
        "max_squared_distance_relative_difference": max(
            (r["squared_distance_relative_difference"] for r in records), default=0.0
        ),
        "max_objective_abs_difference_direct_vs_cached": max(
            (r["objective_abs_difference_direct_vs_cached"] for r in records), default=0.0
        ),
        "max_objective_rel_difference_direct_vs_original": max(
            (r["objective_rel_difference_direct_vs_original"] for r in records), default=0.0
        ),
        "partial_differences_csv": differences_path.name,
        "records_csv": csv_path.name,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
