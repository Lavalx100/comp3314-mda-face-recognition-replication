#!/usr/bin/env python3
"""Compare 2-vs-3 image galleries on nested ORL splits at a fixed 6x6 MDA size.

This is a focused diagnostic for split variation, not a replacement for the
paper-style random gallery protocol or a dimension search. For each repeat,
the 2-image gallery is a strict subset of the 3-image gallery, and 6x6 is
fixed before evaluating probe accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orl_data import load_original_orl  # noqa: E402
from run_experiment import (  # noqa: E402
    equalize_histogram,
    fit_mda,
    make_class_grouping,
    predict_nearest_neighbor,
)


def build_nested_orders(labels: np.ndarray, repeats: int, seed: int) -> list[dict[int, np.ndarray]]:
    orders = []
    class_labels = np.unique(labels)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        orders.append({
            int(label): rng.permutation(np.flatnonzero(labels == label))
            for label in class_labels
        })
    return orders


def evaluate(args: argparse.Namespace) -> None:
    images, labels = load_original_orl(args.archive)
    images = equalize_histogram(images, 64)
    orders = build_nested_orders(labels, args.repeats, args.seed)
    jobs = [
        (gallery, repeat, normalization)
        for gallery in (2, 3)
        for repeat in range(args.repeats)
        for normalization in ("unit_l2", "within_scatter")
    ]

    def run_one(job: tuple[int, int, str]) -> dict:
        gallery, repeat, normalization = job
        train_indices = np.concatenate([
            orders[repeat][int(label)][:gallery]
            for label in np.unique(labels)
        ])
        probe_indices = np.concatenate([
            orders[repeat][int(label)][gallery:]
            for label in np.unique(labels)
        ])
        x_train, y_train = images[train_indices], labels[train_indices]
        x_probe, y_probe = images[probe_indices], labels[probe_indices]
        grouping = make_class_grouping(y_train)
        started = time.perf_counter()
        train_features, probe_features, history = fit_mda(
            x_train,
            y_train,
            x_probe,
            components=(6, 6),
            max_iter=10,
            tolerance=1e-4,
            projection_normalization=normalization,
            stopping_rule="paper_figure",
            mode_order="row_first",
            paper_epsilon=1e-4,
            track_objective_history=False,
            scatter_grouping=grouping,
        )
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = predict_nearest_neighbor(train_features, y_train, probe_features)
        classification_seconds = time.perf_counter() - started
        return {
            "gallery_per_identity": gallery,
            "repeat": repeat + 1,
            "seed": args.seed + repeat,
            "normalization": normalization,
            "row_dim": 6,
            "col_dim": 6,
            "training_images": len(train_indices),
            "probe_images": len(probe_indices),
            "accuracy": float(np.mean(prediction == y_probe)),
            "fit_seconds": fit_seconds,
            "classification_seconds": classification_seconds,
            "iterations": len(history),
            "final_training_fisher_ratio": history[-1],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(run_one, jobs))
    rows.sort(key=lambda row: (row["normalization"], row["gallery_per_identity"], row["repeat"]))
    with (args.output_dir / "raw_nested_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["normalization"], row["gallery_per_identity"])].append(row["accuracy"])
    summary = []
    for (normalization, gallery), values in sorted(grouped.items()):
        summary.append({
            "normalization": normalization,
            "gallery_per_identity": gallery,
            "row_dim": 6,
            "col_dim": 6,
            "repeats": len(values),
            "mean_accuracy": float(np.mean(values)),
            "std_accuracy": float(np.std(values, ddof=1)),
        })
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    metadata = {
        "dataset": "Original Cambridge ORL PGM archive",
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "image_shape": [112, 92],
        "histogram_equalization_levels": 64,
        "gallery_sizes": [2, 3],
        "repeats": args.repeats,
        "repeat_seed_rule": "seed + zero_based_repeat",
        "base_seed": args.seed,
        "gallery_design": "Nested per identity: gallery size 2 is a subset of gallery size 3; remaining images are probes.",
        "dimension": [6, 6],
        "dimension_selection": "None; fixed before evaluating probe labels.",
        "normalizations": ["unit_l2", "within_scatter"],
        "mda_update_order": "row_first",
        "mda_stopping": "paper_figure; epsilon=1e-4; maximum 10 iterations",
        "classifier": "Euclidean 1-nearest-neighbour",
        "workers": args.workers,
        "interpretation": "Targeted split-pairing diagnostic; not the paper-style independent-seed full-grid protocol.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    for normalization in ("unit_l2", "within_scatter"):
        by_repeat = {
            row["repeat"]: row["accuracy"]
            for row in rows
            if row["normalization"] == normalization and row["gallery_per_identity"] == 2
        }
        three = {
            row["repeat"]: row["accuracy"]
            for row in rows
            if row["normalization"] == normalization and row["gallery_per_identity"] == 3
        }
        differences = np.asarray([three[repeat] - by_repeat[repeat] for repeat in sorted(by_repeat)])
        print(
            f"{normalization}: g2={np.mean(list(by_repeat.values())):.4f}, "
            f"g3={np.mean(list(three.values())):.4f}, "
            f"paired g3-g2={differences.mean():+.4f} "
            f"(SD {differences.std(ddof=1):.4f})",
            flush=True,
        )
    print(f"Saved nested diagnostic to {args.output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(".cache/orl-original/att_faces.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-11-09-nested-gallery-diagnostic"))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3314)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 2 or args.workers < 1:
        parser.error("repeats must be at least 2 and workers must be positive")
    if not args.archive.is_file():
        parser.error(f"ORL archive not found: {args.archive}")
    evaluate(args)


if __name__ == "__main__":
    main()
