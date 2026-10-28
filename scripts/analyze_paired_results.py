#!/usr/bin/env python3
"""Summarize paired method differences across matched random splits."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_differences(
    rows: Iterable[dict[str, str]],
    reference_method: str = "MDA/2-2",
    bootstrap_samples: int = 10_000,
    seed: int = 881,
) -> list[dict[str, float | int | str]]:
    """Return reference-minus-comparator accuracy differences in percentage points.

    Bootstrap resampling is over matched split rows. Intervals characterize
    split-generation variability conditional on this fixed dataset; they are
    not population-level confidence intervals or significance tests.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    grouped: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (int(row["train_per_class"]), int(row["seed"]))
        method = row["method"]
        if method in grouped[key]:
            raise ValueError(f"Duplicate method {method!r} for split {key}.")
        grouped[key][method] = float(row["accuracy"])

    by_train: dict[int, list[tuple[int, dict[str, float]]]] = defaultdict(list)
    for (train_count, split_seed), scores in grouped.items():
        if reference_method not in scores:
            raise ValueError(f"Split at train count {train_count} lacks {reference_method!r}.")
        by_train[train_count].append((split_seed, scores))

    rng = np.random.default_rng(seed)
    output: list[dict[str, float | int | str]] = []
    for train_count in sorted(by_train):
        split_scores = [scores for _seed, scores in sorted(by_train[train_count])]
        methods = set(split_scores[0])
        if any(set(scores) != methods for scores in split_scores):
            raise ValueError(f"Methods are not matched for every split at train count {train_count}.")
        if not methods - {reference_method}:
            raise ValueError("At least one comparison method is required.")

        for comparison in sorted(methods - {reference_method}):
            deltas = np.asarray([
                (scores[reference_method] - scores[comparison]) * 100.0
                for scores in split_scores
            ])
            n = len(deltas)
            bootstrap_indices = rng.integers(0, n, size=(bootstrap_samples, n))
            bootstrap_means = deltas[bootstrap_indices].mean(axis=1)
            low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            output.append({
                "train_per_class": train_count,
                "reference_method": reference_method,
                "comparison_method": comparison,
                "splits": n,
                "mean_difference_pp": float(deltas.mean()),
                "sd_difference_pp": float(deltas.std(ddof=1)) if n > 1 else 0.0,
                "bootstrap_95_low_pp": float(low),
                "bootstrap_95_high_pp": float(high),
                "reference_win_fraction": float(np.mean(deltas > 0)),
            })
    return output


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("No paired rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-method", default="MDA/2-2")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=881)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    output = args.output or args.raw_results.with_name("paired_differences.csv")
    rows = paired_differences(
        read_rows(args.raw_results), args.reference_method,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    write_rows(output, rows)
    print(f"Wrote paired split-difference summary to {output.resolve()}")


if __name__ == "__main__":
    main()
