#!/usr/bin/env python3
"""Render an MDA/2-2 dimension sweep without crowding dense grids."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def mix_color(value: float, low: float, high: float) -> str:
    start = (237, 224, 215)
    end = (31, 103, 83)
    ratio = 0.0 if high <= low else (value - low) / (high - low)
    rgb = tuple(round(a + (b - a) * ratio) for a, b in zip(start, end))
    return "#%02x%02x%02x" % rgb


def render(summary_path: Path, output_path: Path, normalization: str) -> None:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == "MDA/2-2"]
    if not rows:
        raise ValueError(f"No MDA/2-2 rows in {summary_path}")

    galleries = sorted({int(row["train_per_class"]) for row in rows})
    row_dims = sorted({int(row["row_dim"]) for row in rows})
    col_dims = sorted({int(row["col_dim"]) for row in rows})
    lookup = {
        (int(row["train_per_class"]), int(row["row_dim"]), int(row["col_dim"])): row
        for row in rows
    }
    scores = [100.0 * float(row["mean_accuracy"]) for row in rows]
    low, high = min(scores), max(scores)
    repeats = max(int(row["repeats"]) for row in rows)
    normalization_label = {
        "unit_l2": "unit-L2",
        "within_scatter": "within-scatter",
    }[normalization]

    width, height = 1320, 1160
    panel_w, panel_h = 630, 510
    left, top = 74, 55
    cell = min(4.0, (panel_h - top - 54) / len(row_dims), (panel_w - left - 150) / len(col_dims))
    grid_w, grid_h = len(col_dims) * cell, len(row_dims) * cell
    row_tick_stride = max(1, len(row_dims) // 12)
    col_tick_stride = max(1, len(col_dims) // 10)
    panels = [(22, 88), (655, 88), (22, 598), (655, 598)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">MDA/2-2 {normalization_label} accuracy across row and column dimensions</title>',
        f'<desc id="desc">Color maps show mean probe accuracy for {len(lookup):,} feasible dimension settings across gallery sizes using {normalization_label} projection axes. Each cell has exact mean and standard deviation in its title tooltip. The outlined cell is the highest probe mean for that gallery size; this selection is optimistic.</desc>',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        '<text x="32" y="36" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#18241f">MDA/2-2 feature-dimension sweep</text>',
        f'<text x="32" y="59" font-family="Arial,sans-serif" font-size="13" fill="#64716a">Original 112×92 ORL · {repeats} matched splits · {normalization_label} axes · corrected Fig. 3 stopping rule · canonical direct grid</text>',
    ]

    for index, gallery in enumerate(galleries):
        px, py = panels[index]
        grid_x, grid_y = px + left, py + top
        panel_rows = [row for row in rows if int(row["train_per_class"]) == gallery]
        best = max(panel_rows, key=lambda row: float(row["mean_accuracy"]))
        best_key = (gallery, int(best["row_dim"]), int(best["col_dim"]))
        parts.append(f'<text x="{px+7}" y="{py+24}" font-family="Arial,sans-serif" font-size="17" font-weight="700" fill="#18241f">{gallery} gallery images per identity</text>')
        parts.append(f'<text x="{grid_x+grid_w/2:.1f}" y="{py+43}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#51616d">Column output dimension</text>')
        parts.append(f'<text x="{px+14}" y="{grid_y+grid_h/2:.1f}" transform="rotate(-90 {px+14},{grid_y+grid_h/2:.1f})" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#51616d">Row output dimension</text>')

        parts.append(f'<rect x="{grid_x}" y="{grid_y}" width="{grid_w:.1f}" height="{grid_h:.1f}" fill="#eef0ed"/>')
        for ri, row_dim in enumerate(row_dims):
            for ci, col_dim in enumerate(col_dims):
                key = (gallery, row_dim, col_dim)
                row = lookup.get(key)
                if row is None:
                    continue
                score = 100.0 * float(row["mean_accuracy"])
                std = 100.0 * float(row["std_accuracy"])
                x, y = grid_x + ci * cell, grid_y + ri * cell
                title = (f"{gallery} gallery images/person; row={row_dim}; column={col_dim}; "
                         f"mean={score:.2f}%; SD={std:.2f}%; n={int(row['repeats'])} splits")
                if key == best_key:
                    parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell:.2f}" height="{cell:.2f}" fill="{mix_color(score, low, high)}" stroke="#111b17" stroke-width="1.1"><title>{title}; highest mean on these probe splits (optimistic)</title></rect>')
                else:
                    parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell:.2f}" height="{cell:.2f}" fill="{mix_color(score, low, high)}"><title>{title}</title></rect>')

        # Label every tenth dimension only; cell values are available on hover.
        for ci, col_dim in enumerate(col_dims):
            if ci == 0 or ci % col_tick_stride == 0:
                x = grid_x + ci * cell + cell / 2
                parts.append(f'<text x="{x:.1f}" y="{grid_y+grid_h+13:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="8" fill="#51616d">{col_dim}</text>')
        for ri, row_dim in enumerate(row_dims):
            if ri == 0 or ri % row_tick_stride == 0:
                y = grid_y + ri * cell + cell * 0.75
                parts.append(f'<text x="{grid_x-5}" y="{y:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="8" fill="#51616d">{row_dim}</text>')

        best_score = 100.0 * float(best["mean_accuracy"])
        parts.append(f'<text x="{grid_x+grid_w+12:.1f}" y="{grid_y+14:.1f}" font-family="Arial,sans-serif" font-size="9" fill="#24332d">Max probe mean</text>')
        parts.append(f'<text x="{grid_x+grid_w+12:.1f}" y="{grid_y+28:.1f}" font-family="Arial,sans-serif" font-size="10" font-weight="700" fill="#24332d">{best_score:.2f}%</text>')
        parts.append(f'<text x="{grid_x+grid_w+12:.1f}" y="{grid_y+42:.1f}" font-family="Arial,sans-serif" font-size="9" fill="#51616d">{best_key[1]} × {best_key[2]}</text>')

    legend_x, legend_y = 1125, 230
    legend_h, legend_w = 230, 14
    for step in range(legend_w):
        value = low + (high - low) * step / max(legend_w - 1, 1)
        parts.append(f'<rect x="{legend_x+step*8}" y="{legend_y}" width="9" height="{legend_h}" fill="{mix_color(value, low, high)}"/>')
    parts.extend([
        f'<text x="{legend_x}" y="{legend_y-13}" font-family="Arial,sans-serif" font-size="11" fill="#51616d">Mean accuracy (%)</text>',
        f'<text x="{legend_x}" y="{legend_y+legend_h+18}" font-family="Arial,sans-serif" font-size="10" fill="#51616d">{low:.1f}%</text>',
        f'<text x="{legend_x+legend_w*8}" y="{legend_y+legend_h+18}" text-anchor="end" font-family="Arial,sans-serif" font-size="10" fill="#51616d">{high:.1f}%</text>',
        '<rect x="1125" y="500" width="14" height="14" fill="#eef0ed" stroke="#98a29c"/><text x="1148" y="511" font-family="Arial,sans-serif" font-size="10" fill="#51616d">Theorem 2 infeasible</text>',
        '<rect x="1125" y="526" width="14" height="14" fill="#8a9b91" stroke="#111b17" stroke-width="1.1"/><text x="1148" y="537" font-family="Arial,sans-serif" font-size="10" fill="#51616d">Highest probe mean</text>',
        '<text x="32" y="1140" font-family="Arial,sans-serif" font-size="12" fill="#9a4b32">Outlined maxima are selected on the same probe splits and are optimistic diagnostics, not unbiased test estimates.</text>',
        '</svg>',
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--normalization",
        choices=("unit_l2", "within_scatter"),
        required=True,
        help="Projection-axis normalization used to generate the input summary.",
    )
    args = parser.parse_args()
    render(args.summary, args.output, args.normalization)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
