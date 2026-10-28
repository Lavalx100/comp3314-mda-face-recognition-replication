#!/usr/bin/env python3
"""Render the repeated-split MDA/2-2 dimension surface as a standalone SVG."""

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


def render(summary_path: Path, output_path: Path) -> None:
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == "MDA/2-2"]
    if not rows:
        raise ValueError(f"No MDA/2-2 rows in {summary_path}")

    galleries = sorted({int(row["train_per_class"]) for row in rows})
    row_dims = sorted({int(row["row_dim"]) for row in rows})
    col_dims = sorted({int(row["col_dim"]) for row in rows})
    lookup = {
        (int(row["train_per_class"]), int(row["row_dim"]), int(row["col_dim"])):
        100.0 * float(row["mean_accuracy"])
        for row in rows
    }
    low = min(lookup.values())
    high = max(lookup.values())

    width, height = 1260, 1120
    panel_w, panel_h = 610, 500
    cell = min(34, 390 / max(len(col_dims), len(row_dims)))
    left, top = 105, 76
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">MDA/2-2 accuracy across projection dimensions</title>',
        '<desc id="desc">Mean probe accuracy across five repeated ORL splits for each row and column output dimension and gallery size. Test-selected maxima are optimistic.</desc>',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        '<text x="40" y="39" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#18241f">MDA/2-2 feature-dimension sweep</text>',
        '<text x="40" y="62" font-family="Arial,sans-serif" font-size="13" fill="#64716a">Original 112×92 ORL · 5 matched splits per gallery size · unit-L2 projections · paper Fig. 3 stopping rule</text>',
    ]

    for p, gallery in enumerate(galleries):
        px = 26 + (p % 2) * panel_w
        py = 90 + (p // 2) * panel_h
        grid_x, grid_y = px + left, py + top
        parts.append(f'<text x="{px+8}" y="{py+28}" font-family="Arial,sans-serif" font-size="18" font-weight="700" fill="#18241f">{gallery} gallery images per identity</text>')
        parts.append(f'<text x="{grid_x + len(col_dims)*cell/2:.1f}" y="{py+55}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#51616d">Column output dimension</text>')
        parts.append(f'<text x="{px+17}" y="{grid_y + len(row_dims)*cell/2:.1f}" transform="rotate(-90 {px+17},{grid_y + len(row_dims)*cell/2:.1f})" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#51616d">Row output dimension</text>')

        for ri, row_dim in enumerate(row_dims):
            for ci, col_dim in enumerate(col_dims):
                key = (gallery, row_dim, col_dim)
                if key not in lookup:
                    continue
                score = lookup[key]
                x, y = grid_x + ci * cell, grid_y + ri * cell
                text_color = "#ffffff" if score >= (low + high) / 2 else "#24332d"
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-1:.1f}" height="{cell-1:.1f}" rx="2" fill="{mix_color(score, low, high)}"/>')
                parts.append(f'<text x="{x+(cell-1)/2:.1f}" y="{y+cell*0.62:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="8.5" fill="{text_color}">{score:.0f}</text>')

        for ci, col_dim in enumerate(col_dims):
            x = grid_x + ci * cell + cell / 2
            parts.append(f'<text x="{x:.1f}" y="{grid_y+len(row_dims)*cell+15:.1f}" text-anchor="end" transform="rotate(-55 {x:.1f},{grid_y+len(row_dims)*cell+15:.1f})" font-family="Arial,sans-serif" font-size="8" fill="#51616d">{col_dim}</text>')
        for ri, row_dim in enumerate(row_dims):
            y = grid_y + ri * cell + cell * 0.65
            parts.append(f'<text x="{grid_x-8}" y="{y:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="8" fill="#51616d">{row_dim}</text>')

    legend_x, legend_y = 1120, 196
    legend_h, legend_w = 240, 14
    for step in range(legend_w):
        value = low + (high - low) * step / max(legend_w - 1, 1)
        parts.append(f'<rect x="{legend_x+step*8}" y="{legend_y}" width="9" height="{legend_h}" fill="{mix_color(value, low, high)}"/>')
    parts.extend([
        f'<text x="{legend_x}" y="{legend_y-12}" font-family="Arial,sans-serif" font-size="11" fill="#51616d">Mean probe accuracy (%)</text>',
        f'<text x="{legend_x}" y="{legend_y+legend_h+18}" font-family="Arial,sans-serif" font-size="10" fill="#51616d">{low:.0f}%</text>',
        f'<text x="{legend_x+legend_w*8}" y="{legend_y+legend_h+18}" text-anchor="end" font-family="Arial,sans-serif" font-size="10" fill="#51616d">{high:.0f}%</text>',
        '<text x="40" y="1090" font-family="Arial,sans-serif" font-size="12" fill="#9a4b32">Each panel uses the common color scale. Selecting the highest probe cell is optimistic; use the full curve and held-out validation for final model selection.</text>',
        '</svg>',
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.summary, args.output)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
