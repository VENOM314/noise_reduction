from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PLOT_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_metric_svg(csv_path: Path, output_path: Path, metric: str, title: str) -> None:
    rows = read_csv(csv_path)
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        groups[row["noise_type"]].append((float(row["K"]), float(row[metric])))
    if not groups:
        return

    width, height = 880, 520
    margin_left, margin_right, margin_top, margin_bottom = 80, 30, 50, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    xs = sorted({x for points in groups.values() for x, _ in points})
    ys = [y for points in groups.values() for _, y in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def sx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111827"/>',
    ]

    for x in xs:
        px = sx(x)
        parts.append(f'<line x1="{px:.1f}" y1="{margin_top + plot_h}" x2="{px:.1f}" y2="{margin_top + plot_h + 6}" stroke="#111827"/>')
        parts.append(f'<text x="{px:.1f}" y="{height - 42}" text-anchor="middle" font-family="Arial" font-size="13">{int(x)}</text>')

    for i in range(5):
        y = y_min + (y_max - y_min) * i / 4
        py = sy(y)
        parts.append(f'<line x1="{margin_left - 6}" y1="{py:.1f}" x2="{margin_left}" y2="{py:.1f}" stroke="#111827"/>')
        parts.append(f'<line x1="{margin_left}" y1="{py:.1f}" x2="{margin_left + plot_w}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y:.2f}</text>')

    parts.append(f'<text x="{width/2}" y="{height - 12}" text-anchor="middle" font-family="Arial" font-size="14">K retained frequencies per window</text>')
    parts.append(f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{metric}</text>')

    for idx, (noise_type, points) in enumerate(sorted(groups.items())):
        color = PLOT_COLORS[idx % len(PLOT_COLORS)]
        points = sorted(points)
        polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        legend_y = margin_top + 22 * idx
        parts.append(f'<rect x="{width - 230}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{width - 212}" y="{legend_y}" font-family="Arial" font-size="13">{noise_type}</text>')

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_summary_bar_svg(csv_path: Path, output_path: Path, metric: str, title: str) -> None:
    rows = read_csv(csv_path)
    if not rows:
        return

    values = [(row["noise_type"], float(row[metric])) for row in rows]
    width, height = 760, 460
    margin_left, margin_right, margin_top, margin_bottom = 80, 40, 54, 92
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_values = [value for _, value in values]
    y_min = min(0.0, min(y_values))
    y_max = max(y_values)
    if y_min == y_max:
        y_max += 1.0
    pad = (y_max - y_min) * 0.10
    y_min -= pad
    y_max += pad

    def sy(y: float) -> float:
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    slot_w = plot_w / len(values)
    bar_w = min(96, slot_w * 0.58)
    zero_y = sy(0.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.1f}" x2="{margin_left + plot_w}" y2="{zero_y:.1f}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111827"/>',
    ]

    for i in range(5):
        y = y_min + (y_max - y_min) * i / 4
        py = sy(y)
        parts.append(f'<line x1="{margin_left - 6}" y1="{py:.1f}" x2="{margin_left}" y2="{py:.1f}" stroke="#111827"/>')
        parts.append(f'<line x1="{margin_left}" y1="{py:.1f}" x2="{margin_left + plot_w}" y2="{py:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y:.2f}</text>')

    for index, (label, value) in enumerate(values):
        cx = margin_left + slot_w * index + slot_w / 2
        top_y = min(sy(value), zero_y)
        bar_h = abs(zero_y - sy(value))
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        parts.append(f'<rect x="{cx - bar_w/2:.1f}" y="{top_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{height - 50}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
        value_y = top_y - 8 if value >= 0 else top_y + bar_h + 16
        parts.append(f'<text x="{cx:.1f}" y="{value_y:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.2f}</text>')

    parts.append(f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{metric}</text>')
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
