#!/usr/bin/env python3

import argparse
import csv
import html
import math
from pathlib import Path


DEFAULT_INPUT = "results/median_comparison.csv"
DEFAULT_OUTPUT = "results/generated_vs_original_max_time_speedup_heatmap.svg"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate an SVG heatmap for generated-vs-original Bcast speedup "
            "using median max rank time."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input median comparison CSV. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output SVG path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--title",
        default="Speedup vs MPI_Bcast",
        help="Plot title.",
    )
    return parser.parse_args()


def resolve_path(path_text, base_dir):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def read_speedups(input_path):
    rows = []
    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "nproc",
            "message_size_bytes",
            "original_over_generated_max_time_sec",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required CSV columns: {', '.join(sorted(missing))}")

        for row in reader:
            value = row["original_over_generated_max_time_sec"]
            if value == "":
                continue
            rows.append(
                {
                    "nproc": int(row["nproc"]),
                    "message_size_bytes": int(row["message_size_bytes"]),
                    "speedup": float(value),
                }
            )

    if not rows:
        raise SystemExit(f"No speedup rows found in {input_path}")
    return rows


def compact_size(num_bytes):
    units = [
        (1024 * 1024, "M"),
        (1024, "K"),
    ]
    for factor, suffix in units:
        if num_bytes >= factor and num_bytes % factor == 0:
            return f"{num_bytes // factor}{suffix}"
    return str(num_bytes)


def lerp(a, b, t):
    return int(round(a + (b - a) * t))


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def color_for_speedup(speedup, max_abs_log2):
    if speedup <= 0 or max_abs_log2 <= 0:
        return "#f7f7f7"

    log_value = math.log(speedup, 2)
    normalized = max(-1.0, min(1.0, log_value / max_abs_log2))

    red = (178, 24, 43)
    white = (247, 247, 247)
    blue = (33, 102, 172)

    if normalized < 0:
        t = normalized + 1.0
        return rgb_to_hex(tuple(lerp(red[i], white[i], t) for i in range(3)))

    t = normalized
    return rgb_to_hex(tuple(lerp(white[i], blue[i], t) for i in range(3)))


def text_color(speedup, max_abs_log2):
    if speedup <= 0 or max_abs_log2 <= 0:
        return "#111111"
    return "#ffffff" if abs(math.log(speedup, 2) / max_abs_log2) > 0.58 else "#111111"


def build_svg(rows, title):
    nprocs = sorted({row["nproc"] for row in rows}, reverse=True)
    sizes = sorted({row["message_size_bytes"] for row in rows})
    speedup_by_cell = {
        (row["nproc"], row["message_size_bytes"]): row["speedup"]
        for row in rows
    }

    speedups = [row["speedup"] for row in rows if row["speedup"] > 0]
    max_abs_log2 = max(abs(math.log(value, 2)) for value in speedups) if speedups else 1.0
    max_abs_log2 = max(max_abs_log2, 0.25)

    cell_w = 92
    cell_h = 58
    left = 110
    top = 72
    right = 165
    bottom = 120
    colorbar_w = 28
    width = left + len(sizes) * cell_w + right
    height = top + len(nprocs) * cell_h + bottom

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(
        f'<text x="{width / 2:.1f}" y="38" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="34" font-weight="700">'
        f'{html.escape(title)}</text>'
    )

    plot_w = len(sizes) * cell_w
    plot_h = len(nprocs) * cell_h

    for y_idx, nproc in enumerate(nprocs):
        y = top + y_idx * cell_h
        svg.append(
            f'<text x="{left - 14}" y="{y + cell_h / 2 + 6:.1f}" text-anchor="end" '
            'font-family="Arial, Helvetica, sans-serif" font-size="18">'
            f'{nproc}</text>'
        )
        for x_idx, size in enumerate(sizes):
            x = left + x_idx * cell_w
            speedup = speedup_by_cell.get((nproc, size))
            fill = "#f7f7f7" if speedup is None else color_for_speedup(speedup, max_abs_log2)
            svg.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{fill}" stroke="#111111" stroke-width="1.5"/>'
            )
            if speedup is not None:
                svg.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 7:.1f}" '
                    f'text-anchor="middle" font-family="Arial, Helvetica, sans-serif" '
                    f'font-size="17" font-weight="700" fill="{text_color(speedup, max_abs_log2)}">'
                    f'{speedup:.2f}x</text>'
                )

    for x_idx, size in enumerate(sizes):
        x = left + x_idx * cell_w + cell_w / 2
        label = compact_size(size)
        svg.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 32}" text-anchor="end" '
            f'transform="rotate(-45 {x:.1f} {top + plot_h + 32})" '
            'font-family="Arial, Helvetica, sans-serif" font-size="17">'
            f'{html.escape(label)}</text>'
        )

    svg.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="26" font-weight="700">'
        'Message Size (bytes)</text>'
    )
    svg.append(
        f'<text x="28" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 28 {top + plot_h / 2:.1f})" '
        'font-family="Arial, Helvetica, sans-serif" font-size="26" font-weight="700">'
        'Number of Processes</text>'
    )

    # Colorbar. Ticks are chosen in ratio space and colored with the same log-scaled map.
    cbar_x = left + plot_w + 58
    cbar_y = top
    steps = 120
    for i in range(steps):
        t0 = i / steps
        log_value = max_abs_log2 * (1.0 - 2.0 * t0)
        speedup = 2 ** log_value
        y = cbar_y + i * plot_h / steps
        svg.append(
            f'<rect x="{cbar_x}" y="{y:.2f}" width="{colorbar_w}" '
            f'height="{plot_h / steps + 0.5:.2f}" '
            f'fill="{color_for_speedup(speedup, max_abs_log2)}" stroke="none"/>'
        )
    svg.append(
        f'<rect x="{cbar_x}" y="{cbar_y}" width="{colorbar_w}" height="{plot_h}" '
        'fill="none" stroke="#111111" stroke-width="1.5"/>'
    )

    tick_values = sorted({0.5, 1.0, 2.0, round(2 ** max_abs_log2, 2)})
    for value in tick_values:
        if value <= 0:
            continue
        normalized = math.log(value, 2) / max_abs_log2
        if normalized < -1 or normalized > 1:
            continue
        y = cbar_y + (1.0 - normalized) * plot_h / 2.0
        svg.append(
            f'<line x1="{cbar_x + colorbar_w}" y1="{y:.1f}" '
            f'x2="{cbar_x + colorbar_w + 7}" y2="{y:.1f}" stroke="#111111" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{cbar_x + colorbar_w + 12}" y="{y + 5:.1f}" '
            'font-family="Arial, Helvetica, sans-serif" font-size="16">'
            f'{value:g}x</text>'
        )

    label_x = cbar_x + colorbar_w + 74
    label_y = cbar_y + plot_h / 2
    svg.append(
        f'<text x="{label_x}" y="{label_y:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 {label_x} {label_y:.1f})" '
        'font-family="Arial, Helvetica, sans-serif" font-size="22">'
        'Speedup: original / generated</text>'
    )

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = resolve_path(args.input, script_dir)
    output_path = resolve_path(args.output, script_dir)

    if not input_path.exists():
        raise SystemExit(
            f"Input CSV not found: {input_path}\n"
            "Run compare_bcast_rule_medians.py first."
        )

    rows = read_speedups(input_path)
    svg = build_svg(rows, args.title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg)

    print(f"Read: {input_path}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
