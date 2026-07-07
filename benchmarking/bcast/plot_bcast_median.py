#!/usr/bin/env python3
"""Plot median MPI_Bcast timings from benchmark CSV files.

Rows from warmup rounds are skipped by default. For every
(ppn, nproc, algorithm, message size) group, the script computes the median of
the selected timing column across actual measured rounds, then writes one PNG
per (ppn, nproc) with message size on the x-axis and time on the y-axis.
"""

import argparse
import csv
import math
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple


GroupKey = Tuple[int, int, str, int]
PlotKey = Tuple[int, int]
LOG_Y_TICK_MULTIPLIERS = (1.0, 2.0, 5.0)
DEFAULT_FONT_SIZE = 18
TITLE_FONT_SIZE = 20


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Compute actual-round median Bcast timings and plot one log-scale "
            "PNG per ppn/nproc group."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help=(
            "Input timing CSV files. If omitted, uses "
            "benchmarking/bcast/bcast_bench_ppn*r_*.csv, excluding "
            "*_collective_counts.csv."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=script_dir / "plots",
        help="Directory for PNG plots and the median summary CSV.",
    )
    parser.add_argument(
        "--time-column",
        default="avg_latency_sec",
        help="Timing column to aggregate. Common choices: avg_latency_sec, max_time_sec.",
    )
    parser.add_argument(
        "--phase",
        default="actual",
        help="Benchmark phase to plot. Defaults to actual, which skips warmup rows.",
    )
    parser.add_argument(
        "--ppn",
        type=int,
        default=None,
        help=(
            "Ranks per node to use when input CSVs do not contain a ppn column "
            "and ppn cannot be parsed from the filename."
        ),
    )
    parser.add_argument(
        "--x-scale",
        choices=("log", "linear"),
        default="log",
        help="Scale for the message-size axis.",
    )
    parser.add_argument(
        "--y-scale",
        choices=("log", "linear"),
        default="log",
        help="Scale for the timing axis.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=(
            "Path for median summary CSV. Defaults to "
            "<output-dir>/bcast_median_summary.csv."
        ),
    )
    parser.add_argument(
        "--no-summary-csv",
        action="store_true",
        help="Do not write the median summary CSV.",
    )
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Include rows whose correct column is false/0.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=DEFAULT_FONT_SIZE,
        help="Base font size for plot text.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG output DPI.")
    return parser.parse_args()


def default_csv_files() -> List[Path]:
    return sorted(
        path
        for path in Path(__file__).resolve().parent.glob("bcast_bench_ppn*r_*.csv")
        if is_timing_csv(path)
    )


def is_timing_csv(path: Path) -> bool:
    return path.suffix == ".csv" and not path.name.endswith("_collective_counts.csv")


def parse_int(value: str, column: str, path: Path, line_number: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid integer in {column}: {value!r}") from exc


def parse_float(value: str, column: str, path: Path, line_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: invalid float in {column}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{line_number}: non-finite value in {column}: {value!r}")
    return parsed


def parse_ppn(row: Dict[str, str], path: Path, fallback_ppn: Optional[int]) -> int:
    ppn_value = row.get("ppn", "").strip()
    if ppn_value:
        return int(ppn_value)

    match = re.search(r"(?:^|_)ppn(\d+)(?:_|$)", path.name)
    if match:
        return int(match.group(1))

    if fallback_ppn is not None:
        return fallback_ppn

    raise ValueError(
        f"{path}: could not determine ppn. Add a ppn column, use filenames like "
        "bcast_bench_ppn32_512r_123.csv, or pass --ppn."
    )


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def human_bytes(value: float, _position: int) -> str:
    if value < 1024:
        return f"{int(value)} B"
    units = ("KiB", "MiB", "GiB")
    scaled = float(value)
    unit = "B"
    for unit in units:
        scaled /= 1024.0
        if scaled < 1024.0:
            break
    if scaled.is_integer():
        return f"{int(scaled)} {unit}"
    return f"{scaled:.1f} {unit}"


def human_seconds(value: float, _position: int = 0) -> str:
    if value == 0.0:
        return "0"

    abs_value = abs(value)
    if abs_value < 1e-6:
        scaled = value * 1e9
        unit = "ns"
    elif abs_value < 1e-3:
        scaled = value * 1e6
        unit = "us"
    elif abs_value < 1.0:
        scaled = value * 1e3
        unit = "ms"
    else:
        scaled = value
        unit = "s"

    if abs(scaled) >= 100.0:
        label = f"{scaled:.0f}"
    elif abs(scaled) >= 10.0:
        label = f"{scaled:.1f}".rstrip("0").rstrip(".")
    else:
        label = f"{scaled:.2f}".rstrip("0").rstrip(".")
    return f"{label} {unit}"


def log_tick_values(min_value: float, max_value: float) -> List[float]:
    if min_value <= 0.0 or max_value <= 0.0:
        return []

    min_decade = int(math.floor(math.log10(min_value)))
    max_decade = int(math.ceil(math.log10(max_value)))
    ticks: List[float] = []
    lower = min_value / 1.05
    upper = max_value * 1.05

    for decade in range(min_decade, max_decade + 1):
        base = 10.0 ** decade
        for multiplier in LOG_Y_TICK_MULTIPLIERS:
            tick = multiplier * base
            if lower <= tick <= upper:
                ticks.append(tick)

    return ticks


def padded_axis_range(values: Sequence[float], scale: str) -> Tuple[float, float]:
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        if scale == "log":
            return min_value / 1.5, max_value * 1.5
        padding = abs(min_value) * 0.1 if min_value != 0.0 else 1.0
        return min_value - padding, max_value + padding

    if scale == "log":
        padding_factor = 1.12
        return min_value / padding_factor, max_value * padding_factor

    padding = (max_value - min_value) * 0.06
    return min_value - padding, max_value + padding


def gnuplot_quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def read_timings(
    csv_files: Sequence[Path],
    phase: str,
    time_column: str,
    fallback_ppn: Optional[int],
    include_incorrect: bool,
    x_scale: str,
    y_scale: str,
) -> Tuple[DefaultDict[GroupKey, List[float]], List[str], Dict[str, int]]:
    timings: DefaultDict[GroupKey, List[float]] = defaultdict(list)
    algorithm_order: List[str] = []
    seen_algorithms = set()
    stats = {
        "files": 0,
        "rows": 0,
        "used": 0,
        "wrong_phase": 0,
        "incorrect": 0,
        "nonpositive": 0,
        "skipped_files": 0,
    }
    required_columns = {"phase", "algorithm", "nproc", "message_size_bytes", time_column}

    for path in csv_files:
        if not is_timing_csv(path):
            stats["skipped_files"] += 1
            continue

        stats["files"] += 1
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"{path}: missing required column(s): {missing}")

            for line_number, row in enumerate(reader, start=2):
                stats["rows"] += 1
                if row["phase"].strip().lower() != phase.lower():
                    stats["wrong_phase"] += 1
                    continue
                if not include_incorrect and "correct" in row and not is_true(row["correct"]):
                    stats["incorrect"] += 1
                    continue

                algorithm = row["algorithm"].strip()
                if algorithm not in seen_algorithms:
                    seen_algorithms.add(algorithm)
                    algorithm_order.append(algorithm)

                nproc = parse_int(row["nproc"], "nproc", path, line_number)
                ppn = parse_ppn(row, path, fallback_ppn)
                message_size = parse_int(
                    row["message_size_bytes"], "message_size_bytes", path, line_number
                )
                time_value = parse_float(row[time_column], time_column, path, line_number)

                if (x_scale == "log" and message_size <= 0) or (
                    y_scale == "log" and time_value <= 0.0
                ):
                    stats["nonpositive"] += 1
                    continue

                timings[(ppn, nproc, algorithm, message_size)].append(time_value)
                stats["used"] += 1

    return timings, algorithm_order, stats


def summarize_timings(
    timings: DefaultDict[GroupKey, List[float]]
) -> Dict[PlotKey, Dict[str, List[Tuple[int, float, int]]]]:
    summary: Dict[PlotKey, Dict[str, List[Tuple[int, float, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (ppn, nproc, algorithm, message_size), values in timings.items():
        summary[(ppn, nproc)][algorithm].append((message_size, median(values), len(values)))

    for algorithms in summary.values():
        for points in algorithms.values():
            points.sort(key=lambda item: item[0])

    return summary


def write_summary_csv(
    summary: Dict[PlotKey, Dict[str, List[Tuple[int, float, int]]]],
    algorithm_order: Sequence[str],
    path: Path,
    time_column: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ppn",
                "nproc",
                "algorithm",
                "message_size_bytes",
                f"median_{time_column}",
                "samples",
            ]
        )
        for ppn, nproc in sorted(summary):
            algorithms = summary[(ppn, nproc)]
            ordered_algorithms = [name for name in algorithm_order if name in algorithms]
            ordered_algorithms.extend(sorted(set(algorithms) - set(ordered_algorithms)))
            for algorithm in ordered_algorithms:
                for message_size, median_time, samples in algorithms[algorithm]:
                    writer.writerow([ppn, nproc, algorithm, message_size, median_time, samples])


def save_plots_matplotlib(
    summary: Dict[PlotKey, Dict[str, List[Tuple[int, float, int]]]],
    algorithm_order: Sequence[str],
    output_dir: Path,
    time_column: str,
    x_scale: str,
    y_scale: str,
    dpi: int,
    font_size: int,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: List[Path] = []

    for ppn, nproc in sorted(summary):
        algorithms = summary[(ppn, nproc)]
        ordered_algorithms = [name for name in algorithm_order if name in algorithms]
        ordered_algorithms.extend(sorted(set(algorithms) - set(ordered_algorithms)))

        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        all_message_sizes = set()
        all_times = []
        for algorithm in ordered_algorithms:
            points = algorithms[algorithm]
            x_values = [message_size for message_size, _median_time, _samples in points]
            y_values = [median_time for _message_size, median_time, _samples in points]
            all_message_sizes.update(x_values)
            all_times.extend(y_values)
            ax.plot(x_values, y_values, marker="o", linewidth=2.6, markersize=6, label=algorithm)

        ax.set_title(
            f"MPI_Bcast Median Actual-Round Time (nproc={nproc}, ppn={ppn})",
            fontsize=font_size + 2,
        )
        ax.set_xlabel("Message size", fontsize=font_size)
        ax.set_ylabel(f"Median {time_column} (seconds)", fontsize=font_size)
        ax.set_xscale(x_scale, base=2) if x_scale == "log" else ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        if all_times:
            ax.set_ylim(*padded_axis_range(all_times, y_scale))
        ax.xaxis.set_major_formatter(FuncFormatter(human_bytes))
        ax.yaxis.set_major_formatter(FuncFormatter(human_seconds))
        ax.tick_params(axis="both", which="major", labelsize=font_size - 2)
        ax.tick_params(axis="both", which="minor", labelsize=font_size - 4)
        if len(all_message_sizes) <= 24:
            ax.set_xticks(sorted(all_message_sizes))
        if y_scale == "log" and all_times:
            y_ticks = log_tick_values(min(all_times), max(all_times))
            if y_ticks:
                ax.set_yticks(y_ticks)
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.65)
        ax.legend(
            title="Algorithm",
            fontsize=max(font_size - 4, 10),
            title_fontsize=max(font_size - 3, 11),
            loc="best",
        )

        output_path = output_dir / f"bcast_median_nproc{nproc}_ppn{ppn}.png"
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        plot_paths.append(output_path)

    return plot_paths


def save_plots_gnuplot(
    summary: Dict[PlotKey, Dict[str, List[Tuple[int, float, int]]]],
    algorithm_order: Sequence[str],
    output_dir: Path,
    time_column: str,
    x_scale: str,
    y_scale: str,
    dpi: int,
    font_size: int,
) -> List[Path]:
    gnuplot = shutil.which("gnuplot")
    if gnuplot is None:
        raise RuntimeError(
            "Plotting requires either matplotlib or gnuplot. Install matplotlib "
            "or put gnuplot on PATH."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: List[Path] = []
    image_width = 12 * dpi
    image_height = 7 * dpi

    for ppn, nproc in sorted(summary):
        algorithms = summary[(ppn, nproc)]
        ordered_algorithms = [name for name in algorithm_order if name in algorithms]
        ordered_algorithms.extend(sorted(set(algorithms) - set(ordered_algorithms)))
        output_path = output_dir / f"bcast_median_nproc{nproc}_ppn{ppn}.png"
        all_message_sizes = sorted(
            {
                message_size
                for points in algorithms.values()
                for message_size, _median_time, _samples in points
            }
        )
        all_times = [
            median_time
            for points in algorithms.values()
            for _message_size, median_time, _samples in points
        ]

        with tempfile.TemporaryDirectory(prefix="bcast_plot_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            plot_parts = []
            for index, algorithm in enumerate(ordered_algorithms):
                data_path = tmp_dir / f"series_{index}.dat"
                with data_path.open("w") as data_file:
                    for message_size, median_time, _samples in algorithms[algorithm]:
                        data_file.write(f"{message_size} {median_time:.17g}\n")
                plot_parts.append(
                    "{} using 1:2 with linespoints linewidth 3 pointsize 1.25 title {}".format(
                        gnuplot_quote(str(data_path)),
                        gnuplot_quote(algorithm),
                    )
                )

            xtics = ""
            if len(all_message_sizes) <= 24:
                tic_parts = [
                    f"{gnuplot_quote(human_bytes(float(message_size), 0))} {message_size}"
                    for message_size in all_message_sizes
                ]
                xtics = "set xtics rotate by -45 right (" + ", ".join(tic_parts) + ")\n"

            ytics = ""
            if y_scale == "log" and all_times:
                y_tick_values = log_tick_values(min(all_times), max(all_times))
                if y_tick_values:
                    tic_parts = [
                        f"{gnuplot_quote(human_seconds(tick))} {tick:.17g}"
                        for tick in y_tick_values
                    ]
                    ytics = "set ytics (" + ", ".join(tic_parts) + ")\nset mytics 10\n"

            yrange = ""
            if all_times:
                y_min, y_max = padded_axis_range(all_times, y_scale)
                yrange = f"set yrange [{y_min:.17g}:{y_max:.17g}]\n"

            script = (
                f"set terminal pngcairo size {image_width},{image_height} noenhanced "
                f"font 'Arial,{font_size}'\n"
                f"set output {gnuplot_quote(str(output_path))}\n"
                f"set title {gnuplot_quote(f'MPI_Bcast Median Actual-Round Time (nproc={nproc}, ppn={ppn})')}\n"
                f"set title font ',{font_size + 2}'\n"
                "set xlabel 'Message size'\n"
                f"set xlabel font ',{font_size}'\n"
                f"set ylabel {gnuplot_quote(f'Median {time_column} (seconds)')}\n"
                f"set ylabel font ',{font_size}'\n"
                "set grid xtics ytics mytics\n"
                f"set key outside right top font ',{max(font_size - 3, 10)}'\n"
                f"{'set logscale x 2' if x_scale == 'log' else 'unset logscale x'}\n"
                f"{'set logscale y' if y_scale == 'log' else 'unset logscale y'}\n"
                f"{yrange}"
                f"{xtics}"
                f"{ytics}"
                "plot "
                + ", \\\n     ".join(plot_parts)
                + "\n"
            )

            script_path = tmp_dir / "plot.gp"
            script_path.write_text(script)
            completed = subprocess.run(
                [gnuplot, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "gnuplot failed for {}:\n{}".format(output_path, completed.stderr.strip())
                )

        plot_paths.append(output_path)

    return plot_paths


def save_plots(
    summary: Dict[PlotKey, Dict[str, List[Tuple[int, float, int]]]],
    algorithm_order: Sequence[str],
    output_dir: Path,
    time_column: str,
    x_scale: str,
    y_scale: str,
    dpi: int,
    font_size: int,
) -> Tuple[List[Path], str]:
    try:
        return (
            save_plots_matplotlib(
                summary=summary,
                algorithm_order=algorithm_order,
                output_dir=output_dir,
                time_column=time_column,
                x_scale=x_scale,
                y_scale=y_scale,
                dpi=dpi,
                font_size=font_size,
            ),
            "matplotlib",
        )
    except ImportError:
        return (
            save_plots_gnuplot(
                summary=summary,
                algorithm_order=algorithm_order,
                output_dir=output_dir,
                time_column=time_column,
                x_scale=x_scale,
                y_scale=y_scale,
                dpi=dpi,
                font_size=font_size,
            ),
            "gnuplot",
        )


def main() -> int:
    args = parse_args()
    csv_files = sorted(args.csv_files) if args.csv_files else default_csv_files()
    if not csv_files:
        raise SystemExit("No input CSV files found.")

    missing_files = [str(path) for path in csv_files if not path.is_file()]
    if missing_files:
        raise SystemExit("Input CSV file(s) not found: " + ", ".join(missing_files))

    timings, algorithm_order, stats = read_timings(
        csv_files=csv_files,
        phase=args.phase,
        time_column=args.time_column,
        fallback_ppn=args.ppn,
        include_incorrect=args.include_incorrect,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
    )
    if not timings:
        raise SystemExit("No usable timing rows found after filtering.")

    summary = summarize_timings(timings)
    plot_paths, renderer = save_plots(
        summary=summary,
        algorithm_order=algorithm_order,
        output_dir=args.output_dir,
        time_column=args.time_column,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        dpi=args.dpi,
        font_size=args.font_size,
    )

    summary_path = args.summary_csv
    if summary_path is None:
        summary_path = args.output_dir / "bcast_median_summary.csv"
    if not args.no_summary_csv:
        write_summary_csv(summary, algorithm_order, summary_path, args.time_column)

    print(
        "Read {files} file(s), scanned {rows} row(s), used {used} {phase} row(s).".format(
            phase=args.phase,
            **stats,
        )
    )
    if stats["incorrect"]:
        print(f"Skipped {stats['incorrect']} incorrect row(s).")
    if stats["nonpositive"]:
        print(f"Skipped {stats['nonpositive']} non-positive row(s) for log-scale plotting.")
    if stats["skipped_files"]:
        print(f"Skipped {stats['skipped_files']} non-timing CSV file(s).")
    print(f"Rendered plots with {renderer}.")
    for path in plot_paths:
        print(f"Wrote plot: {path}")
    if not args.no_summary_csv:
        print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
