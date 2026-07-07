#!/usr/bin/env python3

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


FILENAME_RE = re.compile(
    r"^auto_bcast_(?P<label>original|generated_single_multinode)_"
    r"ppn(?P<ppn>\d+)_(?P<nproc>\d+)r_(?P<jobid>[^.]+)\.csv$"
)

TIME_COLUMNS = ("avg_time_sec", "max_time_sec", "min_time_sec")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pair original/generated auto Bcast benchmark CSVs by job id and "
            "compare median time per message size."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing auto_bcast_*.csv files, relative to this script by default.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output comparison CSV path. Default: <results-dir>/median_comparison.csv",
    )
    return parser.parse_args()


def resolve_path(path_text, base_dir):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def discover_pairs(results_dir):
    files_by_key = defaultdict(dict)

    for path in sorted(results_dir.glob("auto_bcast_*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue

        key = (
            int(match.group("ppn")),
            int(match.group("nproc")),
            match.group("jobid"),
        )
        files_by_key[key][match.group("label")] = path

    complete_pairs = []
    incomplete = []
    for key, files in sorted(files_by_key.items()):
        if "original" in files and "generated_single_multinode" in files:
            complete_pairs.append((key, files["original"], files["generated_single_multinode"]))
        else:
            incomplete.append((key, files))

    return complete_pairs, incomplete


def read_medians(path):
    values = defaultdict(lambda: {column: [] for column in TIME_COLUMNS})
    correctness = defaultdict(lambda: [0, 0])

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            msg_size = int(row["message_size_bytes"])
            correctness[msg_size][1] += 1
            if row.get("correct", "").lower() == "true":
                correctness[msg_size][0] += 1

            for column in TIME_COLUMNS:
                values[msg_size][column].append(float(row[column]))

    medians = {}
    for msg_size, columns in values.items():
        medians[msg_size] = {
            f"median_{column}": statistics.median(samples)
            for column, samples in columns.items()
        }
        medians[msg_size]["correct_iterations"] = correctness[msg_size][0]
        medians[msg_size]["total_iterations"] = correctness[msg_size][1]

    return medians


def comparison_rows(pair_key, original_path, generated_path):
    ppn, nproc, jobid = pair_key
    original = read_medians(original_path)
    generated = read_medians(generated_path)

    common_sizes = sorted(set(original) & set(generated))
    rows = []

    for msg_size in common_sizes:
        row = {
            "jobid": jobid,
            "ppn": ppn,
            "nproc": nproc,
            "message_size_bytes": msg_size,
            "original_correct_iterations": original[msg_size]["correct_iterations"],
            "generated_correct_iterations": generated[msg_size]["correct_iterations"],
            "total_iterations": min(
                original[msg_size]["total_iterations"],
                generated[msg_size]["total_iterations"],
            ),
        }

        for column in TIME_COLUMNS:
            original_median = original[msg_size][f"median_{column}"]
            generated_median = generated[msg_size][f"median_{column}"]
            delta = generated_median - original_median

            row[f"original_median_{column}"] = original_median
            row[f"generated_median_{column}"] = generated_median
            row[f"generated_minus_original_{column}"] = delta
            row[f"generated_over_original_{column}"] = (
                generated_median / original_median if original_median else ""
            )
            row[f"original_over_generated_{column}"] = (
                original_median / generated_median if generated_median else ""
            )

        rows.append(row)

    return rows


def write_rows(rows, output_path):
    fieldnames = [
        "jobid",
        "ppn",
        "nproc",
        "message_size_bytes",
        "original_correct_iterations",
        "generated_correct_iterations",
        "total_iterations",
    ]

    for column in TIME_COLUMNS:
        fieldnames.extend(
            [
                f"original_median_{column}",
                f"generated_median_{column}",
                f"generated_minus_original_{column}",
                f"generated_over_original_{column}",
                f"original_over_generated_{column}",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_dir = resolve_path(args.results_dir, script_dir)
    output_path = (
        resolve_path(args.output, script_dir)
        if args.output
        else results_dir / "median_comparison.csv"
    )

    pairs, incomplete = discover_pairs(results_dir)
    rows = []
    for pair_key, original_path, generated_path in pairs:
        rows.extend(comparison_rows(pair_key, original_path, generated_path))

    write_rows(rows, output_path)

    print(f"Found complete pairs: {len(pairs)}")
    for (ppn, nproc, jobid), _, _ in pairs:
        print(f"  jobid={jobid} ppn={ppn} nproc={nproc}")

    if incomplete:
        print(f"Skipped incomplete pairs: {len(incomplete)}")
        for (ppn, nproc, jobid), files in incomplete:
            labels = ", ".join(sorted(files))
            print(f"  jobid={jobid} ppn={ppn} nproc={nproc} labels={labels}")

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
