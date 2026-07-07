#!/usr/bin/env python3
"""Fit an sklearn regression tree to median MPI_Bcast timings."""

import argparse
import csv
import math
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz


def parse_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Fit an sklearn regression tree to Bcast median timings.")
    parser.add_argument(
        "input_csvs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Input median summary CSV or raw timing CSV files. Raw split files named "
            "*_collective_counts.csv are ignored. If omitted, the default summary "
            "matching --time-column is used when available."
        ),
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=script_dir / "models")
    parser.add_argument(
        "--time-column",
        default="max_time_sec",
        help=(
            "Timing column to fit. Defaults to max_time_sec so the model targets "
            "the per-iteration maximum latency across ranks."
        ),
    )
    parser.add_argument("--phase", default="actual")
    parser.add_argument("--ppn", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-depth-candidates", default="2,3,4,5,6,7,8,9,10")
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=20)
    args = parser.parse_args()
    if not args.input_csvs:
        args.input_csvs = default_input_csvs(script_dir, args.time_column)
    return args


def is_collective_counts_csv(path):
    return path.suffix == ".csv" and path.name.endswith("_collective_counts.csv")


def summary_has_column(path, column):
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return False
    return column in header


def default_input_csvs(script_dir, time_column="max_time_sec"):
    summary_column = f"median_{time_column}"
    summary_candidates = []
    if time_column == "max_time_sec":
        summary_candidates.extend(
            [
                script_dir / "plots_max_latency" / "bcast_median_summary.csv",
                script_dir / "plots" / "bcast_median_summary.csv",
            ]
        )
    elif time_column == "avg_latency_sec":
        summary_candidates.extend(
            [
                script_dir / "plots" / "bcast_median_summary.csv",
                script_dir / "plots_max_latency" / "bcast_median_summary.csv",
            ]
        )
    else:
        summary_candidates.extend(
            [
                script_dir / "plots" / "bcast_median_summary.csv",
                script_dir / "plots_max_latency" / "bcast_median_summary.csv",
            ]
        )

    for summary in summary_candidates:
        if summary_has_column(summary, summary_column):
            return [summary]

    return sorted(
        path
        for path in script_dir.glob("bcast_bench_ppn*r_*.csv")
        if not is_collective_counts_csv(path)
    )


def ppn_from_row_or_name(row, path, fallback):
    if row.get("ppn"):
        return int(row["ppn"])
    match = re.search(r"(?:^|_)ppn(\d+)(?:_|$)", path.name)
    if match:
        return int(match.group(1))
    if fallback is not None:
        return fallback
    raise ValueError(f"{path}: could not determine ppn; pass --ppn")


def load_rows(paths, time_column, phase, fallback_ppn):
    rows = []
    grouped = defaultdict(list)
    summary_col = f"median_{time_column}"
    timing_columns = {"phase", "algorithm", "nproc", "message_size_bytes", time_column}

    for path in paths:
        if is_collective_counts_csv(path):
            continue

        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if summary_col in fields:
                for row in reader:
                    rows.append(
                        {
                            "ppn": int(row["ppn"]),
                            "nproc": int(row["nproc"]),
                            "algorithm": row["algorithm"].strip(),
                            "message_size_bytes": int(row["message_size_bytes"]),
                            "median_time_sec": float(row[summary_col]),
                        }
                    )
            else:
                missing_columns = timing_columns - fields
                if missing_columns:
                    missing = ", ".join(sorted(missing_columns))
                    raise ValueError(f"{path}: missing required timing column(s): {missing}")

                for row in reader:
                    if row["phase"].strip().lower() != phase.lower():
                        continue
                    if "correct" in row and row["correct"].strip().lower() not in {"1", "true", "yes"}:
                        continue
                    key = (
                        ppn_from_row_or_name(row, path, fallback_ppn),
                        int(row["nproc"]),
                        row["algorithm"].strip(),
                        int(row["message_size_bytes"]),
                    )
                    grouped[key].append(float(row[time_column]))

    for (ppn, nproc, algorithm, message_size), values in sorted(grouped.items()):
        rows.append(
            {
                "ppn": ppn,
                "nproc": nproc,
                "algorithm": algorithm,
                "message_size_bytes": message_size,
                "median_time_sec": median(values),
            }
        )

    if not rows:
        raise ValueError("no usable rows found")
    return rows


def build_xy(rows):
    algorithms = sorted({row["algorithm"] for row in rows})
    feature_names = ["log2_nproc", "log2_message_size_bytes", "log2_ppn"] + [
        f"algorithm={name}" for name in algorithms
    ]
    x_rows, y_log, y_seconds = [], [], []
    for row in rows:
        features = [
            math.log2(row["nproc"]),
            math.log2(row["message_size_bytes"]),
            math.log2(row["ppn"]),
        ]
        features.extend(1.0 if row["algorithm"] == name else 0.0 for name in algorithms)
        x_rows.append(features)
        y_seconds.append(row["median_time_sec"])
        y_log.append(math.log2(row["median_time_sec"]))
    return np.asarray(x_rows), np.asarray(y_log), np.asarray(y_seconds), feature_names, algorithms


def split_data(x, y_log, y_seconds, rows, train_frac, val_frac, seed):
    test_frac = 1.0 - train_frac - val_frac
    if train_frac <= 0 or val_frac <= 0 or test_frac <= 0:
        raise ValueError("train/validation/test fractions must all be positive")

    indices = np.arange(len(rows))
    np.random.default_rng(seed).shuffle(indices)
    n_train = int(round(len(rows) * train_frac))
    n_val = int(round(len(rows) * val_frac))
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def score(model, x, y_log, y_seconds, indices):
    pred_log = model.predict(x[indices])
    pred_seconds = np.power(2.0, pred_log)
    truth_seconds = y_seconds[indices]
    ape = np.abs((pred_seconds - truth_seconds) / truth_seconds) * 100.0
    return {
        "rmse": math.sqrt(mean_squared_error(truth_seconds, pred_seconds)),
        "mae": mean_absolute_error(truth_seconds, pred_seconds),
        "r2": r2_score(truth_seconds, pred_seconds),
        "mape": float(np.mean(ape)),
        "log_rmse": math.sqrt(mean_squared_error(y_log[indices], pred_log)),
    }


def human_seconds(value):
    if abs(value) < 1e-6:
        return f"{value * 1e9:.3g} ns"
    if abs(value) < 1e-3:
        return f"{value * 1e6:.3g} us"
    if abs(value) < 1:
        return f"{value * 1e3:.3g} ms"
    return f"{value:.3g} s"


def write_metrics(path, rows, algorithms, train_idx, val_idx, test_idx, depth_scores, best_depth, train_score, test_score):
    nprocs = sorted({row["nproc"] for row in rows})
    ppns = sorted({row["ppn"] for row in rows})
    message_sizes = sorted({row["message_size_bytes"] for row in rows})
    with path.open("w") as handle:
        handle.write("Bcast median-time regression tree (sklearn)\n")
        handle.write("===========================================\n\n")
        handle.write("Model: sklearn.tree.DecisionTreeRegressor\n")
        handle.write("Features: log2(nproc), one-hot algorithm, log2(message_size_bytes), log2(ppn)\n")
        handle.write("Target: log2(median_time_sec)\n")
        handle.write("Metrics are computed after inverse-transforming predictions to seconds.\n")
        handle.write(f"Rows: {len(rows)}\n")
        handle.write(f"Split sizes: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}\n")
        handle.write(f"nproc values: {nprocs}\n")
        handle.write(f"ppn values: {ppns}\n")
        handle.write(f"message sizes: {min(message_sizes)} .. {max(message_sizes)} bytes\n")
        handle.write(f"algorithms: {', '.join(algorithms)}\n")
        if len(ppns) == 1:
            handle.write("Note: ppn is constant in this dataset, so the model cannot learn a ppn effect.\n")

        handle.write("\nValidation model selection:\n")
        for depth, result in depth_scores:
            handle.write(
                f"  max_depth={depth}: RMSE={human_seconds(result['rmse'])}, "
                f"MAE={human_seconds(result['mae'])}, R2={result['r2']:.4f}, "
                f"MAPE_original={result['mape']:.2f}%, log2_RMSE={result['log_rmse']:.4f}\n"
            )
        handle.write(f"\nSelected max_depth: {best_depth} (lowest validation log2_RMSE)\n")
        handle.write("\nFinal model metrics on train+validation:\n")
        handle.write(format_score(train_score))
        handle.write("\nFinal model metrics on held-out test split:\n")
        handle.write(format_score(test_score))


def format_score(result):
    return (
        f"  RMSE={human_seconds(result['rmse'])}, MAE={human_seconds(result['mae'])}, "
        f"R2={result['r2']:.4f}, MAPE_original={result['mape']:.2f}%, "
        f"log2_RMSE={result['log_rmse']:.4f}\n"
    )


def model_metadata(args, feature_names, algorithms, train_score, test_score, extra=None):
    metadata = {
        "model_file_format": "pickle",
        "feature_names": list(feature_names),
        "algorithms": list(algorithms),
        "feature_order": [
            "log2(nproc)",
            "log2(message_size_bytes)",
            "log2(ppn)",
            "one-hot algorithm columns in algorithms order",
        ],
        "target": "log2(median_time_sec)",
        "target_time_column": str(args.time_column),
        "prediction_inverse": "2 ** prediction",
        "training_args": {name: str(value) for name, value in vars(args).items()},
        "train_validation_score": dict(train_score),
        "test_score": dict(test_score),
    }
    if extra:
        metadata.update(extra)
    return metadata


def data_domain(rows):
    return {
        "feature_points": sorted(
            {
                (
                    int(row["nproc"]),
                    int(row["ppn"]),
                    int(row["message_size_bytes"]),
                )
                for row in rows
            }
        ),
        "nproc_values": sorted({int(row["nproc"]) for row in rows}),
        "ppn_values": sorted({int(row["ppn"]) for row in rows}),
        "message_size_bytes_values": sorted({int(row["message_size_bytes"]) for row in rows}),
    }


def save_fitted_model(path, model, args, feature_names, algorithms, train_score, test_score, extra=None):
    payload = {
        "model": model,
        "metadata": model_metadata(args, feature_names, algorithms, train_score, test_score, extra),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    rows = load_rows(args.input_csvs, args.time_column, args.phase, args.ppn)
    x, y_log, y_seconds, feature_names, algorithms = build_xy(rows)
    train_idx, val_idx, test_idx = split_data(
        x, y_log, y_seconds, rows, args.train_frac, args.val_frac, args.seed
    )
    depths = [int(value) for value in args.max_depth_candidates.split(",") if value.strip()]

    depth_scores = []
    best_depth, best_log_rmse = depths[0], float("inf")
    for depth in depths:
        model = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
            random_state=args.seed,
        )
        model.fit(x[train_idx], y_log[train_idx])
        result = score(model, x, y_log, y_seconds, val_idx)
        depth_scores.append((depth, result))
        if result["log_rmse"] < best_log_rmse:
            best_depth, best_log_rmse = depth, result["log_rmse"]

    train_val_idx = np.concatenate([train_idx, val_idx])
    model = DecisionTreeRegressor(
        max_depth=best_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        random_state=args.seed,
    )
    model.fit(x[train_val_idx], y_log[train_val_idx])
    train_score = score(model, x, y_log, y_seconds, train_val_idx)
    test_score = score(model, x, y_log, y_seconds, test_idx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "bcast_regression_tree_metrics.txt"
    dot_path = args.output_dir / "bcast_regression_tree.dot"
    model_path = args.output_dir / "bcast_regression_tree_model.pkl"
    write_metrics(metrics_path, rows, algorithms, train_idx, val_idx, test_idx, depth_scores, best_depth, train_score, test_score)
    save_fitted_model(
        model_path,
        model,
        args,
        feature_names,
        algorithms,
        train_score,
        test_score,
        extra={"selected_max_depth": best_depth, "data_domain": data_domain(rows)},
    )
    export_graphviz(
        model,
        out_file=str(dot_path),
        feature_names=feature_names,
        filled=True,
        rounded=True,
        impurity=True,
    )

    print(f"Rows: {len(rows)}")
    print(f"Split sizes: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print(f"Selected max_depth={best_depth} from candidates {depths} by validation log2_RMSE")
    print(
        "Test metrics after inverse transform: RMSE={}, MAE={}, R2={:.4f}, "
        "MAPE_original={:.2f}%, log2_RMSE={:.4f}".format(
            human_seconds(test_score["rmse"]),
            human_seconds(test_score["mae"]),
            test_score["r2"],
            test_score["mape"],
            test_score["log_rmse"],
        )
    )
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote fitted model: {model_path}")
    print(f"Wrote tree DOT: {dot_path}")


if __name__ == "__main__":
    main()
