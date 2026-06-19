#!/usr/bin/env python3
"""Fit an sklearn random forest to median MPI_Bcast timings."""

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from fit_bcast_regression_tree import (
    build_xy,
    format_score,
    human_seconds,
    data_domain,
    load_rows,
    save_fitted_model,
    score,
    split_data,
)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_summary = script_dir / "plots" / "bcast_median_summary.csv"
    default_inputs = [default_summary] if default_summary.exists() else sorted(script_dir.glob("bcast_bench_ppn*r_*.csv"))

    parser = argparse.ArgumentParser(description="Fit an sklearn random forest to Bcast median timings.")
    parser.add_argument("input_csvs", nargs="*", type=Path, default=default_inputs)
    parser.add_argument("-o", "--output-dir", type=Path, default=script_dir / "models")
    parser.add_argument("--time-column", default="avg_latency_sec")
    parser.add_argument("--phase", default="actual")
    parser.add_argument("--ppn", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-depth-candidates", default="4,6,8,10,12,14,16,18,20")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-features", default="1.0")
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--min-samples-split", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def parse_max_features(value):
    value = value.strip().lower()
    if value in {"sqrt", "log2", None}:
        return value
    if value in {"all", "none"}:
        return 1.0
    number = float(value)
    return int(number) if number > 1 else number


def write_metrics(path, rows, algorithms, train_idx, val_idx, test_idx, depth_scores, best_depth, train_score, test_score, args, max_features):
    nprocs = sorted({row["nproc"] for row in rows})
    ppns = sorted({row["ppn"] for row in rows})
    message_sizes = sorted({row["message_size_bytes"] for row in rows})
    with path.open("w") as handle:
        handle.write("Bcast median-time random forest (sklearn)\n")
        handle.write("=========================================\n\n")
        handle.write("Model: sklearn.ensemble.RandomForestRegressor\n")
        handle.write("Features: log2(nproc), one-hot algorithm, log2(message_size_bytes), log2(ppn)\n")
        handle.write("Target: log2(median_time_sec)\n")
        handle.write("Metrics are computed after inverse-transforming predictions to seconds.\n")
        handle.write(f"Rows: {len(rows)}\n")
        handle.write(f"Split sizes: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}\n")
        handle.write(f"n_estimators: {args.n_estimators}\n")
        handle.write(f"max_features: {args.max_features} -> {max_features}\n")
        handle.write(f"min_samples_leaf: {args.min_samples_leaf}\n")
        handle.write(f"min_samples_split: {args.min_samples_split}\n")
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


def write_feature_importances(path, feature_names, model):
    rows = sorted(zip(feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "importance"])
        writer.writerows(rows)


def write_predictions(path, rows, x, y_log, y_seconds, train_val_idx, test_idx, model):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "row_index",
                "ppn",
                "nproc",
                "algorithm",
                "message_size_bytes",
                "actual_median_time_sec",
                "predicted_median_time_sec",
                "actual_log2_time",
                "predicted_log2_time",
                "absolute_percentage_error_original",
            ]
        )
        for split_name, indices in [("train_validation", train_val_idx), ("test", test_idx)]:
            pred_log = model.predict(x[indices])
            pred_seconds = np.power(2.0, pred_log)
            for row_index, log_pred, pred in zip(indices, pred_log, pred_seconds):
                actual = y_seconds[row_index]
                ape = abs((pred - actual) / actual) * 100.0
                row = rows[row_index]
                writer.writerow(
                    [
                        split_name,
                        row_index,
                        row["ppn"],
                        row["nproc"],
                        row["algorithm"],
                        row["message_size_bytes"],
                        actual,
                        pred,
                        y_log[row_index],
                        log_pred,
                        ape,
                    ]
                )


def main():
    args = parse_args()
    rows = load_rows(args.input_csvs, args.time_column, args.phase, args.ppn)
    x, y_log, y_seconds, feature_names, algorithms = build_xy(rows)
    train_idx, val_idx, test_idx = split_data(
        x, y_log, y_seconds, rows, args.train_frac, args.val_frac, args.seed
    )
    depths = [int(value) for value in args.max_depth_candidates.split(",") if value.strip()]
    max_features = parse_max_features(args.max_features)

    depth_scores = []
    best_depth, best_log_rmse = depths[0], float("inf")
    for depth in depths:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=depth,
            min_samples_leaf=args.min_samples_leaf,
            min_samples_split=args.min_samples_split,
            max_features=max_features,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        model.fit(x[train_idx], y_log[train_idx])
        result = score(model, x, y_log, y_seconds, val_idx)
        depth_scores.append((depth, result))
        if result["log_rmse"] < best_log_rmse:
            best_depth, best_log_rmse = depth, result["log_rmse"]

    train_val_idx = np.concatenate([train_idx, val_idx])
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=best_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_features=max_features,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    model.fit(x[train_val_idx], y_log[train_val_idx])
    train_score = score(model, x, y_log, y_seconds, train_val_idx)
    test_score = score(model, x, y_log, y_seconds, test_idx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "bcast_random_forest_metrics.txt"
    model_path = args.output_dir / "bcast_random_forest_model.pkl"
    importances_path = args.output_dir / "bcast_random_forest_feature_importances.csv"
    predictions_path = args.output_dir / "bcast_random_forest_predictions.csv"
    write_metrics(metrics_path, rows, algorithms, train_idx, val_idx, test_idx, depth_scores, best_depth, train_score, test_score, args, max_features)
    save_fitted_model(
        model_path,
        model,
        args,
        feature_names,
        algorithms,
        train_score,
        test_score,
        extra={
            "selected_max_depth": best_depth,
            "max_features": max_features,
            "data_domain": data_domain(rows),
        },
    )
    write_feature_importances(importances_path, feature_names, model)
    write_predictions(predictions_path, rows, x, y_log, y_seconds, train_val_idx, test_idx, model)

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
    print(f"Wrote feature importances: {importances_path}")
    print(f"Wrote predictions: {predictions_path}")


if __name__ == "__main__":
    main()
