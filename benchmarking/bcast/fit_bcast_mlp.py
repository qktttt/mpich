#!/usr/bin/env python3
"""Fit an sklearn MLP regressor to median MPI_Bcast timings."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fit_bcast_regression_tree import (
    build_xy,
    data_domain,
    default_input_csvs,
    format_score,
    human_seconds,
    load_rows,
    save_fitted_model,
    score,
    split_data,
)


def parse_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Fit an sklearn MLP regressor to Bcast timings.")
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
    parser.add_argument(
        "--hidden-layer-candidates",
        default="64;128;128,64;256,128",
        help=(
            "Semicolon-separated hidden layer candidates. Use commas within one "
            "candidate, e.g. '64;128,64;256,128'."
        ),
    )
    parser.add_argument(
        "--alpha-candidates",
        default="0.0001,0.001,0.01",
        help="Comma-separated L2 regularization candidates.",
    )
    parser.add_argument(
        "--activation",
        choices=("identity", "logistic", "tanh", "relu"),
        default="relu",
    )
    parser.add_argument("--solver", choices=("adam", "lbfgs", "sgd"), default="adam")
    parser.add_argument("--learning-rate-init", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--n-iter-no-change", type=int, default=50)
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable sklearn's internal early-stopping split inside each training fit.",
    )
    args = parser.parse_args()
    if not args.input_csvs:
        args.input_csvs = default_input_csvs(script_dir, args.time_column)
    return args


def parse_hidden_layer_candidates(value):
    candidates = []
    for candidate in value.split(";"):
        candidate = candidate.strip()
        if not candidate:
            continue
        layers = tuple(int(part.strip()) for part in candidate.split(",") if part.strip())
        if not layers or any(layer <= 0 for layer in layers):
            raise ValueError(f"invalid hidden layer candidate: {candidate!r}")
        candidates.append(layers)
    if not candidates:
        raise ValueError("no hidden layer candidates provided")
    return candidates


def parse_float_candidates(value, name):
    candidates = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = float(part)
        if parsed < 0.0:
            raise ValueError(f"{name} candidates must be non-negative")
        candidates.append(parsed)
    if not candidates:
        raise ValueError(f"no {name} candidates provided")
    return candidates


def parse_batch_size(value):
    value = value.strip().lower()
    if value == "auto":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--batch-size must be a positive integer or 'auto'")
    return parsed


def make_mlp_model(hidden_layer_sizes, alpha, args):
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=args.activation,
        solver=args.solver,
        alpha=alpha,
        batch_size=parse_batch_size(args.batch_size),
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        random_state=args.seed,
        tol=args.tol,
        n_iter_no_change=args.n_iter_no_change,
        early_stopping=args.early_stopping,
    )
    return make_pipeline(StandardScaler(), mlp)


def write_metrics(
    path,
    rows,
    algorithms,
    train_idx,
    val_idx,
    test_idx,
    candidate_scores,
    best_hidden_layers,
    best_alpha,
    train_score,
    test_score,
    args,
):
    nprocs = sorted({row["nproc"] for row in rows})
    ppns = sorted({row["ppn"] for row in rows})
    message_sizes = sorted({row["message_size_bytes"] for row in rows})
    with path.open("w") as handle:
        handle.write("Bcast median-time MLP regressor (sklearn)\n")
        handle.write("========================================\n\n")
        handle.write("Model: StandardScaler + sklearn.neural_network.MLPRegressor\n")
        handle.write("Features: log2(nproc), one-hot algorithm, log2(message_size_bytes), log2(ppn)\n")
        handle.write("Target: log2(median_time_sec)\n")
        handle.write(f"Target source timing column: {args.time_column}\n")
        handle.write("Metrics are computed after inverse-transforming predictions to seconds.\n")
        handle.write(f"Rows: {len(rows)}\n")
        handle.write(f"Split sizes: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}\n")
        handle.write(f"activation: {args.activation}\n")
        handle.write(f"solver: {args.solver}\n")
        handle.write(f"learning_rate_init: {args.learning_rate_init}\n")
        handle.write(f"max_iter: {args.max_iter}\n")
        handle.write(f"batch_size: {args.batch_size}\n")
        handle.write(f"tol: {args.tol}\n")
        handle.write(f"n_iter_no_change: {args.n_iter_no_change}\n")
        handle.write(f"early_stopping: {args.early_stopping}\n")
        handle.write(f"nproc values: {nprocs}\n")
        handle.write(f"ppn values: {ppns}\n")
        handle.write(f"message sizes: {min(message_sizes)} .. {max(message_sizes)} bytes\n")
        handle.write(f"algorithms: {', '.join(algorithms)}\n")
        if len(ppns) == 1:
            handle.write("Note: ppn is constant in this dataset, so the model cannot learn a ppn effect.\n")

        handle.write("\nValidation model selection:\n")
        for hidden_layers, alpha, result in candidate_scores:
            handle.write(
                f"  hidden_layers={hidden_layers}, alpha={alpha}: "
                f"RMSE={human_seconds(result['rmse'])}, "
                f"MAE={human_seconds(result['mae'])}, R2={result['r2']:.4f}, "
                f"MAPE_original={result['mape']:.2f}%, log2_RMSE={result['log_rmse']:.4f}\n"
            )
        handle.write(
            f"\nSelected hidden_layers: {best_hidden_layers}, alpha={best_alpha} "
            "(lowest validation log2_RMSE)\n"
        )
        handle.write("\nFinal model metrics on train+validation:\n")
        handle.write(format_score(train_score))
        handle.write("\nFinal model metrics on held-out test split:\n")
        handle.write(format_score(test_score))


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
    hidden_layer_candidates = parse_hidden_layer_candidates(args.hidden_layer_candidates)
    alpha_candidates = parse_float_candidates(args.alpha_candidates, "alpha")

    candidate_scores = []
    best_hidden_layers = hidden_layer_candidates[0]
    best_alpha = alpha_candidates[0]
    best_log_rmse = float("inf")
    for hidden_layers in hidden_layer_candidates:
        for alpha in alpha_candidates:
            model = make_mlp_model(hidden_layers, alpha, args)
            model.fit(x[train_idx], y_log[train_idx])
            result = score(model, x, y_log, y_seconds, val_idx)
            candidate_scores.append((hidden_layers, alpha, result))
            if result["log_rmse"] < best_log_rmse:
                best_hidden_layers = hidden_layers
                best_alpha = alpha
                best_log_rmse = result["log_rmse"]

    train_val_idx = np.concatenate([train_idx, val_idx])
    model = make_mlp_model(best_hidden_layers, best_alpha, args)
    model.fit(x[train_val_idx], y_log[train_val_idx])
    train_score = score(model, x, y_log, y_seconds, train_val_idx)
    test_score = score(model, x, y_log, y_seconds, test_idx)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "bcast_mlp_metrics.txt"
    model_path = args.output_dir / "bcast_mlp_model.pkl"
    predictions_path = args.output_dir / "bcast_mlp_predictions.csv"
    write_metrics(
        metrics_path,
        rows,
        algorithms,
        train_idx,
        val_idx,
        test_idx,
        candidate_scores,
        best_hidden_layers,
        best_alpha,
        train_score,
        test_score,
        args,
    )
    save_fitted_model(
        model_path,
        model,
        args,
        feature_names,
        algorithms,
        train_score,
        test_score,
        extra={
            "selected_hidden_layer_sizes": best_hidden_layers,
            "selected_alpha": best_alpha,
            "target_time_column": args.time_column,
            "data_domain": data_domain(rows),
        },
    )
    write_predictions(predictions_path, rows, x, y_log, y_seconds, train_val_idx, test_idx, model)

    print(f"Rows: {len(rows)}")
    print(f"Split sizes: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print(
        "Selected hidden_layers={}, alpha={} by validation log2_RMSE".format(
            best_hidden_layers, best_alpha
        )
    )
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
    print(f"Wrote predictions: {predictions_path}")


if __name__ == "__main__":
    main()
