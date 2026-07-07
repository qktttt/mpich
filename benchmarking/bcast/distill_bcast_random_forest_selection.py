#!/usr/bin/env python3
"""Distill a Bcast random-forest timing model into a shallow selection tree.

The saved random forest predicts log2(median_time_sec) for one
(nproc, ppn, message size, algorithm) point.  This script evaluates that model
on a Cartesian grid, labels each grid cell with the fastest predicted
algorithm, and fits a small DecisionTreeClassifier that approximates that
selection surface.
"""

import argparse
import csv
import importlib
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models" / "bcast_random_forest_model.pkl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "models" / "bcast_rf_distilled_selection"
DEFAULT_MAX_MESSAGE_SIZE = 32 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fitted Bcast random forest on ppn/node/message or "
            "ppn/nproc/message grids, label the RF-selected fastest algorithm, "
            "and fit shallow selection trees."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--grid-mode",
        choices=("nodes", "nproc", "both"),
        default="both",
        help=(
            "Which distilled tree/grid to generate. nodes uses (nodes, ppn, "
            "message_size_bytes); nproc uses (nproc, ppn, message_size_bytes)."
        ),
    )
    parser.add_argument("--ppn-min", type=int, default=2)
    parser.add_argument("--ppn-max", type=int, default=32)
    parser.add_argument("--ppn-points", type=int, default=5)
    parser.add_argument("--node-min", type=int, default=2)
    parser.add_argument("--node-max", type=int, default=32)
    parser.add_argument("--node-points", type=int, default=5)
    parser.add_argument("--nproc-min", type=int, default=8)
    parser.add_argument("--nproc-max", type=int, default=1024)
    parser.add_argument("--nproc-points", type=int, default=5)
    parser.add_argument("--message-min", type=int, default=2)
    parser.add_argument("--message-max", type=int, default=DEFAULT_MAX_MESSAGE_SIZE)
    parser.add_argument("--message-points", type=int, default=10)
    parser.add_argument(
        "--ppn-values",
        default=None,
        help="Optional comma-separated ppn grid. Overrides --ppn-min/max/points.",
    )
    parser.add_argument(
        "--node-values",
        default=None,
        help="Optional comma-separated node-count grid. Overrides --node-min/max/points.",
    )
    parser.add_argument(
        "--nproc-values",
        default=None,
        help="Optional comma-separated nproc grid. Overrides --nproc-min/max/points.",
    )
    parser.add_argument(
        "--message-size-values",
        default=None,
        help=(
            "Optional comma-separated message-size grid in bytes. Overrides "
            "--message-min/max/points."
        ),
    )
    parser.add_argument(
        "--candidate-algorithms",
        default=None,
        help=(
            "Optional comma-separated algorithm subset. Defaults to all algorithms "
            "stored in the random-forest model metadata."
        ),
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def install_numpy_pickle_shims():
    """Map NumPy 2 pickle module paths to NumPy 1.x equivalents when needed."""
    aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core.umath": "numpy.core.umath",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    for new_name, old_name in aliases.items():
        if new_name not in sys.modules:
            sys.modules[new_name] = importlib.import_module(old_name)


def load_saved_model(path):
    install_numpy_pickle_shims()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata", {})
    return payload, {}


def algorithms_from_metadata(metadata):
    algorithms = metadata.get("algorithms")
    if algorithms:
        return [str(name) for name in algorithms]

    feature_names = metadata.get("feature_names") or []
    algorithms = []
    for name in feature_names:
        if str(name).startswith("algorithm="):
            algorithms.append(str(name).split("=", 1)[1])
    if algorithms:
        return algorithms

    raise ValueError("saved model metadata does not contain algorithm names")


def parse_int_list(value, name):
    if value is None:
        return None
    values = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise ValueError(f"{name} values must be positive")
        values.append(parsed)
    if not values:
        raise ValueError(f"{name} list is empty")
    return sorted(set(values))


def parse_algorithm_list(value, model_algorithms):
    if value is None:
        return list(model_algorithms)
    requested = [part.strip() for part in value.split(",") if part.strip()]
    if not requested:
        raise ValueError("candidate algorithm list is empty")
    missing = [name for name in requested if name not in model_algorithms]
    if missing:
        raise ValueError(
            "candidate algorithm(s) are not present in the saved model metadata: "
            + ", ".join(missing)
        )
    return requested


def log2_spaced_power_grid(min_value, max_value, points, name):
    if min_value <= 0 or max_value <= 0:
        raise ValueError(f"{name} bounds must be positive")
    if min_value > max_value:
        raise ValueError(f"{name} min cannot be greater than max")
    if points < 1:
        raise ValueError(f"{name} points must be positive")
    if points == 1:
        return [min_value]

    exponents = np.linspace(math.log2(min_value), math.log2(max_value), points)
    values = [int(2 ** round(exponent)) for exponent in exponents]
    values[0] = min_value
    values[-1] = max_value
    return sorted(set(values))


def resolve_grid(args):
    ppn_values = parse_int_list(args.ppn_values, "ppn")
    if ppn_values is None:
        ppn_values = log2_spaced_power_grid(args.ppn_min, args.ppn_max, args.ppn_points, "ppn")

    node_values = parse_int_list(args.node_values, "node")
    if node_values is None:
        node_values = log2_spaced_power_grid(
            args.node_min, args.node_max, args.node_points, "node"
        )

    nproc_values = parse_int_list(args.nproc_values, "nproc")
    if nproc_values is None:
        nproc_values = log2_spaced_power_grid(
            args.nproc_min, args.nproc_max, args.nproc_points, "nproc"
        )

    message_sizes = parse_int_list(args.message_size_values, "message size")
    if message_sizes is None:
        message_sizes = log2_spaced_power_grid(
            args.message_min, args.message_max, args.message_points, "message size"
        )

    return ppn_values, node_values, nproc_values, message_sizes


def build_model_feature_rows(ppn, nproc, message_size, algorithms, model_algorithms):
    algorithm_to_index = {name: index for index, name in enumerate(model_algorithms)}
    rows = []
    for algorithm in algorithms:
        one_hot = [0.0] * len(model_algorithms)
        one_hot[algorithm_to_index[algorithm]] = 1.0
        rows.append(
            [
                math.log2(nproc),
                math.log2(message_size),
                math.log2(ppn),
                *one_hot,
            ]
        )
    return np.asarray(rows, dtype=float)


def predict_grid(
    model,
    grid_mode,
    ppn_values,
    primary_values,
    message_sizes,
    algorithms,
    model_algorithms,
):
    cell_rows = []
    timing_rows = []

    for ppn in ppn_values:
        for primary_value in primary_values:
            if grid_mode == "nodes":
                nodes = primary_value
                nproc = ppn * nodes
            elif grid_mode == "nproc":
                nodes = None
                nproc = primary_value
            else:
                raise ValueError(f"unknown grid mode: {grid_mode}")

            for message_size in message_sizes:
                features = build_model_feature_rows(
                    ppn, nproc, message_size, algorithms, model_algorithms
                )
                predicted_log_times = model.predict(features)
                predicted_times = np.power(2.0, predicted_log_times)
                best_index = int(np.argmin(predicted_times))
                best_algorithm = algorithms[best_index]
                best_time = float(predicted_times[best_index])

                cell = {
                    "grid_mode": grid_mode,
                    "ppn": ppn,
                    "nodes": nodes,
                    "nproc": nproc,
                    "message_size_bytes": message_size,
                    "optimal_algorithm": best_algorithm,
                    "optimal_predicted_time_sec": best_time,
                }
                for algorithm, predicted_time in zip(algorithms, predicted_times):
                    cell[f"predicted_time_sec__{algorithm}"] = float(predicted_time)
                    timing_rows.append(
                        {
                            "grid_mode": grid_mode,
                            "ppn": ppn,
                            "nodes": nodes,
                            "nproc": nproc,
                            "message_size_bytes": message_size,
                            "algorithm": algorithm,
                            "predicted_time_sec": float(predicted_time),
                            "is_optimal": int(algorithm == best_algorithm),
                        }
                    )
                cell_rows.append(cell)

    return cell_rows, timing_rows


def selection_features(cell_rows, grid_mode):
    if grid_mode == "nodes":
        primary_name = "nodes"
    elif grid_mode == "nproc":
        primary_name = "nproc"
    else:
        raise ValueError(f"unknown grid mode: {grid_mode}")

    feature_names = [primary_name, "ppn", "message_size_bytes"]
    x_rows = []
    y_labels = []
    for row in cell_rows:
        x_rows.append(
            [
                float(row[primary_name]),
                float(row["ppn"]),
                float(row["message_size_bytes"]),
            ]
        )
        y_labels.append(row["optimal_algorithm"])
    return np.asarray(x_rows, dtype=float), np.asarray(y_labels), feature_names


def write_cell_dataset(path, rows, algorithms):
    fieldnames = [
        "grid_mode",
        "ppn",
        "nodes",
        "nproc",
        "message_size_bytes",
        "optimal_algorithm",
        "optimal_predicted_time_sec",
    ] + [f"predicted_time_sec__{algorithm}" for algorithm in algorithms]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_timing_dataset(path, rows):
    fieldnames = [
        "grid_mode",
        "ppn",
        "nodes",
        "nproc",
        "message_size_bytes",
        "algorithm",
        "predicted_time_sec",
        "is_optimal",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(
    path,
    args,
    grid_mode,
    ppn_values,
    primary_values,
    message_sizes,
    algorithms,
    cell_rows,
    model,
):
    counts = Counter(row["optimal_algorithm"] for row in cell_rows)
    x_rows, y_labels, feature_names = selection_features(cell_rows, grid_mode)
    predictions = model.predict(x_rows)
    accuracy = accuracy_score(y_labels, predictions)
    primary_label = "node-count" if grid_mode == "nodes" else "nproc"

    with path.open("w") as handle:
        handle.write("Bcast random-forest distilled selection tree\n")
        handle.write("===========================================\n\n")
        handle.write(f"Source random forest: {args.model}\n")
        handle.write(f"Grid mode: {grid_mode}\n")
        handle.write(f"Selection tree max_depth: {args.max_depth}\n")
        handle.write(f"Selection tree min_samples_leaf: {args.min_samples_leaf}\n")
        handle.write(f"Selection tree features: {', '.join(feature_names)}\n")
        handle.write(f"Grid cells: {len(cell_rows)}\n")
        handle.write(f"ppn values: {ppn_values}\n")
        handle.write(f"{primary_label} values: {primary_values}\n")
        handle.write(f"message-size values: {message_sizes}\n")
        handle.write(f"candidate algorithms: {', '.join(algorithms)}\n")
        handle.write("\nOptimal algorithm counts from random forest:\n")
        for algorithm in algorithms:
            handle.write(f"  {algorithm}: {counts.get(algorithm, 0)}\n")
        handle.write(f"\nTraining accuracy against RF labels: {accuracy:.6f}\n")
        handle.write("\nClassification report:\n")
        handle.write(classification_report(y_labels, predictions, zero_division=0))
        handle.write("\nConfusion matrix rows/columns use this algorithm order:\n")
        handle.write("  " + ", ".join(model.classes_) + "\n")
        matrix = confusion_matrix(y_labels, predictions, labels=model.classes_)
        for row in matrix:
            handle.write("  " + ", ".join(str(int(value)) for value in row) + "\n")
        handle.write("\nSelection tree rules:\n")
        handle.write(export_text(model, feature_names=feature_names))


def save_selection_model(
    path,
    model,
    args,
    grid_mode,
    feature_names,
    algorithms,
    ppn_values,
    primary_values,
    message_sizes,
):
    primary_key = "node_values" if grid_mode == "nodes" else "nproc_values"
    payload = {
        "model": model,
        "metadata": {
            "model_file_format": "pickle",
            "source_random_forest_model": str(args.model),
            "model_type": "sklearn.tree.DecisionTreeClassifier",
            "max_depth": args.max_depth,
            "grid_mode": grid_mode,
            "feature_names": list(feature_names),
            "feature_order": list(feature_names),
            "feature_scale": "original",
            "target": "optimal_algorithm",
            "algorithms": list(algorithms),
            "ppn_values": list(ppn_values),
            primary_key: list(primary_values),
            "message_size_bytes_values": list(message_sizes),
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def fit_and_write_grid(
    args,
    rf_model,
    grid_mode,
    ppn_values,
    primary_values,
    message_sizes,
    algorithms,
    model_algorithms,
    output_dir,
):
    cell_rows, timing_rows = predict_grid(
        rf_model,
        grid_mode,
        ppn_values,
        primary_values,
        message_sizes,
        algorithms,
        model_algorithms,
    )
    x_rows, y_labels, feature_names = selection_features(cell_rows, grid_mode)
    selection_tree = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    selection_tree.fit(x_rows, y_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_dataset_path = output_dir / "bcast_rf_grid_optimal_algorithms.csv"
    timing_dataset_path = output_dir / "bcast_rf_grid_algorithm_times.csv"
    model_path = output_dir / "bcast_rf_distilled_selection_tree.pkl"
    dot_path = output_dir / "bcast_rf_distilled_selection_tree.dot"
    text_path = output_dir / "bcast_rf_distilled_selection_tree.txt"
    metrics_path = output_dir / "bcast_rf_distilled_selection_metrics.txt"

    write_cell_dataset(cell_dataset_path, cell_rows, algorithms)
    write_timing_dataset(timing_dataset_path, timing_rows)
    save_selection_model(
        model_path,
        selection_tree,
        args,
        grid_mode,
        feature_names,
        algorithms,
        ppn_values,
        primary_values,
        message_sizes,
    )
    export_graphviz(
        selection_tree,
        out_file=str(dot_path),
        feature_names=feature_names,
        class_names=list(selection_tree.classes_),
        filled=True,
        rounded=True,
        impurity=True,
    )
    text_path.write_text(export_text(selection_tree, feature_names=feature_names))
    write_metrics(
        metrics_path,
        args,
        grid_mode,
        ppn_values,
        primary_values,
        message_sizes,
        algorithms,
        cell_rows,
        selection_tree,
    )

    primary_name = "node-count" if grid_mode == "nodes" else "nproc"
    print(
        f"Grid ({grid_mode}): {len(ppn_values)} ppn x {len(primary_values)} "
        f"{primary_name} x {len(message_sizes)} message-size = {len(cell_rows)} cells"
    )
    print(f"Wrote optimal-label grid: {cell_dataset_path}")
    print(f"Wrote per-algorithm timing grid: {timing_dataset_path}")
    print(f"Wrote distilled selection tree model: {model_path}")
    print(f"Wrote distilled selection tree DOT: {dot_path}")
    print(f"Wrote distilled selection tree text: {text_path}")
    print(f"Wrote metrics: {metrics_path}")


def main():
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Random-forest model not found: {args.model}")

    rf_model, metadata = load_saved_model(args.model)
    model_algorithms = algorithms_from_metadata(metadata)
    algorithms = parse_algorithm_list(args.candidate_algorithms, model_algorithms)
    ppn_values, node_values, nproc_values, message_sizes = resolve_grid(args)

    print(f"Loaded random-forest model: {args.model}")
    print(f"Algorithms compared per cell: {', '.join(algorithms)}")

    modes = ["nodes", "nproc"] if args.grid_mode == "both" else [args.grid_mode]
    for mode in modes:
        primary_values = node_values if mode == "nodes" else nproc_values
        output_dir = args.output_dir / f"by_{mode}" if args.grid_mode == "both" else args.output_dir
        fit_and_write_grid(
            args,
            rf_model,
            mode,
            ppn_values,
            primary_values,
            message_sizes,
            algorithms,
            model_algorithms,
            output_dir,
        )


if __name__ == "__main__":
    main()
