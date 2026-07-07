#!/usr/bin/env python3
"""Generate manually partitioned Bcast decision trees from a timing model.

Workflow:
1. Evaluate the saved timing model on a Cartesian
   (nproc, ppn, message size, algorithm) grid.
2. Label every grid cell with the fastest predicted algorithm.
3. Split the grid by manual supervision:
      single_node:    nproc <= ppn
      regular_ppn:    nproc > ppn and nproc % ppn == 0
      nonregular_ppn: nproc > ppn and nproc % ppn != 0
4. Fit one shallow DecisionTreeClassifier per partition.
5. Export CSV datasets, sklearn artifacts, tree text/DOT files, and an
   intermediate JSON rule file that preserves the manual partitioning.

The JSON output is intentionally an intermediate rule representation.  The
current MPICH coll_selection.json format in this checkout does not directly
encode arbitrary ppn or nproc % ppn predicates.
"""

import argparse
import csv
import importlib
import json
import math
import os
import pickle
import sys
from collections import Counter, OrderedDict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models" / "bcast_random_forest_model.pkl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "models" / "bcast_partitioned_selection"
DEFAULT_SOURCE_MODEL_LABEL = "random_forest"
DEFAULT_RETRAIN_COMMAND = (
    "python3 benchmarking/bcast/fit_bcast_random_forest.py --time-column {required}"
)
DEFAULT_MAX_MESSAGE_SIZE = 32 * 1024 * 1024
PARTITION_ORDER = ("single_node", "regular_ppn", "nonregular_ppn")
accuracy_score = None
classification_report = None
confusion_matrix = None
DecisionTreeClassifier = None
export_graphviz = None
export_text = None
_tree = None


def require_sklearn():
    global accuracy_score
    global classification_report
    global confusion_matrix
    global DecisionTreeClassifier
    global export_graphviz
    global export_text
    global _tree

    if DecisionTreeClassifier is not None:
        return

    try:
        from sklearn.metrics import accuracy_score as imported_accuracy_score
        from sklearn.metrics import classification_report as imported_classification_report
        from sklearn.metrics import confusion_matrix as imported_confusion_matrix
        from sklearn.tree import DecisionTreeClassifier as imported_decision_tree_classifier
        from sklearn.tree import export_graphviz as imported_export_graphviz
        from sklearn.tree import export_text as imported_export_text
        from sklearn.tree import _tree as imported_tree
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required to generate partitioned Bcast decision trees. "
            "Run this script in the same Python environment used to train the source model."
        ) from exc

    accuracy_score = imported_accuracy_score
    classification_report = imported_classification_report
    confusion_matrix = imported_confusion_matrix
    DecisionTreeClassifier = imported_decision_tree_classifier
    export_graphviz = imported_export_graphviz
    export_text = imported_export_text
    _tree = imported_tree


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate manual-partitioned Bcast decision trees from a fitted "
            "timing model."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-model-label", default=DEFAULT_SOURCE_MODEL_LABEL)
    parser.add_argument("--retrain-command", default=DEFAULT_RETRAIN_COMMAND)
    parser.add_argument(
        "--require-model-time-column",
        default="max_time_sec",
        help=(
            "Require the saved source model metadata to report this training time column. "
            "Use an empty string to disable the check."
        ),
    )
    parser.add_argument("--ppn-min", type=int, default=2)
    parser.add_argument("--ppn-max", type=int, default=32)
    parser.add_argument("--ppn-points", type=int, default=5)
    parser.add_argument("--nproc-min", type=int, default=8)
    parser.add_argument("--nproc-max", type=int, default=1024)
    parser.add_argument("--nproc-points", type=int, default=8)
    parser.add_argument("--message-min", type=int, default=2)
    parser.add_argument("--message-max", type=int, default=DEFAULT_MAX_MESSAGE_SIZE)
    parser.add_argument("--message-points", type=int, default=10)
    parser.add_argument(
        "--ppn-values",
        default=None,
        help="Optional comma-separated ppn grid. Overrides --ppn-min/max/points.",
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
            "stored in the source model metadata."
        ),
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--tree-features",
        default="nproc,ppn,message_size_bytes",
        help=(
            "Comma-separated feature columns for each partition tree. Available: "
            "nproc, ppn, message_size_bytes, estimated_nodes, last_node_ranks."
        ),
    )
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


def model_time_column(metadata):
    value = metadata.get("target_time_column")
    if value:
        return str(value)
    training_args = metadata.get("training_args") or {}
    value = training_args.get("time_column")
    if value:
        return str(value)
    return None


def require_model_time_column(metadata, required, source_model_label, retrain_command):
    if not required:
        return
    actual = model_time_column(metadata)
    if actual != required:
        actual_label = actual if actual is not None else "unknown"
        command = retrain_command.format(required=required)
        raise SystemExit(
            f"Saved {source_model_label} model was not trained with the required timing column. "
            f"required={required}, actual={actual_label}. "
            f"Retrain with: {command}"
        )


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


def parse_feature_list(value):
    allowed = {
        "nproc",
        "ppn",
        "message_size_bytes",
        "estimated_nodes",
        "last_node_ranks",
    }
    features = [part.strip() for part in value.split(",") if part.strip()]
    if not features:
        raise ValueError("tree feature list is empty")
    unknown = [feature for feature in features if feature not in allowed]
    if unknown:
        raise ValueError("unknown tree feature(s): " + ", ".join(unknown))
    return features


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

    return ppn_values, nproc_values, message_sizes


def partition_name(nproc, ppn):
    if nproc <= ppn:
        return "single_node"
    if nproc % ppn == 0:
        return "regular_ppn"
    return "nonregular_ppn"


def estimated_nodes(nproc, ppn):
    return int(math.ceil(float(nproc) / float(ppn)))


def last_node_ranks(nproc, ppn):
    remainder = nproc % ppn
    return ppn if remainder == 0 else remainder


def build_rf_feature_rows(ppn, nproc, message_size, algorithms, model_algorithms):
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


def predict_grid(model, ppn_values, nproc_values, message_sizes, algorithms, model_algorithms):
    cell_rows = []
    timing_rows = []

    for ppn in ppn_values:
        for nproc in nproc_values:
            partition = partition_name(nproc, ppn)
            nodes = estimated_nodes(nproc, ppn)
            tail_ranks = last_node_ranks(nproc, ppn)
            for message_size in message_sizes:
                features = build_rf_feature_rows(
                    ppn, nproc, message_size, algorithms, model_algorithms
                )
                predicted_log_times = model.predict(features)
                predicted_times = np.power(2.0, predicted_log_times)
                best_index = int(np.argmin(predicted_times))
                best_algorithm = algorithms[best_index]
                best_time = float(predicted_times[best_index])

                cell = {
                    "partition": partition,
                    "ppn": ppn,
                    "nproc": nproc,
                    "estimated_nodes": nodes,
                    "last_node_ranks": tail_ranks,
                    "message_size_bytes": message_size,
                    "optimal_algorithm": best_algorithm,
                    "optimal_predicted_time_sec": best_time,
                }
                for algorithm, predicted_time in zip(algorithms, predicted_times):
                    cell[f"predicted_time_sec__{algorithm}"] = float(predicted_time)
                    timing_rows.append(
                        {
                            "partition": partition,
                            "ppn": ppn,
                            "nproc": nproc,
                            "estimated_nodes": nodes,
                            "last_node_ranks": tail_ranks,
                            "message_size_bytes": message_size,
                            "algorithm": algorithm,
                            "predicted_time_sec": float(predicted_time),
                            "is_optimal": int(algorithm == best_algorithm),
                        }
                    )
                cell_rows.append(cell)

    return cell_rows, timing_rows


def rows_for_partition(cell_rows, partition):
    return [row for row in cell_rows if row["partition"] == partition]


def selection_features(rows, feature_names):
    x_rows = []
    y_labels = []
    for row in rows:
        x_rows.append([float(row[name]) for name in feature_names])
        y_labels.append(row["optimal_algorithm"])
    return np.asarray(x_rows, dtype=float), np.asarray(y_labels)


def algorithm_leaf_container(algorithm):
    if algorithm == "release_gather":
        return OrderedDict(
            [
                ("requires", "MPIDI_CH4_release_gather"),
                ("algorithm=MPIDI_POSIX_mpi_bcast_release_gather", OrderedDict()),
            ]
        )
    if algorithm == "nb":
        return OrderedDict([("algorithm=MPIR_Coll_nb", OrderedDict())])
    if algorithm == "tree":
        return OrderedDict(
            [
                (
                    "algorithm=MPIR_Bcast_intra_tree",
                    OrderedDict(
                        [
                            ("tree_type=kary", OrderedDict()),
                            ("k=2", OrderedDict()),
                            ("is_non_blocking=0", OrderedDict()),
                        ]
                    ),
                )
            ]
        )
    if algorithm == "pipelined_tree":
        return OrderedDict(
            [
                (
                    "algorithm=MPIR_Bcast_intra_pipelined_tree",
                    OrderedDict(
                        [
                            ("tree_type=kary", OrderedDict()),
                            ("k=2", OrderedDict()),
                            ("is_non_blocking=0", OrderedDict()),
                            ("chunk_size=0", OrderedDict()),
                            ("recv_pre_posted=0", OrderedDict()),
                        ]
                    ),
                )
            ]
        )
    return OrderedDict([(f"algorithm=MPIR_Bcast_intra_{algorithm}", OrderedDict())])


def tree_to_rule_json(model, feature_names):
    tree = model.tree_
    classes = list(model.classes_)

    def convert_node(node_id):
        samples = int(tree.n_node_samples[node_id])
        values = tree.value[node_id][0]
        counts = OrderedDict((classes[index], int(count)) for index, count in enumerate(values))
        predicted_class = classes[int(np.argmax(values))]

        if tree.children_left[node_id] == _tree.TREE_LEAF:
            return OrderedDict(
                [
                    ("type", "leaf"),
                    ("algorithm", predicted_class),
                    ("samples", samples),
                    ("class_counts", counts),
                    ("csel_container", algorithm_leaf_container(predicted_class)),
                ]
            )

        feature_name = feature_names[tree.feature[node_id]]
        threshold = float(tree.threshold[node_id])
        return OrderedDict(
            [
                ("type", "decision"),
                ("feature", feature_name),
                ("operator", "<="),
                ("threshold", threshold),
                ("condition", f"{feature_name} <= {threshold:.17g}"),
                ("samples", samples),
                ("class_counts", counts),
                ("yes", convert_node(tree.children_left[node_id])),
                ("no", convert_node(tree.children_right[node_id])),
            ]
        )

    return convert_node(0)


def write_cell_dataset(path, rows, algorithms):
    fieldnames = [
        "partition",
        "ppn",
        "nproc",
        "estimated_nodes",
        "last_node_ranks",
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
        "partition",
        "ppn",
        "nproc",
        "estimated_nodes",
        "last_node_ranks",
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


def write_partition_metrics(path, partition, rows, feature_names, model, source_model_label):
    counts = Counter(row["optimal_algorithm"] for row in rows)
    x_rows, y_labels = selection_features(rows, feature_names)
    predictions = model.predict(x_rows)
    accuracy = accuracy_score(y_labels, predictions)

    with path.open("w") as handle:
        handle.write(f"Bcast partitioned selection tree: {partition}\n")
        handle.write("=" * (34 + len(partition)) + "\n\n")
        handle.write(f"Rows: {len(rows)}\n")
        handle.write(f"Features: {', '.join(feature_names)}\n")
        handle.write(f"Training accuracy against {source_model_label} labels: {accuracy:.6f}\n")
        handle.write(f"\nOptimal algorithm counts from {source_model_label}:\n")
        for algorithm, count in sorted(counts.items()):
            handle.write(f"  {algorithm}: {count}\n")
        handle.write("\nClassification report:\n")
        handle.write(classification_report(y_labels, predictions, zero_division=0))
        handle.write("\nConfusion matrix rows/columns use this algorithm order:\n")
        handle.write("  " + ", ".join(model.classes_) + "\n")
        matrix = confusion_matrix(y_labels, predictions, labels=model.classes_)
        for row in matrix:
            handle.write("  " + ", ".join(str(int(value)) for value in row) + "\n")
        handle.write("\nSelection tree rules:\n")
        handle.write(export_text(model, feature_names=feature_names))


def save_partition_model(path, model, metadata):
    payload = {"model": model, "metadata": metadata}
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def fit_partition_tree(rows, feature_names, args):
    x_rows, y_labels = selection_features(rows, feature_names)
    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    model.fit(x_rows, y_labels)
    return model


def write_partition_outputs(output_dir, partition, rows, feature_names, args):
    partition_dir = output_dir / partition
    partition_dir.mkdir(parents=True, exist_ok=True)
    model = fit_partition_tree(rows, feature_names, args)

    text_path = partition_dir / "selection_tree.txt"
    dot_path = partition_dir / "selection_tree.dot"
    model_path = partition_dir / "selection_tree.pkl"
    metrics_path = partition_dir / "selection_metrics.txt"

    text_path.write_text(export_text(model, feature_names=feature_names))
    export_graphviz(
        model,
        out_file=str(dot_path),
        feature_names=feature_names,
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        impurity=True,
    )
    save_partition_model(
        model_path,
        model,
        {
            "model_file_format": "pickle",
            "model_type": "sklearn.tree.DecisionTreeClassifier",
            "source_model_label": args.source_model_label,
            "partition": partition,
            "manual_partition_condition": partition_condition(partition),
            "feature_names": list(feature_names),
            "feature_scale": "original",
            "target": "optimal_algorithm",
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
        },
    )
    write_partition_metrics(metrics_path, partition, rows, feature_names, model, args.source_model_label)
    return {
        "model": model,
        "text_path": text_path,
        "dot_path": dot_path,
        "model_path": model_path,
        "metrics_path": metrics_path,
    }


def partition_condition(partition):
    if partition == "single_node":
        return "nproc <= ppn"
    if partition == "regular_ppn":
        return "nproc > ppn and nproc % ppn == 0"
    if partition == "nonregular_ppn":
        return "nproc > ppn and nproc % ppn != 0"
    raise ValueError(f"unknown partition: {partition}")


def write_partitioned_rules_json(
    path,
    args,
    source_time_column,
    feature_names,
    ppn_values,
    nproc_values,
    message_sizes,
    algorithms,
    partition_rows,
    partition_models,
):
    data = OrderedDict()
    data["metadata"] = OrderedDict(
        [
            ("source_model", str(args.model)),
            ("source_model_label", args.source_model_label),
            ("source_model_time_column", source_time_column),
            ("methodology", "manual_partitioned_model_distillation"),
            ("feature_scale", "original"),
            ("tree_features", list(feature_names)),
            ("target", "optimal_algorithm"),
            ("candidate_algorithms", list(algorithms)),
            ("ppn_values", list(ppn_values)),
            ("nproc_values", list(nproc_values)),
            ("message_size_bytes_values", list(message_sizes)),
            ("max_depth", args.max_depth),
            ("min_samples_leaf", args.min_samples_leaf),
            (
                "coll_selection_note",
                "Intermediate rules. Current coll_selection.json does not directly encode ppn or modulo predicates.",
            ),
        ]
    )
    data["manual_partition_flow"] = OrderedDict(
        [
            (
                "if",
                OrderedDict(
                    [
                        ("condition", "nproc <= ppn"),
                        ("then_partition", "single_node"),
                        (
                            "else_if",
                            OrderedDict(
                                [
                                    ("condition", "nproc % ppn == 0"),
                                    ("then_partition", "regular_ppn"),
                                    ("else_partition", "nonregular_ppn"),
                                ]
                            ),
                        ),
                    ]
                ),
            )
        ]
    )
    data["partitions"] = OrderedDict()
    for partition in PARTITION_ORDER:
        rows = partition_rows.get(partition, [])
        model = partition_models.get(partition, {}).get("model")
        entry = OrderedDict(
            [
                ("condition", partition_condition(partition)),
                ("rows", len(rows)),
                ("tree_features", list(feature_names)),
            ]
        )
        if model is None:
            entry["tree"] = None
        else:
            entry["tree"] = tree_to_rule_json(model, feature_names)
        data["partitions"][partition] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")


def write_overall_metrics(path, partition_rows, feature_names, args, source_time_column):
    with path.open("w") as handle:
        handle.write("Bcast manual-partitioned decision tree generation\n")
        handle.write("=================================================\n\n")
        handle.write(f"Source {args.source_model_label} model: {args.model}\n")
        handle.write(f"Source model time column: {source_time_column}\n")
        handle.write(f"Features per partition tree: {', '.join(feature_names)}\n")
        handle.write(f"max_depth: {args.max_depth}\n")
        handle.write(f"min_samples_leaf: {args.min_samples_leaf}\n")
        handle.write("\nManual partitions:\n")
        for partition in PARTITION_ORDER:
            rows = partition_rows.get(partition, [])
            counts = Counter(row["optimal_algorithm"] for row in rows)
            handle.write(f"  {partition}: {len(rows)} rows, condition: {partition_condition(partition)}\n")
            for algorithm, count in sorted(counts.items()):
                handle.write(f"    {algorithm}: {count}\n")


def main():
    args = parse_args()
    require_sklearn()
    if not args.model.is_file():
        raise SystemExit(f"{args.source_model_label} model not found: {args.model}")

    feature_names = parse_feature_list(args.tree_features)
    rf_model, metadata = load_saved_model(args.model)
    source_time_column = model_time_column(metadata)
    require_model_time_column(
        metadata,
        args.require_model_time_column,
        args.source_model_label,
        args.retrain_command,
    )
    model_algorithms = algorithms_from_metadata(metadata)
    algorithms = parse_algorithm_list(args.candidate_algorithms, model_algorithms)
    ppn_values, nproc_values, message_sizes = resolve_grid(args)

    cell_rows, timing_rows = predict_grid(
        rf_model, ppn_values, nproc_values, message_sizes, algorithms, model_algorithms
    )
    partition_rows = {
        partition: rows_for_partition(cell_rows, partition) for partition in PARTITION_ORDER
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell_dataset_path = args.output_dir / "bcast_partitioned_grid_optimal_algorithms.csv"
    timing_dataset_path = args.output_dir / "bcast_partitioned_grid_algorithm_times.csv"
    rules_json_path = args.output_dir / "bcast_partitioned_decision_rules.json"
    metrics_path = args.output_dir / "bcast_partitioned_selection_metrics.txt"

    write_cell_dataset(cell_dataset_path, cell_rows, algorithms)
    write_timing_dataset(timing_dataset_path, timing_rows)

    partition_models = OrderedDict()
    for partition in PARTITION_ORDER:
        rows = partition_rows[partition]
        if not rows:
            continue
        partition_models[partition] = write_partition_outputs(
            args.output_dir, partition, rows, feature_names, args
        )

    write_partitioned_rules_json(
        rules_json_path,
        args,
        source_time_column,
        feature_names,
        ppn_values,
        nproc_values,
        message_sizes,
        algorithms,
        partition_rows,
        partition_models,
    )
    write_overall_metrics(metrics_path, partition_rows, feature_names, args, source_time_column)

    print(f"Loaded {args.source_model_label} model: {args.model}")
    print(f"Source model time column: {source_time_column}")
    print(
        f"Grid: {len(ppn_values)} ppn x {len(nproc_values)} nproc x "
        f"{len(message_sizes)} message-size = {len(cell_rows)} cells"
    )
    print(f"Algorithms compared per cell: {', '.join(algorithms)}")
    print(f"Wrote optimal-label grid: {cell_dataset_path}")
    print(f"Wrote per-algorithm timing grid: {timing_dataset_path}")
    for partition in PARTITION_ORDER:
        rows = partition_rows[partition]
        if not rows:
            print(f"Skipped empty partition: {partition}")
            continue
        outputs = partition_models[partition]
        print(f"Wrote {partition} tree text: {outputs['text_path']}")
        print(f"Wrote {partition} metrics: {outputs['metrics_path']}")
    print(f"Wrote partitioned decision rules JSON: {rules_json_path}")
    print(f"Wrote overall metrics: {metrics_path}")


if __name__ == "__main__":
    main()
