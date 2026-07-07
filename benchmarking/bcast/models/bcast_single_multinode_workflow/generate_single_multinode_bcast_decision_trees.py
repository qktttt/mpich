#!/usr/bin/env python3
"""Generate separate Bcast selection trees from saved single-node/multinode RF models.

This script is the second stage of the single-node/multinode Bcast workflow.
The first stage trains two RandomForestRegressor timing models, one for
single-node cases and one for multinode cases.  This script loads those saved
timing models and distills each one into a shallow, readable algorithm
selection tree.

The workflow is:
1. Load a saved RF timing model for each requested partition.
2. Build a grid of ppn, nproc, message_size_bytes, and candidate algorithms.
3. Ask the RF model to predict each algorithm's runtime for every grid point.
4. Label each grid point with the fastest RF-predicted algorithm.
5. Train a DecisionTreeClassifier that maps runtime features to that algorithm.
6. Export CSV data, raw tree artifacts, simplified tree text, metrics, and JSON.

The final decision tree predicts an algorithm class.  It does not predict time.
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


WORKFLOW_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = WORKFLOW_DIR / "outputs"
PARTITIONS = {
    "single_node": "nproc <= ppn",
    "multinode": "nproc > ppn",
}

accuracy_score = None
classification_report = None
confusion_matrix = None
DecisionTreeClassifier = None
export_graphviz = None
export_text = None
_tree = None


def require_sklearn():
    """Import sklearn lazily and fail with a workflow-specific message.

    The saved RF models and the distilled selection tree are sklearn objects.
    Importing sklearn here keeps the module import lightweight and produces a
    clearer error if the user runs this script in a Python environment that can
    read the file but cannot load or train the models.
    """
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
            "scikit-learn is required to generate Bcast decision trees. "
            "Run this script in the same Python environment used to train the RF models."
        ) from exc

    accuracy_score = imported_accuracy_score
    classification_report = imported_classification_report
    confusion_matrix = imported_confusion_matrix
    DecisionTreeClassifier = imported_decision_tree_classifier
    export_graphviz = imported_export_graphviz
    export_text = imported_export_text
    _tree = imported_tree


def parse_args():
    """Parse command-line options that control model loading and tree export."""
    parser = argparse.ArgumentParser(
        description=(
            "Load saved single-node and multinode Bcast random-forest timing models "
            "and distill each one into a shallow optimal-algorithm decision tree."
        )
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Directory containing single_node/bcast_random_forest_model.pkl and "
            "multinode/bcast_random_forest_model.pkl."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --model-dir.",
    )
    parser.add_argument(
        "--partitions",
        default="single_node,multinode",
        help="Comma-separated partition list to generate. Available: single_node,multinode.",
    )
    parser.add_argument(
        "--ppn-values",
        default=None,
        help="Optional comma-separated ppn grid. Defaults to values stored in each RF model metadata.",
    )
    parser.add_argument(
        "--nproc-values",
        default=None,
        help="Optional comma-separated nproc grid. Defaults to values stored in each RF model metadata.",
    )
    parser.add_argument(
        "--message-size-values",
        default=None,
        help=(
            "Optional comma-separated message-size grid in bytes. Defaults to values "
            "stored in each RF model metadata."
        ),
    )
    parser.add_argument(
        "--candidate-algorithms",
        default=None,
        help="Optional comma-separated algorithm subset. Defaults to algorithms stored in each RF model metadata.",
    )
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--tree-features",
        default="nproc,ppn,message_size_bytes",
        help=(
            "Comma-separated feature columns for each decision tree. Available: "
            "nproc, ppn, message_size_bytes, estimated_nodes, last_node_ranks."
        ),
    )
    return parser.parse_args()


def install_numpy_pickle_shims():
    """Install compatibility aliases for NumPy pickle module path changes.

    Some saved sklearn/numpy objects refer to NumPy 2 module paths such as
    ``numpy._core``.  Older NumPy versions expose the same objects under
    ``numpy.core``.  These aliases let pickle resolve either spelling when
    loading models created in a different NumPy environment.
    """
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
    """Load a saved model pickle and return ``(model, metadata)``.

    The workflow saves models as ``{"model": model, "metadata": metadata}``.
    This function also accepts a bare model pickle and returns an empty metadata
    dictionary for that case.
    """
    install_numpy_pickle_shims()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("metadata", {})
    return payload, {}


def parse_partition_list(value):
    """Parse and validate the comma-separated partition list."""
    partitions = [part.strip() for part in value.split(",") if part.strip()]
    if not partitions:
        raise ValueError("partition list is empty")
    unknown = [partition for partition in partitions if partition not in PARTITIONS]
    if unknown:
        raise ValueError("unknown partition(s): " + ", ".join(unknown))
    return partitions


def parse_int_list(value, name):
    """Parse a comma-separated positive integer list.

    Returns ``None`` when the user did not provide an override, allowing the
    caller to fall back to values recorded in the saved RF model metadata.
    """
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
    """Resolve candidate algorithms against the algorithms known by the RF model."""
    if value is None:
        return list(model_algorithms)
    requested = [part.strip() for part in value.split(",") if part.strip()]
    if not requested:
        raise ValueError("candidate algorithm list is empty")
    missing = [name for name in requested if name not in model_algorithms]
    if missing:
        raise ValueError("candidate algorithm(s) are not present in the saved model: " + ", ".join(missing))
    return requested


def parse_feature_list(value):
    """Parse and validate feature names used by the distilled selection tree."""
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


def algorithms_from_metadata(metadata):
    """Extract algorithm names from saved RF model metadata.

    Newer workflow artifacts store an explicit ``algorithms`` list.  Older
    artifacts may only store feature names such as ``algorithm=binomial``; this
    function supports both formats.
    """
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


def observed_domain_values(metadata, key):
    """Read observed training-domain values from RF metadata."""
    domain = metadata.get("data_domain") or {}
    values = domain.get(key)
    if values:
        return sorted({int(value) for value in values})
    raise ValueError(f"saved model metadata does not contain data_domain.{key}")


def resolve_grid(metadata, args):
    """Resolve ppn, nproc, and message-size grid values.

    User-provided command-line values take precedence.  Otherwise the script
    reuses the observed domain values saved with the RF model, so the selection
    tree is distilled over the same discrete domain the timing model saw.
    """
    ppn_values = parse_int_list(args.ppn_values, "ppn")
    if ppn_values is None:
        ppn_values = observed_domain_values(metadata, "ppn_values")

    nproc_values = parse_int_list(args.nproc_values, "nproc")
    if nproc_values is None:
        nproc_values = observed_domain_values(metadata, "nproc_values")

    message_sizes = parse_int_list(args.message_size_values, "message size")
    if message_sizes is None:
        message_sizes = observed_domain_values(metadata, "message_size_bytes_values")

    return ppn_values, nproc_values, message_sizes


def is_in_partition(partition, nproc, ppn):
    """Return whether ``(nproc, ppn)`` belongs to a workflow partition."""
    if partition == "single_node":
        return nproc <= ppn
    if partition == "multinode":
        return nproc > ppn
    raise ValueError(f"unknown partition: {partition}")


def estimated_nodes(nproc, ppn):
    """Estimate how many nodes are needed for ``nproc`` ranks at this ppn."""
    return int(math.ceil(float(nproc) / float(ppn)))


def last_node_ranks(nproc, ppn):
    """Return the number of ranks on the last node for this ``nproc``/``ppn``."""
    remainder = nproc % ppn
    return ppn if remainder == 0 else remainder


def build_rf_feature_rows(ppn, nproc, message_size, algorithms, model_algorithms):
    """Build RF timing-model feature rows for every candidate algorithm.

    The RF timing models were trained with log-scaled numeric features plus an
    algorithm one-hot vector:

    ``log2(nproc), log2(message_size_bytes), log2(ppn), algorithm_one_hot...``

    For one grid point, we create one row per candidate algorithm so the RF can
    predict the runtime of each option under identical runtime conditions.
    """
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


def predict_grid(partition, model, ppn_values, nproc_values, message_sizes, algorithms, model_algorithms):
    """Evaluate the RF model on a grid and label each cell with the best algorithm.

    ``cell_rows`` has one row per runtime condition and stores the selected
    fastest algorithm.  ``timing_rows`` has one row per runtime condition per
    algorithm and stores all predicted timings for inspection/debugging.
    """
    cell_rows = []
    timing_rows = []

    for ppn in ppn_values:
        for nproc in nproc_values:
            if not is_in_partition(partition, nproc, ppn):
                continue
            nodes = estimated_nodes(nproc, ppn)
            tail_ranks = last_node_ranks(nproc, ppn)
            for message_size in message_sizes:
                features = build_rf_feature_rows(ppn, nproc, message_size, algorithms, model_algorithms)
                predicted_log_times = model.predict(features)
                # RF predicts log2(time), matching the training target; convert
                # back to seconds before comparing algorithms.
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

    if not cell_rows:
        raise ValueError(f"grid for {partition} is empty")
    return cell_rows, timing_rows


def selection_features(rows, feature_names):
    """Convert RF-labeled grid rows into classifier inputs and labels."""
    x_rows = []
    y_labels = []
    for row in rows:
        x_rows.append([float(row[name]) for name in feature_names])
        y_labels.append(row["optimal_algorithm"])
    return np.asarray(x_rows, dtype=float), np.asarray(y_labels)


def algorithm_leaf_container(algorithm):
    """Translate a workflow algorithm label into a CSEL-style container.

    The tree JSON is an intermediate format, but keeping CSEL container names in
    the leaves makes it easier to convert the result into ``coll_selection.json``
    later.  ``release_gather`` is guarded by its own CSEL condition because that
    algorithm is only valid when the CH4 POSIX release-gather checks pass.
    """
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
    """Convert a trained sklearn tree into a readable nested rule dictionary."""
    tree = model.tree_
    classes = list(model.classes_)

    def convert_node(node_id):
        """Recursively convert one sklearn tree node."""
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


def leaf_algorithms(rule_node):
    """Return the set of algorithms used by all leaves below ``rule_node``."""
    if rule_node["type"] == "leaf":
        return {rule_node["algorithm"]}
    return leaf_algorithms(rule_node["yes"]) | leaf_algorithms(rule_node["no"])


def simplify_rule_tree(rule_node):
    """Collapse redundant subtrees that always choose the same algorithm.

    sklearn may keep a split even when both sides of that split eventually lead
    to the same class.  That can happen because the classifier is optimizing the
    full tree impurity under a depth constraint, not minimizing the human-visible
    rule count.  For CSEL-style rules, those splits are unnecessary: if every
    leaf below a subtree selects ``release_gather``, the whole subtree is
    equivalent to a single ``release_gather`` leaf.
    """
    if rule_node["type"] == "leaf":
        return rule_node

    simplified = OrderedDict(rule_node)
    simplified["yes"] = simplify_rule_tree(rule_node["yes"])
    simplified["no"] = simplify_rule_tree(rule_node["no"])

    algorithms = leaf_algorithms(simplified)
    if len(algorithms) == 1:
        algorithm = next(iter(algorithms))
        return OrderedDict(
            [
                ("type", "leaf"),
                ("algorithm", algorithm),
                ("samples", simplified["samples"]),
                ("class_counts", simplified["class_counts"]),
                ("csel_container", algorithm_leaf_container(algorithm)),
                ("simplified_from", simplified["condition"]),
            ]
        )

    return simplified


def rule_tree_to_text(rule_node, depth=0):
    """Render the simplified rule dictionary in sklearn-like text-tree format."""
    prefix = "|   " * depth
    if rule_node["type"] == "leaf":
        return f"{prefix}|--- class: {rule_node['algorithm']}\n"

    feature = rule_node["feature"]
    threshold = rule_node["threshold"]
    text = f"{prefix}|--- {feature} <= {threshold:.2f}\n"
    text += rule_tree_to_text(rule_node["yes"], depth + 1)
    text += f"{prefix}|--- {feature} >  {threshold:.2f}\n"
    text += rule_tree_to_text(rule_node["no"], depth + 1)
    return text


def write_cell_dataset(path, rows, algorithms):
    """Write one row per grid cell with the RF-selected optimal algorithm."""
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_timing_dataset(path, rows):
    """Write one row per grid cell per algorithm with RF-predicted timings."""
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_selection_model(path, model, metadata):
    """Save the distilled sklearn DecisionTreeClassifier plus metadata."""
    with path.open("wb") as handle:
        pickle.dump({"model": model, "metadata": metadata}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_metrics(path, partition, rows, feature_names, model, source_model_path, simplified_text):
    """Write accuracy and diagnostic metrics for the distilled selection tree."""
    counts = Counter(row["optimal_algorithm"] for row in rows)
    x_rows, y_labels = selection_features(rows, feature_names)
    predictions = model.predict(x_rows)
    accuracy = accuracy_score(y_labels, predictions)

    with path.open("w") as handle:
        handle.write(f"Bcast {partition} RF-distilled selection tree\n")
        handle.write("=" * (41 + len(partition)) + "\n\n")
        handle.write(f"Source random forest model: {source_model_path}\n")
        handle.write(f"Partition condition: {PARTITIONS[partition]}\n")
        handle.write(f"Rows: {len(rows)}\n")
        handle.write(f"Features: {', '.join(feature_names)}\n")
        handle.write(f"Training accuracy against RF labels: {accuracy:.6f}\n")
        handle.write("\nOptimal algorithm counts from RF:\n")
        for algorithm, count in sorted(counts.items()):
            handle.write(f"  {algorithm}: {count}\n")
        handle.write("\nClassification report:\n")
        handle.write(classification_report(y_labels, predictions, zero_division=0))
        handle.write("\nConfusion matrix rows/columns use this algorithm order:\n")
        handle.write("  " + ", ".join(model.classes_) + "\n")
        matrix = confusion_matrix(y_labels, predictions, labels=model.classes_)
        for row in matrix:
            handle.write("  " + ", ".join(str(int(value)) for value in row) + "\n")
        handle.write("\nSimplified selection tree rules:\n")
        handle.write(simplified_text)
        handle.write("\nRaw sklearn selection tree rules:\n")
        handle.write(export_text(model, feature_names=feature_names))


def write_rule_json(
    path,
    partition,
    source_model_path,
    feature_names,
    algorithms,
    grid_values,
    model,
    rows,
    args,
    simplified_tree,
):
    """Write the simplified decision rules as a structured JSON artifact."""
    data = OrderedDict()
    data["metadata"] = OrderedDict(
        [
            ("source_model", str(source_model_path)),
            ("source_model_label", "random_forest"),
            ("methodology", "single_multinode_model_distillation"),
            ("partition", partition),
            ("partition_condition", PARTITIONS[partition]),
            ("feature_scale", "original"),
            ("tree_features", list(feature_names)),
            ("target", "optimal_algorithm"),
            ("candidate_algorithms", list(algorithms)),
            ("ppn_values", list(grid_values["ppn_values"])),
            ("nproc_values", list(grid_values["nproc_values"])),
            ("message_size_bytes_values", list(grid_values["message_sizes"])),
            ("max_depth", args.max_depth),
            ("min_samples_leaf", args.min_samples_leaf),
            (
                "tree_simplification",
                "collapsed any subtree whose leaves all select the same algorithm",
            ),
        ]
    )
    data["partition"] = OrderedDict(
        [
            ("condition", PARTITIONS[partition]),
            ("rows", len(rows)),
            ("tree_features", list(feature_names)),
            ("tree", simplified_tree),
        ]
    )
    with path.open("w") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")


def fit_and_write_partition_tree(partition, rf_model, rf_metadata, source_model_path, output_dir, args):
    """Distill one partition's RF timing model into a selection tree.

    This is the main per-partition pipeline: resolve the RF domain, generate
    RF labels, train the classifier, simplify its exported rules, and write all
    artifacts for that partition.
    """
    model_algorithms = algorithms_from_metadata(rf_metadata)
    algorithms = parse_algorithm_list(args.candidate_algorithms, model_algorithms)
    feature_names = parse_feature_list(args.tree_features)
    ppn_values, nproc_values, message_sizes = resolve_grid(rf_metadata, args)
    cell_rows, timing_rows = predict_grid(
        partition, rf_model, ppn_values, nproc_values, message_sizes, algorithms, model_algorithms
    )
    x_rows, y_labels = selection_features(cell_rows, feature_names)
    selection_tree = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    selection_tree.fit(x_rows, y_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_dataset_path = output_dir / "bcast_rf_grid_optimal_algorithms.csv"
    timing_dataset_path = output_dir / "bcast_rf_grid_algorithm_times.csv"
    text_path = output_dir / "selection_tree.txt"
    raw_text_path = output_dir / "selection_tree_raw.txt"
    dot_path = output_dir / "selection_tree.dot"
    model_path = output_dir / "selection_tree.pkl"
    metrics_path = output_dir / "selection_metrics.txt"
    rules_json_path = output_dir / "decision_rules.json"
    grid_values = {
        "ppn_values": ppn_values,
        "nproc_values": nproc_values,
        "message_sizes": message_sizes,
    }

    write_cell_dataset(cell_dataset_path, cell_rows, algorithms)
    write_timing_dataset(timing_dataset_path, timing_rows)
    raw_rule_tree = tree_to_rule_json(selection_tree, feature_names)
    simplified_rule_tree = simplify_rule_tree(raw_rule_tree)
    simplified_text = rule_tree_to_text(simplified_rule_tree)
    text_path.write_text(simplified_text)
    raw_text_path.write_text(export_text(selection_tree, feature_names=feature_names))
    export_graphviz(
        selection_tree,
        out_file=str(dot_path),
        feature_names=feature_names,
        class_names=list(selection_tree.classes_),
        filled=True,
        rounded=True,
        impurity=True,
    )
    save_selection_model(
        model_path,
        selection_tree,
        {
            "model_file_format": "pickle",
            "model_type": "sklearn.tree.DecisionTreeClassifier",
            "source_random_forest_model": str(source_model_path),
            "partition": partition,
            "partition_condition": PARTITIONS[partition],
            "feature_names": list(feature_names),
            "feature_scale": "original",
            "target": "optimal_algorithm",
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "algorithms": list(algorithms),
            "ppn_values": list(ppn_values),
            "nproc_values": list(nproc_values),
            "message_size_bytes_values": list(message_sizes),
        },
    )
    write_metrics(
        metrics_path,
        partition,
        cell_rows,
        feature_names,
        selection_tree,
        source_model_path,
        simplified_text,
    )
    write_rule_json(
        rules_json_path,
        partition,
        source_model_path,
        feature_names,
        algorithms,
        grid_values,
        selection_tree,
        cell_rows,
        args,
        simplified_rule_tree,
    )

    return {
        "cell_rows": len(cell_rows),
        "text_path": text_path,
        "raw_text_path": raw_text_path,
        "dot_path": dot_path,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "rules_json_path": rules_json_path,
        "cell_dataset_path": cell_dataset_path,
        "timing_dataset_path": timing_dataset_path,
    }


def main():
    """Run the workflow for each requested partition."""
    args = parse_args()
    require_sklearn()
    output_root = args.output_dir if args.output_dir is not None else args.model_dir
    partitions = parse_partition_list(args.partitions)

    for partition in partitions:
        model_path = args.model_dir / partition / "bcast_random_forest_model.pkl"
        if not model_path.is_file():
            raise SystemExit(f"Random-forest model not found for {partition}: {model_path}")

        rf_model, rf_metadata = load_saved_model(model_path)
        output_dir = output_root / partition / "selection_tree"
        outputs = fit_and_write_partition_tree(
            partition, rf_model, rf_metadata, model_path, output_dir, args
        )
        print(f"\nPartition {partition}: {PARTITIONS[partition]}")
        print(f"  Loaded RF model: {model_path}")
        print(f"  Grid cells: {outputs['cell_rows']}")
        print(f"  Wrote tree text: {outputs['text_path']}")
        print(f"  Wrote raw tree text: {outputs['raw_text_path']}")
        print(f"  Wrote tree model: {outputs['model_path']}")
        print(f"  Wrote metrics: {outputs['metrics_path']}")
        print(f"  Wrote rule JSON: {outputs['rules_json_path']}")


if __name__ == "__main__":
    main()
