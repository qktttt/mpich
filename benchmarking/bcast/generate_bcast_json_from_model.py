#!/usr/bin/env python3
"""Generate MPICH Bcast selection JSON from a saved fitted model."""

import argparse
import csv
import importlib
import importlib.util
import json
import math
import pickle
import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MPICH_ROOT = SCRIPT_DIR.parent.parent
REGULAR_JSON_DIR = MPICH_ROOT / "regular_collective_json_generation_files"
DEFAULT_MODEL = SCRIPT_DIR / "models" / "bcast_random_forest_model.pkl"
DEFAULT_SOURCE_MODEL_LABEL = "random_forest"
DEFAULT_RETRAIN_COMMAND = (
    "python3 benchmarking/bcast/fit_bcast_random_forest.py --time-column {required}"
)
DEFAULT_TEMPLATE = MPICH_ROOT / "src" / "mpi" / "coll" / "coll_selection.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "models" / "bcast_coll_selection.json"
DEFAULT_SECTION_OUTPUT = SCRIPT_DIR / "models" / "bcast_intra_auto_generated.json"
DEFAULT_SUMMARY = SCRIPT_DIR / "plots" / "bcast_median_summary.csv"

# These are the Bcast algorithms already used by this checkout's CSEL JSON.
# The benchmark also measures nb, circ_graph, smp, tree, pipelined_tree, and
# release_gather, but those need extra conditions or CSEL parameters that are
# not represented in the current default Bcast JSON.
DEFAULT_POF2_CANDIDATES = [
    "binomial",
    "scatter_recursive_doubling_allgather",
    "scatter_ring_allgather",
]
DEFAULT_ANY_CANDIDATES = [
    "binomial",
    "scatter_ring_allgather",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MPICH Bcast coll_selection JSON from a saved fitted model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
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
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--section-output",
        type=Path,
        default=DEFAULT_SECTION_OUTPUT,
        help="Write only the generated name=bcast-intra-auto section to this JSON file.",
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--nproc-values", default=None, help="Comma-separated total rank counts.")
    parser.add_argument("--ppn-values", default=None, help="Comma-separated ranks-per-node values.")
    parser.add_argument("--message-size-values", default=None, help="Comma-separated byte counts.")
    parser.add_argument(
        "--candidate-algorithms",
        default=",".join(DEFAULT_POF2_CANDIDATES),
        help="Comma-separated algorithms to compare for the pof2 branch.",
    )
    parser.add_argument(
        "--non-pof2-candidate-algorithms",
        default=",".join(DEFAULT_ANY_CANDIDATES),
        help="Comma-separated algorithms to compare for the non-pof2 fallback branch.",
    )
    parser.add_argument(
        "--allow-multiple-ppn",
        action="store_true",
        help="Allow a grid with multiple ppn values. Current MPICH CSEL JSON cannot condition on ppn.",
    )
    return parser.parse_args()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_src_import_shims(param_module):
    src_module = types.ModuleType("src")
    active_learner_module = types.ModuleType("src.active_learner")
    algs_module = types.ModuleType("src.active_learner.algs")
    user_config_module = types.ModuleType("src.user_config")
    config_manager_module = types.ModuleType("src.user_config.config_manager")
    json_file_module = types.ModuleType("src.json_file")

    def missing_read_algs(*_args, **_kwargs):
        raise RuntimeError("read_algs is not used; algorithms come from the saved model metadata")

    def missing_add_algs(*_args, **_kwargs):
        raise RuntimeError("add_algs is replaced with a Bcast model adapter")

    class ConfigManager:
        @classmethod
        def get_instance(cls):
            return cls()

        def get_value(self, *_args, **_kwargs):
            raise RuntimeError("ConfigManager is not used by this standalone generator")

    algs_module.read_algs = missing_read_algs
    algs_module.add_algs = missing_add_algs
    config_manager_module.ConfigManager = ConfigManager

    sys.modules.setdefault("src", src_module)
    sys.modules.setdefault("src.active_learner", active_learner_module)
    sys.modules["src.active_learner.algs"] = algs_module
    sys.modules.setdefault("src.user_config", user_config_module)
    sys.modules["src.user_config.config_manager"] = config_manager_module
    sys.modules.setdefault("src.json_file", json_file_module)
    sys.modules["src.json_file.param_algs_to_json"] = param_module


def load_regular_json_helpers():
    param_module = load_module(
        "regular_collective_param_algs_to_json",
        REGULAR_JSON_DIR / "param_algs_to_json.py",
    )
    install_src_import_shims(param_module)
    json_module = load_module(
        "regular_collective_json_file",
        REGULAR_JSON_DIR / "json_file.py",
    )
    return json_module, param_module


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
        raise ValueError(
            f"Saved {source_model_label} model was not trained with the required timing column. "
            f"required={required}, actual={actual_label}. "
            f"Retrain with: {command}"
        )


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
        raise ValueError(f"{name} did not contain any values")
    return sorted(set(values))


def parse_algorithm_list(value):
    algorithms = [part.strip() for part in value.split(",") if part.strip()]
    if not algorithms:
        raise ValueError("algorithm list cannot be empty")
    return algorithms


def is_power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def feature_log(value):
    if not is_power_of_two(value):
        raise ValueError(
            f"{value} is not a power of two; the provided regular collective workflow "
            "stores integer log2 feature-space points"
        )
    return int(math.log2(value)) + 1


def read_summary_domain(path):
    if not path.is_file():
        return None

    triples = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"nproc", "ppn", "message_size_bytes"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required column(s): {', '.join(sorted(missing))}")
        for row in reader:
            nproc = int(row["nproc"])
            ppn = int(row["ppn"])
            message_size = int(row["message_size_bytes"])
            if nproc % ppn != 0:
                continue
            nodes = nproc // ppn
            if is_power_of_two(nodes) and is_power_of_two(ppn) and is_power_of_two(message_size):
                triples.add((nproc, ppn, message_size))

    if not triples:
        raise ValueError(f"{path}: no power-of-two Bcast feature points found")
    return sorted(triples)


def metadata_domain(metadata):
    domain = metadata.get("data_domain") or {}
    feature_points = domain.get("feature_points") or []
    triples = []
    for point in feature_points:
        if len(point) != 3:
            continue
        nproc, ppn, message_size = (int(point[0]), int(point[1]), int(point[2]))
        if nproc % ppn != 0:
            continue
        nodes = nproc // ppn
        if is_power_of_two(nodes) and is_power_of_two(ppn) and is_power_of_two(message_size):
            triples.append((nproc, ppn, message_size))
    return sorted(set(triples)) or None


def cli_domain(args):
    nproc_values = parse_int_list(args.nproc_values, "nproc")
    ppn_values = parse_int_list(args.ppn_values, "ppn")
    message_sizes = parse_int_list(args.message_size_values, "message size")
    if nproc_values is None or ppn_values is None or message_sizes is None:
        return None

    triples = []
    for nproc in nproc_values:
        for ppn in ppn_values:
            if nproc % ppn != 0:
                raise ValueError(f"nproc={nproc} is not divisible by ppn={ppn}")
            nodes = nproc // ppn
            feature_log(nodes)
            feature_log(ppn)
            for message_size in message_sizes:
                feature_log(message_size)
                triples.append((nproc, ppn, message_size))
    return sorted(set(triples))


def resolve_domain(args, metadata):
    triples = cli_domain(args)
    if triples is None:
        triples = metadata_domain(metadata)
    if triples is None:
        triples = read_summary_domain(args.summary_csv)
    if triples is None:
        raise ValueError(
            "could not infer feature grid; pass --nproc-values, --ppn-values, "
            "and --message-size-values"
        )

    ppn_values = sorted({ppn for _, ppn, _ in triples})
    if len(ppn_values) != 1 and not args.allow_multiple_ppn:
        raise ValueError(
            "current MPICH coll_selection.json cannot condition on ppn; pass a single "
            "ppn value or use --allow-multiple-ppn to flatten ppn-specific rules"
        )
    return triples


def build_feature_space(triples):
    rows = []
    for nproc, ppn, message_size in triples:
        nodes = nproc // ppn
        rows.append([feature_log(nodes), feature_log(ppn), feature_log(message_size)])
    return np.asarray(rows, dtype=float)


def add_model_algs(feature_space, algs):
    feature_space = np.asarray(feature_space, dtype=float)
    if np.squeeze(feature_space).ndim == 1:
        feature_space = np.reshape(feature_space, (1, feature_space.size))

    rows = []
    for feature_row in feature_space:
        for alg_id in sorted(algs):
            rows.append(list(feature_row) + [alg_id])
    return np.asarray(rows, dtype=float)


class BcastModelAdapter:
    def __init__(self, model, model_algorithms):
        self.model = model
        self.model_algorithms = list(model_algorithms)
        self.algorithm_to_index = {name: idx for idx, name in enumerate(self.model_algorithms)}

    def predict(self, old_rows):
        old_rows = np.asarray(old_rows, dtype=float)
        if old_rows.ndim == 1:
            old_rows = np.reshape(old_rows, (1, old_rows.size))

        converted_rows = []
        for row in old_rows:
            node_log_plus_one, ppn_log_plus_one, msg_log_plus_one, alg_id = row
            log2_nodes = node_log_plus_one - 1.0
            log2_ppn = ppn_log_plus_one - 1.0
            log2_nproc = log2_nodes + log2_ppn
            log2_message_size = msg_log_plus_one - 1.0
            algorithm = self.current_algs[int(alg_id)]
            one_hot = [0.0] * len(self.model_algorithms)
            one_hot[self.algorithm_to_index[algorithm]] = 1.0
            converted_rows.append([log2_nproc, log2_message_size, log2_ppn] + one_hot)
        return self.model.predict(np.asarray(converted_rows, dtype=float))

    def with_algs(self, algs):
        self.current_algs = dict(algs)
        return self


def algorithms_from_metadata(metadata):
    algorithms = metadata.get("algorithms")
    if algorithms:
        return list(algorithms)
    feature_names = metadata.get("feature_names") or []
    algorithms = []
    for name in feature_names:
        if name.startswith("algorithm="):
            algorithms.append(name.split("=", 1)[1])
    if algorithms:
        return algorithms
    raise ValueError("saved model metadata does not contain algorithm names")


def make_candidate_dict(requested, model_algorithms):
    missing = [name for name in requested if name not in model_algorithms]
    if missing:
        raise ValueError(
            "candidate algorithm(s) are not present in the saved model metadata: "
            + ", ".join(missing)
        )
    return OrderedDict((idx, name) for idx, name in enumerate(requested))


def algorithm_container(algorithm, param_module):
    alg_str, param_value = param_module.split_param_alg(algorithm)
    if param_value is not None and alg_str == "tree":
        params = {
            "tree_type=knomial_1": {},
            f"k={param_value}": {},
            "is_non_blocking=0": {},
        }
    elif algorithm == "tree":
        params = {
            "tree_type=kary": {},
            "k=2": {},
            "is_non_blocking=0": {},
        }
    elif algorithm == "pipelined_tree":
        params = {
            "tree_type=kary": {},
            "k=2": {},
            "is_non_blocking=0": {},
            "chunk_size=0": {},
            "recv_pre_posted=0": {},
        }
    elif param_value is not None:
        params = param_module.get_param_rules("Bcast", alg_str, param_value) or {}
    else:
        params = {}

    return {f"algorithm=MPIR_Bcast_intra_{alg_str}": params}


def rules_to_current_bcast_dict(rules, algs, param_module):
    by_comm_size = OrderedDict()

    for feature_set_bytes, alg_id in rules.items():
        feature_set = np.frombuffer(feature_set_bytes, dtype=int)
        nodes = int(2 ** (feature_set[0] - 1))
        ppn = int(2 ** (feature_set[1] - 1))
        message_size = int(2 ** (feature_set[2] - 1))
        comm_size = nodes * ppn
        comm_key = f"comm_size<={comm_size}"
        msg_key = f"avg_msg_size<={message_size}"
        if comm_key not in by_comm_size:
            by_comm_size[comm_key] = OrderedDict()
        by_comm_size[comm_key][msg_key] = algorithm_container(algs[alg_id], param_module)

    return by_comm_size


def normalize_any_keys(node):
    if not isinstance(node, dict):
        return node

    normalized = OrderedDict()
    for key, value in node.items():
        new_key = "any" if isinstance(key, str) and key.endswith("=any") else key
        normalized[new_key] = normalize_any_keys(value)
    return normalized


def collapse_single_any(node):
    if not isinstance(node, dict):
        return node

    collapsed = OrderedDict((key, collapse_single_any(value)) for key, value in node.items())
    if list(collapsed.keys()) == ["any"]:
        return collapsed["any"]
    return collapsed


def current_csel_sort_key(key):
    if key == "any":
        return (2, "", 0, key)
    for operator in ("<=", "<"):
        if operator in key:
            name, value = key.split(operator, 1)
            try:
                threshold = int(value)
            except ValueError:
                threshold = 0
            return (0, name, threshold, key)
    return (1, key, 0, key)


def sort_current_csel_dict(node):
    if not isinstance(node, dict):
        return node
    return OrderedDict(
        (key, sort_current_csel_dict(value))
        for key, value in sorted(node.items(), key=lambda item: current_csel_sort_key(item[0]))
    )


def generate_branch(json_module, param_module, adapter, feature_space, algs):
    adapter.with_algs(algs)
    json_module.add_algs = add_model_algs
    predictions = adapter.predict(add_model_algs(feature_space, algs))
    selections = json_module.get_selections(predictions, algs)
    rules = json_module.get_rules(feature_space, selections, algs, adapter)
    branch = rules_to_current_bcast_dict(rules, algs, param_module)
    json_module.any_helper(branch)
    return sort_current_csel_dict(collapse_single_any(normalize_any_keys(branch)))


def generated_bcast_section(pof2_branch, any_branch):
    return OrderedDict(
        [
            (
                "MPIDI_CH4_release_gather",
                {"algorithm=MPIDI_POSIX_mpi_bcast_release_gather": {}},
            ),
            ("comm_size<8", {"algorithm=MPIR_Bcast_intra_binomial": {}}),
            ("pof2", pof2_branch),
            ("any", any_branch),
        ]
    )


def write_output(template_path, output_path, bcast_section):
    with template_path.open() as handle:
        data = json.load(handle, object_pairs_hook=OrderedDict)

    if "name=bcast-intra-auto" not in data:
        raise ValueError(f"{template_path}: missing name=bcast-intra-auto")

    data["name=bcast-intra-auto"] = bcast_section
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")


def write_bcast_section_output(output_path, bcast_section):
    section_data = OrderedDict([("name=bcast-intra-auto", bcast_section)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(section_data, handle, indent=4)
        handle.write("\n")


def main():
    args = parse_args()
    json_module, param_module = load_regular_json_helpers()
    model, metadata = load_saved_model(args.model)
    require_model_time_column(
        metadata,
        args.require_model_time_column,
        args.source_model_label,
        args.retrain_command,
    )
    source_time_column = model_time_column(metadata)
    model_algorithms = algorithms_from_metadata(metadata)
    triples = resolve_domain(args, metadata)
    feature_space = build_feature_space(triples)

    pof2_requested = parse_algorithm_list(args.candidate_algorithms)
    any_requested = parse_algorithm_list(args.non_pof2_candidate_algorithms)
    pof2_algs = make_candidate_dict(pof2_requested, model_algorithms)
    any_algs = make_candidate_dict(any_requested, model_algorithms)

    adapter = BcastModelAdapter(model, model_algorithms)
    pof2_branch = generate_branch(json_module, param_module, adapter, feature_space, pof2_algs)
    any_branch = generate_branch(json_module, param_module, adapter, feature_space, any_algs)
    bcast_section = generated_bcast_section(pof2_branch, any_branch)
    write_output(args.template, args.output, bcast_section)
    write_bcast_section_output(args.section_output, bcast_section)

    print(f"Loaded fitted {args.source_model_label} model: {args.model}")
    print(f"Source model time column: {source_time_column}")
    print(f"Feature points: {feature_space.shape[0]}")
    print(f"pof2 candidates: {', '.join(pof2_algs.values())}")
    print(f"non-pof2 candidates: {', '.join(any_algs.values())}")
    print(f"Wrote Bcast coll selection JSON: {args.output}")
    print(f"Wrote generated Bcast section JSON: {args.section_output}")


if __name__ == "__main__":
    main()
