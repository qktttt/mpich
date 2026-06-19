"""Shared utilities for Bcast timing regression models."""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


RawGroupKey = Tuple[int, int, str, int]


def default_input_csvs(script_path: str) -> List[Path]:
    script_dir = Path(script_path).resolve().parent
    default_summary = script_dir / "plots" / "bcast_median_summary.csv"
    if default_summary.exists():
        return [default_summary]
    return sorted(script_dir.glob("bcast_bench_ppn*r_*.csv"))


def parse_ppn(row: Dict[str, str], path: Path, fallback_ppn: Optional[int]) -> int:
    ppn_value = row.get("ppn", "").strip()
    if ppn_value:
        return int(ppn_value)

    match = re.search(r"(?:^|_)ppn(\d+)(?:_|$)", path.name)
    if match:
        return int(match.group(1))

    if fallback_ppn is not None:
        return fallback_ppn

    raise ValueError(f"{path}: could not determine ppn; pass --ppn or use ppn in the filename")


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_median_rows(
    input_csvs: Sequence[Path], time_column: str, phase: str, fallback_ppn: Optional[int]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    raw_timings: DefaultDict[RawGroupKey, List[float]] = defaultdict(list)
    summary_target_column = f"median_{time_column}"

    for path in input_csvs:
        if not path.is_file():
            raise FileNotFoundError(str(path))

        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if summary_target_column in fieldnames:
                required = {"ppn", "nproc", "algorithm", "message_size_bytes", summary_target_column}
                missing = required - fieldnames
                if missing:
                    raise ValueError(f"{path}: missing required column(s): {', '.join(sorted(missing))}")
                for row in reader:
                    rows.append(
                        {
                            "ppn": int(row["ppn"]),
                            "nproc": int(row["nproc"]),
                            "algorithm": row["algorithm"].strip(),
                            "message_size_bytes": int(row["message_size_bytes"]),
                            "median_time_sec": float(row[summary_target_column]),
                        }
                    )
            else:
                required = {"phase", "algorithm", "nproc", "message_size_bytes", time_column}
                missing = required - fieldnames
                if missing:
                    raise ValueError(f"{path}: missing required column(s): {', '.join(sorted(missing))}")
                for row in reader:
                    if row["phase"].strip().lower() != phase.lower():
                        continue
                    if "correct" in row and not is_true(row["correct"]):
                        continue
                    key = (
                        parse_ppn(row, path, fallback_ppn),
                        int(row["nproc"]),
                        row["algorithm"].strip(),
                        int(row["message_size_bytes"]),
                    )
                    raw_timings[key].append(float(row[time_column]))

    if raw_timings:
        for (ppn, nproc, algorithm, message_size), values in sorted(raw_timings.items()):
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
        raise ValueError("no usable median timing rows found")
    return rows


def log2_positive(value: float, name: str) -> float:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive for log2 transformation; got {value}")
    return math.log(value, 2.0)


def inverse_log2(value: float) -> float:
    return 2.0 ** value


def build_feature_matrix(
    rows: Sequence[Dict[str, object]]
) -> Tuple[List[List[float]], List[float], List[float], List[str], List[str]]:
    algorithms = sorted({str(row["algorithm"]) for row in rows})
    feature_names = ["log2(nproc)", "log2(message_size_bytes)", "log2(ppn)"] + [
        f"algorithm={name}" for name in algorithms
    ]
    x_rows: List[List[float]] = []
    y_log_values: List[float] = []
    y_original_values: List[float] = []

    for row in rows:
        algorithm = str(row["algorithm"])
        features = [
            log2_positive(float(row["nproc"]), "nproc"),
            log2_positive(float(row["message_size_bytes"]), "message_size_bytes"),
            log2_positive(float(row["ppn"]), "ppn"),
        ]
        features.extend(1.0 if algorithm == name else 0.0 for name in algorithms)
        target = float(row["median_time_sec"])
        x_rows.append(features)
        y_log_values.append(log2_positive(target, "median_time_sec"))
        y_original_values.append(target)

    return x_rows, y_log_values, y_original_values, feature_names, algorithms


def split_indices(
    n_rows: int, train_frac: float, val_frac: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    if train_frac <= 0.0 or val_frac <= 0.0 or train_frac + val_frac >= 1.0:
        raise ValueError("require 0 < train_frac, 0 < val_frac, and train_frac + val_frac < 1")

    from sklearn.model_selection import train_test_split

    test_frac = 1.0 - train_frac - val_frac
    all_indices = list(range(n_rows))
    train_val_indices, test_indices = train_test_split(
        all_indices, test_size=test_frac, random_state=seed, shuffle=True
    )
    relative_val_frac = val_frac / (train_frac + val_frac)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=relative_val_frac, random_state=seed + 1, shuffle=True
    )
    return list(train_indices), list(val_indices), list(test_indices)


def parse_depth_candidates(value: str) -> List[int]:
    depths = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        depth = int(part)
        if depth < 1:
            raise ValueError("max_depth candidates must be positive")
        depths.append(depth)
    if not depths:
        raise ValueError("no max_depth candidates provided")
    return depths


def subset_rows(rows: Sequence[Sequence[float]], indices: Sequence[int]) -> List[Sequence[float]]:
    return [rows[index] for index in indices]


def subset_values(values: Sequence[float], indices: Sequence[int]) -> List[float]:
    return [values[index] for index in indices]


def evaluate_log_model(
    model,
    x_rows: Sequence[Sequence[float]],
    y_log_values: Sequence[float],
    y_original_values: Sequence[float],
    indices: Sequence[int],
) -> Dict[str, float]:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        r2_score,
    )

    x_subset = subset_rows(x_rows, indices)
    y_log_true = subset_values(y_log_values, indices)
    y_original_true = subset_values(y_original_values, indices)
    y_log_pred = list(model.predict(x_subset))
    y_original_pred = [inverse_log2(value) for value in y_log_pred]

    return {
        "rmse": math.sqrt(mean_squared_error(y_original_true, y_original_pred)),
        "mae": mean_absolute_error(y_original_true, y_original_pred),
        "r2": r2_score(y_original_true, y_original_pred),
        "mape": 100.0 * mean_absolute_percentage_error(y_original_true, y_original_pred),
        "log_rmse": math.sqrt(mean_squared_error(y_log_true, y_log_pred)),
        "log_mae": mean_absolute_error(y_log_true, y_log_pred),
        "log_r2": r2_score(y_log_true, y_log_pred),
    }


def prediction_rows(
    model,
    rows: Sequence[Dict[str, object]],
    x_rows: Sequence[Sequence[float]],
    y_log_values: Sequence[float],
    y_original_values: Sequence[float],
    split_name: str,
    indices: Sequence[int],
) -> Iterable[List[object]]:
    y_log_pred = list(model.predict(subset_rows(x_rows, indices)))
    for index, log_prediction in zip(indices, y_log_pred):
        prediction = inverse_log2(log_prediction)
        actual = y_original_values[index]
        ape = abs((prediction - actual) / actual) * 100.0 if actual != 0.0 else 0.0
        row = rows[index]
        yield [
            split_name,
            index,
            row["ppn"],
            row["nproc"],
            row["algorithm"],
            row["message_size_bytes"],
            actual,
            prediction,
            y_log_values[index],
            log_prediction,
            ape,
        ]


def human_seconds(value: float) -> str:
    abs_value = abs(value)
    if abs_value < 1e-6:
        return f"{value * 1e9:.3g} ns"
    if abs_value < 1e-3:
        return f"{value * 1e6:.3g} us"
    if abs_value < 1.0:
        return f"{value * 1e3:.3g} ms"
    return f"{value:.3g} s"


def human_bytes(value: float) -> str:
    if value < 1024:
        return f"{int(value)} B"
    scaled = value
    for unit in ("KiB", "MiB", "GiB"):
        scaled /= 1024.0
        if scaled < 1024.0:
            return f"{scaled:g} {unit}"
    return f"{scaled:g} GiB"


def sklearn_missing_message(exc: BaseException) -> str:
    return (
        "scikit-learn is required for this script. Install it in the Python "
        f"environment you use to run the model, e.g. `python3 -m pip install scikit-learn`. "
        f"Original import error: {type(exc).__name__}: {exc}"
    )
