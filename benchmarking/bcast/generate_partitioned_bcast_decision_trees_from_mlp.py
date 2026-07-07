#!/usr/bin/env python3
"""Generate manually partitioned Bcast decision trees from the fitted MLP model."""

from pathlib import Path

import generate_partitioned_bcast_decision_trees as generator


SCRIPT_DIR = Path(__file__).resolve().parent

generator.DEFAULT_MODEL = SCRIPT_DIR / "models" / "bcast_mlp_model.pkl"
generator.DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "models" / "bcast_mlp_partitioned_selection"
generator.DEFAULT_SOURCE_MODEL_LABEL = "mlp"
generator.DEFAULT_RETRAIN_COMMAND = (
    "python3 benchmarking/bcast/fit_bcast_mlp.py --time-column {required}"
)


if __name__ == "__main__":
    generator.main()
