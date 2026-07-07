#!/usr/bin/env python3
"""Generate MPICH Bcast selection JSON from the fitted MLP timing model."""

from pathlib import Path

import generate_bcast_json_from_model as generator


SCRIPT_DIR = Path(__file__).resolve().parent

generator.DEFAULT_MODEL = SCRIPT_DIR / "models" / "bcast_mlp_model.pkl"
generator.DEFAULT_OUTPUT = SCRIPT_DIR / "models" / "bcast_mlp_coll_selection.json"
generator.DEFAULT_SECTION_OUTPUT = SCRIPT_DIR / "models" / "bcast_mlp_intra_auto_generated.json"
generator.DEFAULT_SOURCE_MODEL_LABEL = "mlp"
generator.DEFAULT_RETRAIN_COMMAND = (
    "python3 benchmarking/bcast/fit_bcast_mlp.py --time-column {required}"
)


if __name__ == "__main__":
    generator.main()
