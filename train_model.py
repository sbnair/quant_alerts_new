#!/usr/bin/env python3
"""
V2 training CLI.

    python train_model.py            # backfill labels, then train if enough rows
    python train_model.py --labels   # only backfill labels

Labels are the realized next-session range for every logged ScanResult;
the model is a LightGBM regressor whose predictions replace the hand-tuned
percentile weights (blended by config.ML_BLEND) once instance/lgbm_ranker.txt
exists.
"""

import json
import sys

from app import app
from scanner import ml_model


def main():
    if "--labels" in sys.argv:
        n = ml_model.backfill_labels(app)
        print(f"labels backfilled: {n}")
        return
    meta = ml_model.train(app)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
