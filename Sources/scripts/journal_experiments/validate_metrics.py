#!/usr/bin/env python3
"""Recompute classification metrics from a raw prediction artifact."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.training_metrics import validate_prediction_artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="NPZ containing predictions and labels")
    args = parser.parse_args(argv)
    metrics = validate_prediction_artifact(args.artifact)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    main()
