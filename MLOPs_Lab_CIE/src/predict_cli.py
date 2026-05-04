"""Task 2 — CLI predictor consumed by the Docker image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
FEATURES = ["patients_ahead", "staff_count", "is_emergency", "dept_load"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MediQueue wait-time CLI predictor")
    p.add_argument("--patients_ahead", type=int, required=True)
    p.add_argument("--staff_count", type=int, required=True)
    p.add_argument("--is_emergency", type=int, required=True, choices=[0, 1])
    p.add_argument("--dept_load", type=int, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame(
        [[args.patients_ahead, args.staff_count, args.is_emergency, args.dept_load]],
        columns=FEATURES,
    )
    pred = float(model.predict(row)[0])
    print(json.dumps({"prediction": round(pred, 4)}))


if __name__ == "__main__":
    main()
