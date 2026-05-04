"""Task 1 — train Ridge and RandomForest, log to MLflow, save best by MAE."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_data.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MLRUNS_DIR = ROOT / "mlruns"

FEATURES = ["patients_ahead", "staff_count", "is_emergency", "dept_load"]
TARGET = "wait_time_min"
EXPERIMENT_NAME = "mediqueue-wait-time-min"


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    runs: list[dict] = []
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("team", "ml_engineering")
            mlflow.log_params(model.get_params())

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate(y_test.to_numpy(), preds)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(2))

            local_path = MODELS_DIR / f"{name.lower()}.joblib"
            joblib.dump(model, local_path)

            runs.append(
                {
                    "name": name,
                    "run_id": run.info.run_id,
                    "model_path": str(local_path),
                    **metrics,
                }
            )
            print(f"[{name}] run_id={run.info.run_id} mae={metrics['mae']:.4f}")

    best = min(runs, key=lambda r: r["mae"])
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(joblib.load(best["model_path"]), best_path)

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [
            {
                "name": r["name"],
                "mae": round(r["mae"], 6),
                "rmse": round(r["rmse"], 6),
                "r2": round(r["r2"], 6),
                "mape": round(r["mape"], 6),
            }
            for r in runs
        ],
        "best_model": best["name"],
        "best_metric_name": "mae",
        "best_metric_value": round(best["mae"], 6),
    }
    (RESULTS_DIR / "step1_s1.json").write_text(json.dumps(payload, indent=2))

    meta = {"best_model": best["name"], "run_id": best["run_id"], "mae": best["mae"]}
    (MODELS_DIR / "best_model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Best model: {best['name']} (MAE={best['mae']:.4f})")


if __name__ == "__main__":
    main()
