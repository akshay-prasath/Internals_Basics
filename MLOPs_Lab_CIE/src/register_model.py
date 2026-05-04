"""Task 4 — register the best run's model in the MLflow Model Registry."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = ROOT / "mlruns"
RESULTS_DIR = ROOT / "results"
META_PATH = ROOT / "models" / "best_model_meta.json"
REGISTERED_NAME = "mediqueue-wait-time-min-predictor"


def main() -> None:
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    client = MlflowClient()

    meta = json.loads(META_PATH.read_text())
    run_id = meta["run_id"]
    run = client.get_run(run_id)
    mae = float(run.data.metrics["mae"])

    try:
        client.create_registered_model(REGISTERED_NAME)
    except mlflow.exceptions.MlflowException:
        pass

    model_uri = f"runs:/{run_id}/model"
    mv = client.create_model_version(name=REGISTERED_NAME, source=model_uri, run_id=run_id)
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "source_metric", "mae")
    client.set_model_version_tag(REGISTERED_NAME, mv.version, "source_metric_value", f"{mae}")

    payload = {
        "registered_model_name": REGISTERED_NAME,
        "version": int(mv.version),
        "run_id": run_id,
        "source_metric": "mae",
        "source_metric_value": round(mae, 6),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "step4_s6.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
