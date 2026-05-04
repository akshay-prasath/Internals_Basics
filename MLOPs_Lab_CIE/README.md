# MediQueue Wait-Time Predictor

MLOps CIE submission — predict outpatient wait times for MediQueue using two regression models, track experiments with MLflow, ship the predictor as a Docker CLI, serve it through FastAPI, and version the production model in the MLflow Model Registry.

**Course:** 24AM6AEMLO — MLOps, BMS College of Engineering
**Question Paper Code:** mlops-cie-210

## Dataset

`data/training_data.csv` — 25 rows, 4 features, target `wait_time_min`.

| Feature | Range |
| --- | --- |
| `patients_ahead` | 1–30 |
| `staff_count` | 2–15 |
| `is_emergency` | 0–1 |
| `dept_load` | 1–5 |

Train/test split is fixed: `test_size=0.2`, `random_state=42`.

## Repository layout

```
MLOPs_Lab_CIE/
├── data/
│   └── training_data.csv
├── src/
│   ├── train.py            # Task 1 — trains Ridge + RandomForest, logs to MLflow
│   ├── predict_cli.py      # Task 2 — argparse CLI, container entrypoint
│   ├── api.py              # Task 3 — FastAPI service on port 8888
│   └── register_model.py   # Task 4 — registers best run in MLflow Registry
├── models/                 # saved joblib artifacts (best_model.joblib + meta)
├── results/
│   ├── step1_s1.json       # Task 1 proof
│   ├── step2_s3.json       # Task 2 proof
│   ├── step3_s4.json       # Task 3 proof
│   └── step4_s6.json       # Task 4 proof
├── Dockerfile              # python:3.10-slim, runs predict_cli.py
├── requirements.txt
└── .gitignore
```

## Setup

```bash
cd MLOPs_Lab_CIE
python3 -m pip install --user -r requirements.txt
```

## Task 1 — Experiment Tracking & Model Comparison

Train Ridge and RandomForest, log params + MAE/RMSE/R²/MAPE to MLflow under experiment `mediqueue-wait-time-min` with tag `team=ml_engineering`, save the lower-MAE model.

```bash
python3 src/train.py
```

Produces:
- `models/best_model.joblib` and `models/best_model_meta.json`
- `results/step1_s1.json`
- MLflow runs in `./mlruns/`

**Result:** Ridge wins — MAE **5.386**, RMSE 6.595, R² 0.814, MAPE 0.099. RandomForest MAE 10.39 on this small dataset.

Inspect the runs visually:

```bash
python3 -m mlflow ui --backend-store-uri file:./mlruns --port 5000
# → http://localhost:5000
```

## Task 2 — Docker Packaging

Containerise the CLI predictor.

```bash
docker build -t mediqueue-predictor:v1 .
docker run --rm mediqueue-predictor:v1 \
  --patients_ahead 14 --staff_count 10 --is_emergency 0 --dept_load 3
# → {"prediction": 46.8627}
```

The image is `python:3.10-slim`, installs `requirements.txt`, and copies in `src/` + `models/`. Entry point is `python src/predict_cli.py`. Proof: `results/step2_s3.json`.

## Task 3 — FastAPI Serving

Serve the best model on port **8888** with Pydantic input validation.

```bash
cd src
python3 -m uvicorn api:app --host 0.0.0.0 --port 8888
```

Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | `{"status":"running","model":"Ridge","version":"1.0"}` |
| `POST` | `/score` | accepts the 4 features, returns `{"prediction": <float>}` |

Test it:

```bash
curl http://127.0.0.1:8888/health

curl -X POST http://127.0.0.1:8888/score \
  -H 'Content-Type: application/json' \
  -d '{"patients_ahead": 14, "staff_count": 10, "is_emergency": 0, "dept_load": 3}'
# → {"prediction":46.8627}

# Out-of-range value → HTTP 422
curl -i -X POST http://127.0.0.1:8888/score \
  -H 'Content-Type: application/json' \
  -d '{"patients_ahead": 99, "staff_count": 10, "is_emergency": 0, "dept_load": 3}'
```

Pydantic enforces the ranges from the dataset spec; invalid inputs return HTTP 422. Proof: `results/step3_s4.json`.

## Task 4 — Model Versioning

Register the best run's model in the MLflow Model Registry as `mediqueue-wait-time-min-predictor` and link it to the originating run.

```bash
python3 src/register_model.py
```

Produces `results/step4_s6.json` with the registered name, version number, source `run_id`, and the MAE the version was selected on. Browse the registry in the MLflow UI under the **Models** tab.

## Reproducing everything end-to-end

```bash
python3 src/train.py            # Task 1
docker build -t mediqueue-predictor:v1 .
docker run --rm mediqueue-predictor:v1 \
  --patients_ahead 14 --staff_count 10 --is_emergency 0 --dept_load 3   # Task 2
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8888 &             # Task 3
python3 src/register_model.py   # Task 4
```

All four `results/*.json` files are the machine-readable proof of execution.
