"""Task 3 — FastAPI service serving the best model on port 8888."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "best_model_meta.json"
FEATURES = ["patients_ahead", "staff_count", "is_emergency", "dept_load"]

_meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {"best_model": "unknown"}
_model = joblib.load(MODEL_PATH)

app = FastAPI(title="MediQueue Wait Time API", version="1.0")


class WaitTimeRequest(BaseModel):
    patients_ahead: int = Field(..., ge=1, le=30)
    staff_count: int = Field(..., ge=2, le=15)
    is_emergency: int = Field(..., ge=0, le=1)
    dept_load: int = Field(..., ge=1, le=5)


class WaitTimeResponse(BaseModel):
    prediction: float


@app.get("/health")
def health() -> dict:
    return {"status": "running", "model": _meta.get("best_model", "unknown"), "version": "1.0"}


@app.post("/score", response_model=WaitTimeResponse)
def score(req: WaitTimeRequest) -> WaitTimeResponse:
    row = pd.DataFrame([req.model_dump()], columns=FEATURES)
    pred = float(_model.predict(row)[0])
    return WaitTimeResponse(prediction=round(pred, 4))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8888, reload=False)
