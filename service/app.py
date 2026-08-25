 
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("nova_pay_fraud_model_lean.joblib")))


def load_artifact() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    required = {"model", "feature_cols", "threshold"}
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"Model artifact is missing keys: {sorted(missing)}")
    return artifact


artifact = load_artifact()
model = artifact["model"]
FEATURES = list(artifact["feature_cols"])
THRESH = float(artifact["threshold"])

app = FastAPI(title="Nova Pay Fraud Scoring API", version="2.0")    # app title and version for the OpenAPI docs

# Pydantic BaseModel built from the model's OWN feature list (lean form).
class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    txn_velocity_1h: float
    txn_velocity_24h: float
    ip_risk_score: float
    device_trust_score: float
    country_location_mismatch: int = Field(ge=0, le=1)
    amount_usd: float = Field(ge=0)

class ScoreResponse(BaseModel):
    fraud_probability: float                                        # probability of fraud (0.0-1.0)
    is_fraud: bool                                                  # whether the transaction is classified as fraud (True/False)
    threshold: float                                                # the threshold used for classification (0.0-1.0)

@app.get("/health")                                                 # health check endpoint for monitoring
def health(): 
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        load_artifact()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model is not ready") from exc
    return {"status": "ready", "n_features": len(FEATURES)}

@app.post("/score", response_model=ScoreResponse)                                   # score endpoint for scoring transactions
def score(txn: Transaction):
    row = np.array([[getattr(txn, feature) for feature in FEATURES]], dtype=float)
    prob = float(model.predict_proba(row)[0, 1])
    return ScoreResponse(
        fraud_probability=round(prob, 4),
        is_fraud=prob >= THRESH,
        threshold=THRESH,
    )
