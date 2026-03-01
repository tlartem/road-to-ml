"""Model serving API — loads model from MLflow Registry."""

import logging
import os
import sys

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

app = FastAPI(title="Water Quality Prediction API")

MODEL_NAME = os.environ.get("MODEL_NAME", "water-quality-classifier")
MODEL = None


class PredictRequest(BaseModel):
    ph: float = 7.0
    Hardness: float = 200.0
    Solids: float = 20000.0
    Chloramines: float = 7.0
    Sulfate: float = 330.0
    Conductivity: float = 400.0
    Organic_carbon: float = 14.0
    Trihalomethanes: float = 66.0
    Turbidity: float = 4.0


class PredictResponse(BaseModel):
    potability: int
    probability: float
    model_name: str
    model_version: str


@app.on_event("startup")
def load_model():
    global MODEL

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    log.info("Loading model '%s' from MLflow Registry...", MODEL_NAME)

    model_uri = f"models:/{MODEL_NAME}/latest"
    MODEL = mlflow.sklearn.load_model(model_uri)
    log.info("Model loaded successfully")


@app.get("/health")
def health():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([request.model_dump()])
    prediction = MODEL.predict(df)[0]
    probability = MODEL.predict_proba(df)[0][1]

    log.info("Prediction: potability=%d, probability=%.4f", prediction, probability)

    return PredictResponse(
        potability=int(prediction),
        probability=round(float(probability), 4),
        model_name=MODEL_NAME,
        model_version="latest",
    )
