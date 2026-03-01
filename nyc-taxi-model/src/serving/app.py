"""NYC Taxi model serving API — loads model from MLflow Registry."""

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

app = FastAPI(title="NYC Taxi Prediction API")

DURATION_MODEL_NAME = os.environ.get("DURATION_MODEL_NAME", "taxi-duration")
DURATION_MODEL = None


class DurationRequest(BaseModel):
    pickup_zone_id: int = 161
    dropoff_zone_id: int = 236
    trip_distance: float = 3.5
    pickup_hour: int = 14
    pickup_day_of_week: int = 2
    pickup_month: int = 1
    passenger_count: int = 1


class DurationResponse(BaseModel):
    predicted_duration_min: float
    model_name: str
    model_version: str


@app.on_event("startup")
def load_models():
    global DURATION_MODEL

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    log.info("Loading duration model '%s' from MLflow Registry...", DURATION_MODEL_NAME)
    model_uri = f"models:/{DURATION_MODEL_NAME}/latest"
    DURATION_MODEL = mlflow.sklearn.load_model(model_uri)
    log.info("Duration model loaded")


@app.get("/health")
def health():
    if DURATION_MODEL is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "models": {"duration": DURATION_MODEL_NAME}}


@app.post("/predict/duration", response_model=DurationResponse)
def predict_duration(request: DurationRequest):
    if DURATION_MODEL is None:
        raise HTTPException(status_code=503, detail="Duration model not loaded")

    df = pd.DataFrame([request.model_dump()])
    prediction = DURATION_MODEL.predict(df)[0]

    log.info(
        "Duration prediction: %.2f min (zone %d→%d, %.1f mi)",
        prediction,
        request.pickup_zone_id,
        request.dropoff_zone_id,
        request.trip_distance,
    )

    return DurationResponse(
        predicted_duration_min=round(float(prediction), 2),
        model_name=DURATION_MODEL_NAME,
        model_version="latest",
    )
