"""NYC Taxi model serving API — loads model from MLflow Registry.

Loads the 'champion' model alias from MLflow. Background thread checks
for new champion versions every 60 seconds — no pod restart needed.
Fetches zone-level features from Feast feature server at prediction time.
"""

import logging
import os
import sys
import threading
import time

import mlflow
import pandas as pd
import requests
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
MODEL_REFRESH_INTERVAL = int(os.environ.get("MODEL_REFRESH_INTERVAL", "60"))
FEAST_SERVER_URL = os.environ.get(
    "FEAST_SERVER_URL", "http://feast-server.feast.svc.cluster.local:6566"
)

DURATION_MODEL = None
CURRENT_MODEL_VERSION = None

ZONE_FEATURE_NAMES = [
    "zone_stats:zone_avg_fare",
    "zone_stats:zone_avg_duration_min",
    "zone_stats:zone_avg_distance",
]


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


def load_champion():
    """Load champion model from MLflow Registry. Returns (model, version) or raises."""
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(DURATION_MODEL_NAME, "champion")
    model_uri = f"models:/{DURATION_MODEL_NAME}@champion"
    model = mlflow.sklearn.load_model(model_uri)
    return model, mv.version


def refresh_model_loop():
    """Background thread: check for new champion version every N seconds."""
    global DURATION_MODEL, CURRENT_MODEL_VERSION
    while True:
        time.sleep(MODEL_REFRESH_INTERVAL)
        try:
            client = mlflow.tracking.MlflowClient()
            mv = client.get_model_version_by_alias(DURATION_MODEL_NAME, "champion")
            if mv.version != CURRENT_MODEL_VERSION:
                log.info("New champion detected: v%s (was v%s), reloading...",
                         mv.version, CURRENT_MODEL_VERSION)
                model, version = load_champion()
                DURATION_MODEL = model
                CURRENT_MODEL_VERSION = version
                log.info("Model refreshed to v%s", version)
        except Exception as e:
            log.debug("Model refresh check failed: %s", e)


def get_zone_features(zone_ids):
    """Fetch zone stats from Feast online store for given zone IDs."""
    try:
        resp = requests.post(
            f"{FEAST_SERVER_URL}/get-online-features",
            json={
                "features": ZONE_FEATURE_NAMES,
                "entities": {"zone_id": zone_ids},
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()

        feature_names = data["metadata"]["feature_names"]
        results = {}
        for i, name in enumerate(feature_names):
            results[name] = data["results"][i]["values"]

        return results
    except Exception as e:
        log.warning("Feast request failed, using defaults: %s", e)
        n = len(zone_ids)
        return {
            "zone_avg_fare": [0.0] * n,
            "zone_avg_duration_min": [0.0] * n,
            "zone_avg_distance": [0.0] * n,
        }


@app.on_event("startup")
def startup():
    global DURATION_MODEL, CURRENT_MODEL_VERSION

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    log.info("Loading champion model '%s' from MLflow Registry...", DURATION_MODEL_NAME)

    model, version = load_champion()
    DURATION_MODEL = model
    CURRENT_MODEL_VERSION = version
    log.info("Champion model loaded: v%s", version)

    # Start background refresh thread
    t = threading.Thread(target=refresh_model_loop, daemon=True)
    t.start()
    log.info("Model refresh thread started (interval=%ds)", MODEL_REFRESH_INTERVAL)


@app.get("/health")
def health():
    if DURATION_MODEL is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "healthy",
        "models": {"duration": DURATION_MODEL_NAME},
        "version": CURRENT_MODEL_VERSION,
    }


@app.post("/predict/duration", response_model=DurationResponse)
def predict_duration(request: DurationRequest):
    if DURATION_MODEL is None:
        raise HTTPException(status_code=503, detail="Duration model not loaded")

    # Fetch zone features from Feast for both pickup and dropoff zones
    zone_features = get_zone_features([request.pickup_zone_id, request.dropoff_zone_id])

    features = {
        "pickup_zone_id": request.pickup_zone_id,
        "dropoff_zone_id": request.dropoff_zone_id,
        "trip_distance": request.trip_distance,
        "pickup_hour": request.pickup_hour,
        "pickup_day_of_week": request.pickup_day_of_week,
        "pickup_month": request.pickup_month,
        "passenger_count": request.passenger_count,
        # Pickup zone stats (index 0)
        "pu_zone_avg_fare": zone_features["zone_avg_fare"][0],
        "pu_zone_avg_duration_min": zone_features["zone_avg_duration_min"][0],
        "pu_zone_avg_distance": zone_features["zone_avg_distance"][0],
        # Dropoff zone stats (index 1)
        "do_zone_avg_fare": zone_features["zone_avg_fare"][1],
        "do_zone_avg_duration_min": zone_features["zone_avg_duration_min"][1],
        "do_zone_avg_distance": zone_features["zone_avg_distance"][1],
    }

    df = pd.DataFrame([features])
    # Ensure all zone features are float (Feast may return None)
    for col in df.columns:
        if col.startswith(("pu_zone_", "do_zone_")):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    prediction = DURATION_MODEL.predict(df)[0]

    log.info(
        "Duration prediction: %.2f min (zone %d→%d, %.1f mi, model v%s)",
        prediction,
        request.pickup_zone_id,
        request.dropoff_zone_id,
        request.trip_distance,
        CURRENT_MODEL_VERSION,
    )

    return DurationResponse(
        predicted_duration_min=round(float(prediction), 2),
        model_name=DURATION_MODEL_NAME,
        model_version=CURRENT_MODEL_VERSION,
    )
