"""NYC Taxi model serving API — 3 models from MLflow Registry.

Models: duration, fare, demand. Each uses 'champion' alias.
Background thread refreshes models every 60s — no pod restart needed.
Zone features fetched from Feast server at prediction time.
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

MODEL_REFRESH_INTERVAL = int(os.environ.get("MODEL_REFRESH_INTERVAL", "60"))
FEAST_SERVER_URL = os.environ.get(
    "FEAST_SERVER_URL", "http://feast-server.feast.svc.cluster.local:6566"
)

# Model registry: name → {"model": model_obj, "version": str}
MODEL_NAMES = ["taxi-duration", "taxi-fare", "taxi-demand"]
MODELS = {}

ZONE_FEATURE_NAMES = [
    "zone_stats:zone_avg_fare",
    "zone_stats:zone_avg_duration_min",
    "zone_stats:zone_avg_distance",
    "zone_stats:zone_trip_count",
]


# --- Request/Response schemas ---

class TripRequest(BaseModel):
    """Shared request for duration and fare prediction."""
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


class FareResponse(BaseModel):
    predicted_fare: float
    model_name: str
    model_version: str


class DemandRequest(BaseModel):
    zone_id: int = 161
    pickup_hour: int = 14
    day_of_week: int = 2
    month: int = 1


class DemandResponse(BaseModel):
    predicted_trip_count: float
    model_name: str
    model_version: str


# --- Model loading ---

def load_champion(model_name):
    """Load champion model. Returns (model, version) or raises."""
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(model_name, "champion")
    model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
    return model, mv.version


def refresh_model_loop():
    """Background: check for new champion versions every N seconds."""
    while True:
        time.sleep(MODEL_REFRESH_INTERVAL)
        client = mlflow.tracking.MlflowClient()
        for name in MODEL_NAMES:
            if name not in MODELS:
                # Try loading newly available models
                try:
                    model, version = load_champion(name)
                    MODELS[name] = {"model": model, "version": version}
                    log.info("Loaded new model %s v%s", name, version)
                except Exception:
                    pass
                continue
            try:
                mv = client.get_model_version_by_alias(name, "champion")
                if mv.version != MODELS[name]["version"]:
                    log.info("New champion for %s: v%s → v%s", name, MODELS[name]["version"], mv.version)
                    model, version = load_champion(name)
                    MODELS[name] = {"model": model, "version": version}
            except Exception:
                pass


def get_zone_features(zone_ids):
    """Fetch zone stats from Feast online store."""
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
            "zone_trip_count": [0.0] * n,
        }


def build_trip_features(request: TripRequest):
    """Build feature dict for duration/fare models (trip features + zone stats)."""
    zone_features = get_zone_features([request.pickup_zone_id, request.dropoff_zone_id])

    features = {
        "pickup_zone_id": request.pickup_zone_id,
        "dropoff_zone_id": request.dropoff_zone_id,
        "trip_distance": request.trip_distance,
        "pickup_hour": request.pickup_hour,
        "pickup_day_of_week": request.pickup_day_of_week,
        "pickup_month": request.pickup_month,
        "passenger_count": request.passenger_count,
        "pu_zone_avg_fare": zone_features["zone_avg_fare"][0],
        "pu_zone_avg_duration_min": zone_features["zone_avg_duration_min"][0],
        "pu_zone_avg_distance": zone_features["zone_avg_distance"][0],
        "do_zone_avg_fare": zone_features["zone_avg_fare"][1],
        "do_zone_avg_duration_min": zone_features["zone_avg_duration_min"][1],
        "do_zone_avg_distance": zone_features["zone_avg_distance"][1],
    }

    df = pd.DataFrame([features])
    for col in df.columns:
        if col.startswith(("pu_zone_", "do_zone_")):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


# --- Startup ---

@app.on_event("startup")
def startup():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    for name in MODEL_NAMES:
        try:
            model, version = load_champion(name)
            MODELS[name] = {"model": model, "version": version}
            log.info("Loaded %s champion v%s", name, version)
        except Exception as e:
            log.warning("Could not load %s: %s (will retry in background)", name, e)

    if not MODELS:
        log.error("No models loaded — at least one champion must exist")
        sys.exit(1)

    t = threading.Thread(target=refresh_model_loop, daemon=True)
    t.start()
    log.info("Model refresh thread started (interval=%ds)", MODEL_REFRESH_INTERVAL)


# --- Endpoints ---

@app.get("/health")
def health():
    loaded = {name: info["version"] for name, info in MODELS.items()}
    return {"status": "healthy", "models": loaded}


@app.post("/predict/duration", response_model=DurationResponse)
def predict_duration(request: TripRequest):
    if "taxi-duration" not in MODELS:
        raise HTTPException(status_code=503, detail="Duration model not loaded")

    df = build_trip_features(request)
    m = MODELS["taxi-duration"]
    prediction = m["model"].predict(df)[0]

    log.info("Duration: %.2f min (zone %d→%d)", prediction, request.pickup_zone_id, request.dropoff_zone_id)

    return DurationResponse(
        predicted_duration_min=round(float(prediction), 2),
        model_name="taxi-duration",
        model_version=m["version"],
    )


@app.post("/predict/fare", response_model=FareResponse)
def predict_fare(request: TripRequest):
    if "taxi-fare" not in MODELS:
        raise HTTPException(status_code=503, detail="Fare model not loaded")

    df = build_trip_features(request)
    m = MODELS["taxi-fare"]
    prediction = m["model"].predict(df)[0]

    log.info("Fare: $%.2f (zone %d→%d)", prediction, request.pickup_zone_id, request.dropoff_zone_id)

    return FareResponse(
        predicted_fare=round(float(prediction), 2),
        model_name="taxi-fare",
        model_version=m["version"],
    )


@app.post("/predict/demand", response_model=DemandResponse)
def predict_demand(request: DemandRequest):
    if "taxi-demand" not in MODELS:
        raise HTTPException(status_code=503, detail="Demand model not loaded")

    # Demand uses zone stats directly (not pu/do split)
    zone_features = get_zone_features([request.zone_id])
    features = {
        "zone_id": request.zone_id,
        "pickup_hour": request.pickup_hour,
        "day_of_week": request.day_of_week,
        "month": request.month,
        "zone_avg_fare": zone_features["zone_avg_fare"][0],
        "zone_avg_duration_min": zone_features["zone_avg_duration_min"][0],
        "zone_avg_distance": zone_features["zone_avg_distance"][0],
        "zone_trip_count": zone_features["zone_trip_count"][0],
    }

    df = pd.DataFrame([features])
    for col in ["zone_avg_fare", "zone_avg_duration_min", "zone_avg_distance", "zone_trip_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    m = MODELS["taxi-demand"]
    prediction = m["model"].predict(df)[0]

    log.info("Demand: %.1f trips (zone %d, hour %d)", prediction, request.zone_id, request.pickup_hour)

    return DemandResponse(
        predicted_trip_count=round(float(prediction), 1),
        model_name="taxi-demand",
        model_version=m["version"],
    )
