import os
import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import hopsworks
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aqi_api")

TARGET_COLUMNS = ["target_aqi_day1", "target_aqi_day2", "target_aqi_day3"]

POLLUTANT_COLUMNS = ["pm2_5", "pm10", "co", "no2", "so2", "ozone"]
WEATHER_COLUMNS = ["temperature", "humidity", "pressure", "wind_speed"]

MODEL_NAMES = {
    "day1": "lahore_aqi_day1",
    "day2": "lahore_aqi_day2",
    "day3": "lahore_aqi_day3",
}

# AQI category logic

AQI_CATEGORIES = [
    (0, 50, "Good", "#00E400", False),
    (51, 100, "Moderate", "#FFFF00", False),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF7E00", False),
    (151, 200, "Unhealthy", "#FF0000", True),
    (201, 300, "Very Unhealthy", "#8F3F97", True),
    (301, 500, "Hazardous", "#7E0023", True),
]

HEALTH_ADVICE = {
    "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
    "Moderate": "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Sensitive groups (children, elderly, respiratory/heart conditions) should reduce prolonged outdoor exertion.",
    "Unhealthy": "Everyone may begin to experience health effects. Limit prolonged outdoor exertion.",
    "Very Unhealthy": "Health alert: everyone may experience more serious health effects. Avoid outdoor activity.",
    "Hazardous": "Health emergency. Everyone should avoid all outdoor exertion and stay indoors.",
}


def classify_aqi(aqi: float) -> dict:
    aqi = round(aqi)
    for low, high, label, color, alert in AQI_CATEGORIES:
        if low <= aqi <= high:
            return {
                "category": label,
                "color": color,
                "hazardous_alert": alert,
                "advice": HEALTH_ADVICE[label],
            }
    return {
        "category": "Hazardous",
        "color": "#7E0023",
        "hazardous_alert": True,
        "advice": HEALTH_ADVICE["Hazardous"],
    }


# Global cache: populated once at startup, reused across requests
state: dict = {
    "project": None,
    "feature_store": None,
    "model_registry": None,
    "models": {},   
    "model_meta": {},    
    "model_features": None,
    "cache": {"latest_date": None, "result": None},
}


def get_latest_model_meta(model_registry, model_name: str):
    models = model_registry.get_models(name=model_name)
    if not models:
        raise RuntimeError(f"No versions found for model '{model_name}'")
    return max(models, key=lambda m: m.version)


def extract_metrics(model_meta) -> dict:
    """Hopsworks stores metrics logged at training time on `training_metrics`."""
    raw = getattr(model_meta, "training_metrics", None) or {}

    def pick(*keys):
        for k in keys:
            if k in raw:
                try:
                    return round(float(raw[k]), 4)
                except (TypeError, ValueError):
                    return raw[k]
        return None

    return {
        "r2": pick("r2", "R2", "r2_score"),
        "rmse": pick("rmse", "RMSE"),
        "mae": pick("mae", "MAE"),
    }


def load_models():
    """Login to Hopsworks and load feature store + models once."""
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY environment variable is not set")

    logger.info("Logging into Hopsworks...")
    project = hopsworks.login(
        api_key_value=os.environ.get("HOPSWORKS_API_KEY")
        )
    feature_store = project.get_feature_store()
    model_registry = project.get_model_registry()

    models, model_meta = {}, {}
    for day, model_name in MODEL_NAMES.items():
        logger.info(f"Loading latest version of {model_name}...")
        meta = get_latest_model_meta(model_registry, model_name)
        model_dir = meta.download()
        model = joblib.load(os.path.join(model_dir, f"{model_name}.pkl"))
        models[day] = model
        model_meta[day] = {
            "name": model_name,
            "version": meta.version,
            "metrics": extract_metrics(meta),
        }

    state["project"] = project
    state["feature_store"] = feature_store
    state["model_registry"] = model_registry
    state["models"] = models
    state["model_meta"] = model_meta
    state["model_features"] = list(models["day1"].feature_names_in_)
    state["cache"] = {"latest_date": None, "result": None}
    logger.info("Models loaded and cached successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_models()
    except Exception as e:
        logger.error(f"Startup model load failed: {e}")
    yield
    state.clear()


app = FastAPI(
    title="Lahore AQI Prediction API",
    description="Predicts Lahore AQI for the next 3 days",
    version="2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DayPrediction(BaseModel):
    date: str
    aqi: float
    category: str
    color: str
    hazardous_alert: bool
    advice: str

class PredictionResponse(BaseModel):
    latest_available_date: str
    pipeline_updated: bool
    current_conditions: dict
    current_aqi: dict
    predictions: dict[str, DayPrediction]

@app.get("/")
def home():
    return {"message": "Lahore AQI Prediction API is running"}

@app.post("/refresh-models")
def refresh_models():
    """Manually reload models/feature store without restarting the process."""
    try:
        load_models()
        return {"status": "models reloaded"}
    except Exception as e:
        logger.exception("Model refresh failed")
        raise HTTPException(status_code=500, detail=f"Model refresh failed: {e}")

@app.get("/models")
def get_model_info():
    """Model names, versions, and evaluation metrics (R2, RMSE, MAE)."""
    if not state["model_meta"]:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")
    return {"models": state["model_meta"]}

def _load_daily_data():
    try:
        daily_feature_group = state["feature_store"].get_feature_group(
            name="aqi_daily", version=1
        )
    except Exception as e:
        logger.exception("Failed to fetch feature group")
        raise HTTPException(status_code=502, detail=f"Feature store error: {e}")

    if daily_feature_group is None:
        raise HTTPException(status_code=404, detail="Daily feature group not found")

    try:
        daily_data = daily_feature_group.read()
    except Exception as e:
        logger.exception("Failed to read feature group data")
        raise HTTPException(status_code=502, detail=f"Failed to read feature data: {e}")

    if daily_data is None or daily_data.empty:
        raise HTTPException(status_code=404, detail="No data available in feature group")

    daily_data["date"] = pd.to_datetime(daily_data["date"])
    daily_data = daily_data.sort_values("date").reset_index(drop=True)
    daily_data = daily_data.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return daily_data

@app.get("/history")
def get_history(days: int = 30):
    """Last N days of actual AQI + pollutant readings, for trend charts."""
    daily_data = _load_daily_data()
    recent = daily_data.tail(days).copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")

    cols = ["date", "daily_aqi"] + [c for c in POLLUTANT_COLUMNS if c in recent.columns]
    return {"history": recent[cols].to_dict(orient="records")}

@app.get("/predict", response_model=PredictionResponse)
def predict_aqi():
    if not state["models"]:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded yet. Try /refresh-models or check server logs.",
        )

    daily_data = _load_daily_data()
    latest_row = daily_data.tail(1).copy()
    latest_date = latest_row["date"].iloc[0]
    latest_date_str = latest_date.strftime("%Y-%m-%d")

    cache = state["cache"]
    if cache["latest_date"] == latest_date_str and cache["result"] is not None:
        result = dict(cache["result"])
        result["pipeline_updated"] = False
        return result

    prediction_input = latest_row.drop(columns=["date"] + TARGET_COLUMNS, errors="ignore")
    prediction_input = prediction_input.apply(pd.to_numeric, errors="coerce").fillna(0)
    prediction_input = prediction_input.reindex(columns=state["model_features"]).fillna(0)

    try:
        pred_day1 = float(state["models"]["day1"].predict(prediction_input)[0])
        pred_day2 = float(state["models"]["day2"].predict(prediction_input)[0])
        pred_day3 = float(state["models"]["day3"].predict(prediction_input)[0])
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    current_conditions = {
        col: (float(latest_row[col].iloc[0]) if col in latest_row.columns else None)
        for col in WEATHER_COLUMNS + POLLUTANT_COLUMNS
    }
    current_aqi_value = float(latest_row["daily_aqi"].iloc[0]) if "daily_aqi" in latest_row.columns else None

    result = {
        "latest_available_date": latest_date_str,
        "pipeline_updated": True,
        "current_conditions": current_conditions,
        "current_aqi": {
            "aqi": round(current_aqi_value, 2) if current_aqi_value is not None else None,
            **classify_aqi(current_aqi_value or 0),
        },
        "predictions": {
            "day_1": {
                "date": (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "aqi": round(pred_day1, 2),
                **classify_aqi(pred_day1),
            },
            "day_2": {
                "date": (latest_date + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                "aqi": round(pred_day2, 2),
                **classify_aqi(pred_day2),
            },
            "day_3": {
                "date": (latest_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                "aqi": round(pred_day3, 2),
                **classify_aqi(pred_day3),
            },
        },
    }

    state["cache"] = {"latest_date": latest_date_str, "result": result}
    return result






