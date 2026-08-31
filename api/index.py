```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(title="Lahore AQI Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


def load_dashboard():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "dashboard.json"
    )

    try:
        with open(path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Dashboard data not found"
        )


@app.get("/")
def home():
    return {
        "message": "Lahore AQI Prediction API is running"
    }


@app.get("/api/dashboard")
def dashboard():
    return load_dashboard()


@app.get("/api/predictions")
def predictions():
    data = load_dashboard()
    return {
        "latest_available_date": data["latest_available_date"],
        "predictions": data["predictions"]
    }


@app.get("/api/history")
def history():
    data = load_dashboard()
    return {
        "latest_available_date": data["latest_available_date"],
        "last_7_days": data["last_7_days"],
        "last_24_hours": data["last_24_hours"]
    }


@app.get("/api/pollutants")
def pollutants():
    data = load_dashboard()
    return data["pollutants"]


@app.get("/api/weather")
def weather():
    data = load_dashboard()
    return data["weather"]


@app.get("/api/shap")
def shap():
    data = load_dashboard()
    return data["shap"]


@app.get("/api/models")
def models():
    data = load_dashboard()
    return data["models"]
```
