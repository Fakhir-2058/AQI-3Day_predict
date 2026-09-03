import os
import json
import joblib
import shap
import hopsworks
import pandas as pd
import numpy as np

# hopsworks
project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"])

fs = project.get_feature_store()
registry = project.get_model_registry()

print("Connected to Hopsworks!")

# daily data
daily_fg = fs.get_feature_group("aqi_daily", version=1)
daily_data = daily_fg.read()

daily_data["date"] = pd.to_datetime(daily_data["date"])
daily_data = (
    daily_data.sort_values("date")
    .drop_duplicates("date", keep="last")
    .reset_index(drop=True)
)

print("Daily data:", daily_data.shape)

# hourly data
hourly_fg = fs.get_feature_group("aqi_hourly", version=1)
hourly_data = hourly_fg.read()

hourly_data["time"] = pd.to_datetime(hourly_data["time"])
hourly_data = (
    hourly_data.sort_values("time")
    .drop_duplicates("time", keep="last")
    .reset_index(drop=True)
)

print("Hourly data:", hourly_data.shape)

# 7-day history
last_date = daily_data["date"].max()

last_7_days = daily_data[
    daily_data["date"] >= last_date - pd.Timedelta(days=6)
]

history_7_days = {
    "min": round(float(last_7_days["daily_aqi"].min()), 2),
    "max": round(float(last_7_days["daily_aqi"].max()), 2),
    "average": round(float(last_7_days["daily_aqi"].mean()), 2)
}

# prediction input
latest = daily_data.tail(1).copy()

targets = [
    "target_aqi_day1",
    "target_aqi_day2",
    "target_aqi_day3"
]

X = latest.drop(
    columns=["date"] + targets,
    errors="ignore"
)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# load latest models
def load_model(name):
    models = registry.get_models(name=name)
    latest_model = max(models, key=lambda m: m.version)

    model_dir = latest_model.download()
    model_path = os.path.join(model_dir, f"{name}.pkl")

    return joblib.load(model_path), latest_model.version

model_day1, version1 = load_model("lahore_aqi_day1")
model_day2, version2 = load_model("lahore_aqi_day2")
model_day3, version3 = load_model("lahore_aqi_day3")

# match features
model_features = list(model_day1.feature_names_in_)

X = X.reindex(
    columns=model_features,
    fill_value=0
)

background = daily_data.drop(
    columns=["date"] + targets,
    errors="ignore"
)

background = background.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)

background = background.reindex(
    columns=model_features,
    fill_value=0
)

if len(background) > 200:
    background = background.sample(
        200,
        random_state=42
    )

# predictions
prediction_day1 = float(model_day1.predict(X)[0])
prediction_day2 = float(model_day2.predict(X)[0])
prediction_day3 = float(model_day3.predict(X)[0])

# shap
def get_shap(model, X, background):
    try:

        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            importance = np.abs(shap_values[0])

        # Ridge / Linear models
        else:
            # Use historical data as SHAP background
            explainer = shap.LinearExplainer(
                model,
                background
            )

            shap_values = explainer.shap_values(X)

            importance = np.abs(shap_values[0])

        result = [
            {
                "feature": feature,
                "importance": round(float(value), 4)
            }
            for feature, value in zip(X.columns, importance)
        ]

        return sorted(
            result,
            key=lambda x: x["importance"],
            reverse=True
        )[:10]

    except Exception as e:
        print("SHAP error:", e)
        return []


    
shap_day1 = get_shap(model_day1, X, background)
shap_day2 = get_shap(model_day2, X, background)
shap_day3 = get_shap(model_day3, X, background)

# last 24 hours
last_24 = hourly_data.tail(24)

observed_24h = [
    {
        "time": row["time"].strftime("%Y-%m-%d %H:%M"),
        "aqi": round(float(row["us_aqi"]), 2)
    }
    for _, row in last_24.iterrows()
]

# current pollutants and weather
current = hourly_data.tail(1).iloc[0]

pollutants = {
    "pm2_5": round(float(current["pm2_5"]), 2),
    "pm10": round(float(current["pm10"]), 2),
    "co": round(float(current["co"]), 2),
    "no2": round(float(current["no2"]), 2),
    "so2": round(float(current["so2"]), 2),
    "ozone": round(float(current["ozone"]), 2)
}

weather = {
    "temperature": round(float(current["temperature"]), 2),
    "humidity": round(float(current["humidity"]), 2),
    "pressure": round(float(current["pressure"]), 2),
    "wind_speed": round(float(current["wind_speed"]), 2)
}

# aqi status and advice
def get_aqi_status(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"

def get_health_advice(aqi):
    if aqi <= 50:
        return "Air quality is good. Enjoy normal outdoor activities."
    elif aqi <= 100:
        return "Air quality is acceptable. Sensitive people should be cautious."
    elif aqi <= 150:
        return "Sensitive groups should reduce prolonged outdoor activity."
    elif aqi <= 200:
        return "Everyone should reduce prolonged outdoor activity."
    elif aqi <= 300:
        return "Avoid prolonged outdoor activity. Sensitive people should stay indoors."
    return "Avoid outdoor activity and remain indoors when possible."

# prediction dates
day1_date = last_date + pd.Timedelta(days=1)
day2_date = last_date + pd.Timedelta(days=2)
day3_date = last_date + pd.Timedelta(days=3)

current_aqi = float(latest["daily_aqi"].iloc[0])

# dashboard json
dashboard = {
    "location": {
        "city": "Lahore",
        "latitude": 31.558,
        "longitude": 74.3507
    },
    "latest_available_date": last_date.strftime("%Y-%m-%d"),
    "current": {
        "aqi": round(current_aqi, 2),
        "status": get_aqi_status(current_aqi),
        "health_advice": get_health_advice(current_aqi)
    },
    "predictions": {
        "day_1": {
            "date": day1_date.strftime("%Y-%m-%d"),
            "aqi": round(prediction_day1, 2),
            "status": get_aqi_status(prediction_day1),
            "health_advice": get_health_advice(prediction_day1)
        },
        "day_2": {
            "date": day2_date.strftime("%Y-%m-%d"),
            "aqi": round(prediction_day2, 2),
            "status": get_aqi_status(prediction_day2),
            "health_advice": get_health_advice(prediction_day2)
        },
        "day_3": {
            "date": day3_date.strftime("%Y-%m-%d"),
            "aqi": round(prediction_day3, 2),
            "status": get_aqi_status(prediction_day3),
            "health_advice": get_health_advice(prediction_day3)
        }
    },
    "last_7_days": history_7_days,
    "last_24_hours": observed_24h,
    "pollutants": pollutants,
    "weather": weather,
    "models": {
        "day_1": {
            "name": type(model_day1).__name__,
            "version": version1
        },
        "day_2": {
            "name": type(model_day2).__name__,
            "version": version2
        },
        "day_3": {
            "name": type(model_day3).__name__,
            "version": version3
        }
    },
    "shap": {
        "day_1": shap_day1,
        "day_2": shap_day2,
        "day_3": shap_day3
    }
}

os.makedirs("data", exist_ok=True)

with open("data/dashboard.json", "w") as f:
    json.dump(dashboard, f, indent=2)

print("Dashboard JSON created successfully!")
print(
    "Predictions:",
    prediction_day1,
    prediction_day2,
    prediction_day3
)
print(
    "SHAP features:",
    len(shap_day1),
    len(shap_day2),
    len(shap_day3)
)
