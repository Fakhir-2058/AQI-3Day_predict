import pandas as pd
import numpy as np
import hopsworks
import os
import joblib
import json
from datetime import datetime
import shap

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

    else:
        return "Hazardous"


def get_health_advice(aqi):

    if aqi <= 50:
        return [
            "Air quality is good.",
            "Normal outdoor activities are generally suitable."
        ]

    elif aqi <= 100:
        return [
            "Air quality is acceptable.",
            "Unusually sensitive people should consider reducing prolonged outdoor activity."
        ]

    elif aqi <= 150:
        return [
            "Sensitive groups should reduce prolonged outdoor activity.",
            "Consider limiting strenuous outdoor activities."
        ]

    elif aqi <= 200:
        return [
            "Reduce prolonged or heavy outdoor activity.",
            "Sensitive groups should avoid strenuous outdoor activity."
        ]

    elif aqi <= 300:
        return [
            "Avoid prolonged outdoor activity.",
            "Sensitive groups should remain indoors when possible."
        ]

    else:
        return [
            "Avoid outdoor activity.",
            "Remain indoors and reduce exposure to outdoor air pollution."
        ]



project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)

print("Connected to Hopsworks!")

feature_store = project.get_feature_store()

print("Feature Store loaded!")

daily_feature_group = feature_store.get_feature_group(
    name="aqi_daily",
    version=1
)


if daily_feature_group is None:
    raise Exception(
        "Feature Group 'aqi_daily' version 1 was not found."
    )
print("Daily Feature Group loaded!")



daily_data = daily_feature_group.read()


print("Daily feature data loaded!")
print("Daily feature data shape:", daily_data.shape)


daily_data["date"] = pd.to_datetime(daily_data["date"])


daily_data = daily_data.sort_values("date").reset_index(drop=True)


daily_data = daily_data.drop_duplicates(
    subset=["date"],
    keep="last").reset_index(drop=True)

print("Daily feature data cleaned!")


print("\nLatest rows from Daily Feature Group:")

print(daily_data[
        [
            "date",
            "daily_aqi",
            "target_aqi_day1",
            "target_aqi_day2",
            "target_aqi_day3"
        ]].tail())

target_columns = [
    "target_aqi_day1",
    "target_aqi_day2",
    "target_aqi_day3"
]

hourly_feature_group = feature_store.get_feature_group(
    name="aqi_hourly",
    version=1
)

print("Hourly Feature Group loaded!")

hourly_data = hourly_feature_group.read()

print("Hourly data loaded!")
print("Hourly data shape:", hourly_data.shape)

hourly_data["time"] = pd.to_datetime(hourly_data["time"])

hourly_data = (
    hourly_data
    .sort_values("time")
    .drop_duplicates(subset=["time"], keep="last")
    .reset_index(drop=True)
)

seven_day_data = daily_data.tail(7).copy()
seven_day_min = float(seven_day_data["daily_aqi"].min())
seven_day_max = float(seven_day_data["daily_aqi"].max())
seven_day_average = float(seven_day_data["daily_aqi"].mean())

seven_day_history = []

for _, row in seven_day_data.iterrows():

    seven_day_history.append({
        "date": row["date"].strftime("%Y-%m-%d"),
        "aqi": round(float(row["daily_aqi"]), 2),
        "status": get_aqi_status(float(row["daily_aqi"]))
    })

last_24_hours = hourly_data.tail(24).copy()

observed_24h = []

for _, row in last_24_hours.iterrows():

    observed_24h.append({
        "time": row["time"].strftime("%Y-%m-%d %H:%M"),
        "aqi": (
            None
            if pd.isna(row["us_aqi"])
            else round(float(row["us_aqi"]), 2)
        )
    })

latest_hour = hourly_data.iloc[-1]

pollutants = {
    "pm2_5": (
        None if pd.isna(latest_hour["pm2_5"])
        else round(float(latest_hour["pm2_5"]), 2)
    ),

    "pm10": (
        None if pd.isna(latest_hour["pm10"])
        else round(float(latest_hour["pm10"]), 2)
    ),

    "co": (
        None if pd.isna(latest_hour["co"])
        else round(float(latest_hour["co"]), 2)
    ),

    "no2": (
        None if pd.isna(latest_hour["no2"])
        else round(float(latest_hour["no2"]), 2)
    ),

    "so2": (
        None if pd.isna(latest_hour["so2"])
        else round(float(latest_hour["so2"]), 2)
    ),

    "ozone": (
        None if pd.isna(latest_hour["ozone"])
        else round(float(latest_hour["ozone"]), 2)
    )
}

weather_information = {
    "temperature": (
        None if pd.isna(latest_hour["temperature"])
        else round(float(latest_hour["temperature"]), 2)
    ),

    "humidity": (
        None if pd.isna(latest_hour["humidity"])
        else round(float(latest_hour["humidity"]), 2)
    ),

    "pressure": (
        None if pd.isna(latest_hour["pressure"])
        else round(float(latest_hour["pressure"]), 2)
    ),

    "wind_speed": (
        None if pd.isna(latest_hour["wind_speed"])
        else round(float(latest_hour["wind_speed"]), 2)
    )
}


def get_shap_importance(model, prediction_input):

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(prediction_input)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.array(shap_values)

        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        importance = np.abs(shap_values)

        feature_names = prediction_input.columns.tolist()

        shap_data = []

        for feature, value in zip(
            feature_names,
            importance
        ):

            shap_data.append({
                "feature": feature,
                "importance": round(float(value), 6)
            })

        shap_data = sorted(
            shap_data,
            key=lambda x: x["importance"],
            reverse=True
        )

        return shap_data[:10]

    except Exception as e:

        print(
            "SHAP calculation failed:",
            str(e)
        )

        return []



latest_prediction_data = daily_data.tail(1).copy()
print("\nLATEST DATA USED FOR PREDICTION:")

print(latest_prediction_data)

latest_date = latest_prediction_data["date"].iloc[0]


prediction_input = latest_prediction_data.drop(
    columns=["date"] + target_columns,
    errors="ignore")


prediction_input = prediction_input.apply(
    pd.to_numeric,
    errors="coerce")


if prediction_input.isnull().any().any():

    missing_columns = prediction_input.columns[
        prediction_input.isnull().any()
    ].tolist()

    print("\nMissing values found in:")
    print(missing_columns)


    prediction_input = prediction_input.fillna(0)

print("\nPREDICTION INPUT:")
print(prediction_input)


print("\nPrediction input shape:",prediction_input.shape)


# LOAD MODEL REGISTRY

model_registry = project.get_model_registry()
print("\nModel Registry loaded!")


# FUNCTION TO GET LATEST MODEL VERSION

def get_latest_model(model_name):

    models = model_registry.get_models(name=model_name)
    latest_version = max(model.version
        for model in models)

    latest_model = model_registry.get_model(name=model_name,
        version=latest_version
    )

    print(
        f"{model_name} latest version loaded: {latest_version}"
    )
    return latest_model


# GET LATEST DAY 1, DAY 2, DAY 3 MODELS

day1_model = get_latest_model("lahore_aqi_day1")
day2_model = get_latest_model("lahore_aqi_day2")
day3_model = get_latest_model("lahore_aqi_day3")


# DOWNLOAD MODELS

day1_model_dir = day1_model.download()
day2_model_dir = day2_model.download()
day3_model_dir = day3_model.download()

print("\nModels downloaded successfully!")




# LOAD MODELS

import joblib
import os


day1_model_path = os.path.join(day1_model_dir,
    "lahore_aqi_day1.pkl")

day2_model_path = os.path.join(day2_model_dir,
    "lahore_aqi_day2.pkl")

day3_model_path = os.path.join(day3_model_dir,
    "lahore_aqi_day3.pkl")

model_day1 = joblib.load(day1_model_path)
model_day2 = joblib.load(day2_model_path)
model_day3 = joblib.load(day3_model_path)

print("All models loaded successfully!")

def get_model_algorithm(model):

    return type(model).__name__

model_information = {

    "day_1": {
        "model": get_model_algorithm(model_day1),
        "version": int(day1_model.version),
        "MAE": None,
        "RMSE": None,
        "R2": None
    },

    "day_2": {
        "model": get_model_algorithm(model_day2),
        "version": int(day2_model.version),
        "MAE": None,
        "RMSE": None,
        "R2": None
    },

    "day_3": {
        "model": get_model_algorithm(model_day3),
        "version": int(day3_model.version),
        "MAE": None,
        "RMSE": None,
        "R2": None
    }
}




print("\nDay 1 model:")
print(model_day1)

print("\nDay 2 model:")
print(model_day2)

print("\nDay 3 model:")
print(model_day3)



# MATCH MODEL FEATURE ORDER

if hasattr(model_day1, "feature_names_in_"):

    model_features = list(model_day1.feature_names_in_)

    prediction_input = prediction_input.reindex(
        columns=model_features)

else:
    raise Exception(
        "Model feature names are not available."
    )


# CHECK FOR MISSING VALUES

if prediction_input.isnull().any().any():
    raise Exception(
        "Prediction input still contains missing values."
    )

print(
    "\nFeature structure matched successfully!"
)

# MODEL COMPARISON CHECK


print("\nModel target-related prediction comparison:")

print(model_day1.predict(prediction_input))
print(model_day2.predict(prediction_input))
print(model_day3.predict(prediction_input))


prediction_day1 = model_day1.predict(prediction_input)[0]
prediction_day2 = model_day2.predict(prediction_input)[0]
prediction_day3 = model_day3.predict(prediction_input)[0]

print("\nCalculating SHAP importance...")

shap_day1 = get_shap_importance(
    model_day1,
    prediction_input
)

shap_day2 = get_shap_importance(
    model_day2,
    prediction_input
)

shap_day3 = get_shap_importance(
    model_day3,
    prediction_input
)

print("SHAP Day 1:")
print(shap_day1)

print("\nSHAP Day 2:")
print(shap_day2)

print("\nSHAP Day 3:")
print(shap_day3)

prediction_results = {

    "day_1": {
        "date": prediction_date_day1.strftime("%Y-%m-%d"),
        "aqi": round(float(prediction_day1), 2),
        "status": get_aqi_status(prediction_day1),
        "health_advice": get_health_advice(prediction_day1)
    },

    "day_2": {
        "date": prediction_date_day2.strftime("%Y-%m-%d"),
        "aqi": round(float(prediction_day2), 2),
        "status": get_aqi_status(prediction_day2),
        "health_advice": get_health_advice(prediction_day2)
    },

    "day_3": {
        "date": prediction_date_day3.strftime("%Y-%m-%d"),
        "aqi": round(float(prediction_day3), 2),
        "status": get_aqi_status(prediction_day3),
        "health_advice": get_health_advice(prediction_day3)
    }
}



print("\nRaw predictions:")

print("Day 1:",prediction_day1)
print("Day 2:",prediction_day2)
print("Day 3:",prediction_day3)

# FINAL RESULTS

prediction_date_day1 = latest_date + pd.Timedelta(days=1)
prediction_date_day2 = latest_date + pd.Timedelta(days=2)
prediction_date_day3 = latest_date + pd.Timedelta(days=3)


latest_date = latest_prediction_data["date"].iloc[0]

prediction_date_day1 = (
    latest_date + pd.Timedelta(days=1)
)

prediction_date_day2 = (
    latest_date + pd.Timedelta(days=2)
)

prediction_date_day3 = (
    latest_date + pd.Timedelta(days=3)
)

print("\nLIVE LAHORE AQI PREDICTION")


print("\nLatest available date:",latest_date.strftime("%Y-%m-%d"))


print("\nPrediction for Day 1:",prediction_date_day1.strftime("%Y-%m-%d"),
    ":",
    round(prediction_day1, 2))

print("Prediction for Day 2:",prediction_date_day2.strftime("%Y-%m-%d"),
    ":",
    round(prediction_day2, 2))

print("Prediction for Day 3:",prediction_date_day3.strftime("%Y-%m-%d"),
    ":",
    round(prediction_day3, 2))


print("\nPrediction completed successfully!")



# CREATE DASHBOARD JSON

import json

dashboard_data = {

    "location": {
        "city": "Lahore",
        "latitude": 31.558,
        "longitude": 74.3507
    },

    "last_updated": pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),

    "latest_available_date": latest_date.strftime(
        "%Y-%m-%d"
    ),

    "current_aqi": {
        "value": round(
            float(latest_prediction_data["daily_aqi"].iloc[0]),
            2
        ),
        "status": get_aqi_status(
            float(latest_prediction_data["daily_aqi"].iloc[0])
        )
    },

    "seven_day_summary": {

        "min": round(seven_day_min, 2),

        "max": round(seven_day_max, 2),

        "average": round(seven_day_average, 2),

        "history": seven_day_history
    },

    "observed_24_hours": observed_24h,

    "pollutants": pollutants,

    "weather": weather_information,

    "predictions": prediction_results,

    "models": model_information,

    "shap": {

        "day_1": shap_day1,

        "day_2": shap_day2,

        "day_3": shap_day3
    }
}


os.makedirs("data", exist_ok=True)

with open(
    "data/dashboard.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dashboard_data,
        f,
        indent=2
    )

print("dashboard.json generated successfully!")

print("\nDashboard JSON saved at:")
print(os.path.abspath("data/dashboard.json"))
