import pandas as pd
import numpy as np
import hopsworks


project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)

print("Connected to Hopsworks!")

feature_store = project.get_feature_store()

print("Feature Store loaded!")


hourly_feature_group = feature_store.get_feature_group(
    name="aqi_hourly",
    version=1
)


if hourly_feature_group is None:
    raise Exception(
        "Feature Group 'aqi_hourly' version 1 was not found."
    )

print("Hourly Feature Group loaded!")


hourly_data = hourly_feature_group.read()

print("Hourly data loaded!")
print("Hourly data shape:", hourly_data.shape)

print(hourly_data.tail())



hourly_data["time"] = pd.to_datetime(hourly_data["time"])

hourly_data = hourly_data.sort_values("time").reset_index(drop=True)
hourly_data = hourly_data.drop_duplicates(subset=["time"])

print("Hourly data cleaned!")
print("Cleaned shape:", hourly_data.shape)

hourly_data["date"] = hourly_data["time"].dt.date


daily_data = hourly_data.groupby("date").agg({

    "us_aqi": "mean",
    "pm2_5": "mean",
    "pm10": "mean",
    "temperature": "mean",
    "humidity": "mean",
    "pressure": "mean",
    "wind_speed": "mean",
    "co": "mean",
    "no2": "mean",
    "so2": "mean",
    "ozone": "mean"

}).reset_index()


daily_data = daily_data.rename(
    columns={
        "us_aqi": "daily_aqi"
    }
)


daily_data["date"] = pd.to_datetime(daily_data["date"])

daily_data["day"] = daily_data["date"].dt.day
daily_data["month"] = daily_data["date"].dt.month
daily_data["weekday"] = daily_data["date"].dt.weekday

daily_data["aqi_change_rate"] = (daily_data["daily_aqi"].pct_change())

daily_data["aqi_lag_1d"] = (daily_data["daily_aqi"].shift(1))
daily_data["aqi_lag_2d"] = (daily_data["daily_aqi"].shift(2))
daily_data["aqi_lag_3d"] = (daily_data["daily_aqi"].shift(3))
daily_data["aqi_lag_7d"] = (daily_data["daily_aqi"].shift(7))
daily_data["aqi_lag_14d"] = (daily_data["daily_aqi"].shift(14))
daily_data["aqi_lag_21d"] = (daily_data["daily_aqi"].shift(21))
daily_data["aqi_lag_30d"] = (daily_data["daily_aqi"].shift(30))

daily_data["aqi_rolling_3d"] = (daily_data["daily_aqi"].shift(1).rolling(3).mean())
daily_data["aqi_rolling_7d"] = (daily_data["daily_aqi"].shift(1).rolling(7).mean())
daily_data["aqi_rolling_14d"] = (daily_data["daily_aqi"].shift(1).rolling(14).mean())
daily_data["aqi_rolling_21d"] = (daily_data["daily_aqi"].shift(1).rolling(21).mean())
daily_data["aqi_rolling_30d"] = (daily_data["daily_aqi"].shift(1).rolling(30).mean())

daily_data["pm25_lag_1d"] = (daily_data["pm2_5"].shift(1))
daily_data["pm25_lag_3d"] = (daily_data["pm2_5"].shift(3))
daily_data["pm25_lag_7d"] = (daily_data["pm2_5"].shift(7))

daily_data["pm25_rolling_3d"] = (daily_data["pm2_5"].shift(1).rolling(3).mean())
daily_data["pm25_rolling_7d"] = (daily_data["pm2_5"].shift(1).rolling(7).mean())
daily_data["pm25_rolling_14d"] = (daily_data["pm2_5"].shift(1).rolling(14).mean())


daily_data["temp_change"] = (daily_data["temperature"].diff())
daily_data["humidity_change"] = (daily_data["humidity"].diff())
daily_data["pressure_change"] = (daily_data["pressure"].diff())
daily_data["wind_change"] = (daily_data["wind_speed"].diff())

daily_data["temperature_rolling_3d"] = (daily_data["temperature"].shift(1).rolling(3).mean())
daily_data["temperature_rolling_7d"] = (daily_data["temperature"].shift(1).rolling(7).mean())


daily_data = daily_data.dropna().reset_index(drop=True)

print("\nFeature engineering completed!")
print(daily_data.tail())



latest_prediction_data = daily_data.tail(1).copy()
print("\nLATEST DATA USED FOR PREDICTION:")

print(latest_prediction_data)


daily_feature_group = feature_store.get_feature_group(
    name="aqi_daily",
    version=1
)


if daily_feature_group is None:
    raise Exception(
        "Feature Group 'aqi_daily' version 1 was not found."
    )


print("\nDaily Feature Group loaded!")


training_df = daily_feature_group.read()

print("Training feature data loaded!")

print("Training data shape:", training_df.shape)



target_columns = [

    "target_aqi_day1",
    "target_aqi_day2",
    "target_aqi_day3"

]


feature_columns = training_df.drop(columns=target_columns).columns.tolist()

if "date" in feature_columns:
    feature_columns.remove(
        "date"
    )


print("\nExpected model features:")

print(feature_columns)

print("Number of features:", len(feature_columns))


prediction_input = latest_prediction_data.copy()

prediction_input = prediction_input.drop(
    columns=["date"],
    errors="ignore"
)


prediction_input = prediction_input.reindex(
    columns=feature_columns
)


prediction_input = prediction_input.fillna(0)


print("\nPREDICTION INPUT:")

print(prediction_input)

print("Prediction input shape:", prediction_input.shape)



# ============================================================
# CHECK FEATURE COUNT
# ============================================================

if len(prediction_input.columns) != len(feature_columns):

    raise Exception(
        "Prediction features do not match training features."
    )


print("\nFeature structure matched successfully!")



# ============================================================
# LOAD MODEL REGISTRY
# ============================================================

model_registry = project.get_model_registry()


print("\nModel Registry loaded!")



# ============================================================
# GET DAY 1 MODEL
# ============================================================

day1_model = model_registry.get_model(
    name="lahore_aqi_day1",
    version=None
)


print("Day 1 model loaded!")


# ============================================================
# GET DAY 2 MODEL
# ============================================================

day2_model = model_registry.get_model(
    name="lahore_aqi_day2",
    version=None
)


print("Day 2 model loaded!")



# ============================================================
# GET DAY 3 MODEL
# ============================================================

day3_model = model_registry.get_model(
    name="lahore_aqi_day3",
    version=None
)


print("Day 3 model loaded!")



# ============================================================
# DOWNLOAD MODELS
# ============================================================

day1_model_dir = day1_model.download()
day2_model_dir = day2_model.download()
day3_model_dir = day3_model.download()


print("\nModels downloaded successfully!")



# ============================================================
# LOAD MODELS
# ============================================================

import joblib
import os


day1_model_path = os.path.join(
    day1_model_dir,
    "lahore_aqi_day1.pkl"
)


day2_model_path = os.path.join(
    day2_model_dir,
    "lahore_aqi_day2.pkl"
)


day3_model_path = os.path.join(
    day3_model_dir,
    "lahore_aqi_day3.pkl"
)


model_day1 = joblib.load(
    day1_model_path
)


model_day2 = joblib.load(
    day2_model_path
)


model_day3 = joblib.load(
    day3_model_path
)


print("All models loaded successfully!")



# ============================================================
# MAKE PREDICTIONS
# ============================================================

prediction_day1 = model_day1.predict(
    prediction_input
)[0]


prediction_day2 = model_day2.predict(
    prediction_input
)[0]


prediction_day3 = model_day3.predict(
    prediction_input
)[0]



# ============================================================
# FINAL RESULTS
# ============================================================

latest_date = latest_prediction_data[
    "date"
].iloc[0]


prediction_date_day1 = (
    latest_date + pd.Timedelta(days=1)
)


prediction_date_day2 = (
    latest_date + pd.Timedelta(days=2)
)


prediction_date_day3 = (
    latest_date + pd.Timedelta(days=3)
)


print("\n")

print("=" * 60)

print("LIVE LAHORE AQI PREDICTION")

print("=" * 60)


print(
    "\nLatest available date:",
    latest_date.strftime("%Y-%m-%d")
)


print(
    "\nPrediction for Day 1:",
    prediction_date_day1.strftime("%Y-%m-%d"),
    "→",
    round(prediction_day1, 2)
)


print(
    "Prediction for Day 2:",
    prediction_date_day2.strftime("%Y-%m-%d"),
    "→",
    round(prediction_day2, 2)
)


print(
    "Prediction for Day 3:",
    prediction_date_day3.strftime("%Y-%m-%d"),
    "→",
    round(prediction_day3, 2)
)


print("\nPrediction completed successfully!")

print("=" * 60)