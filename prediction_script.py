import pandas as pd
import numpy as np
import hopsworks
import os
import joblib

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


print(
    "\nLatest rows from Daily Feature Group:"
)

print(
    daily_data[
        [
            "date",
            "daily_aqi",
            "target_aqi_day1",
            "target_aqi_day2",
            "target_aqi_day3"
        ]
    ].tail()
)

target_columns = [
    "target_aqi_day1",
    "target_aqi_day2",
    "target_aqi_day3"
]


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


print("\nMODEL INFORMATION")

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


print("\nRaw predictions:")

print("Day 1:",prediction_day1)
print("Day 2:",prediction_day2)
print("Day 3:",prediction_day3)

# FINAL RESULTS


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
