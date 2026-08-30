
import pandas as pd
import numpy as np
import hopsworks
import os
# Hopswork connection
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

# Horly Data reading from aqi_houly feature group
hourly_data = hourly_feature_group.read()
print("Hourly data loaded!")
print(hourly_data.head())


hourly_data["time"] = pd.to_datetime(hourly_data["time"])
# cleaning duplicate rows from horly data

hourly_data = hourly_data.sort_values("time").reset_index(drop=True)
hourly_data = hourly_data.drop_duplicates(subset=["time"])

print("Hourly data cleaned!")
print("Cleaned shape:", hourly_data.shape)

hourly_data["date"] = hourly_data["time"].dt.date

# Creating Daily data 
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

# using US AQI 
daily_data = daily_data.rename(
    columns={
        "us_aqi": "daily_aqi"
    })

# Feature engineering on daily data
daily_data["date"] = pd.to_datetime(daily_data["date"])

# Time Based Features
daily_data["day"] = daily_data["date"].dt.day
daily_data["month"] = daily_data["date"].dt.month
daily_data["weekday"] = daily_data["date"].dt.weekday

daily_data["aqi_change_rate"] = (daily_data["daily_aqi"].pct_change())

# AQI Lag Features
daily_data["aqi_lag_1d"] = (daily_data["daily_aqi"].shift(1))
daily_data["aqi_lag_2d"] = (daily_data["daily_aqi"].shift(2))
daily_data["aqi_lag_3d"] = (daily_data["daily_aqi"].shift(3))
daily_data["aqi_lag_7d"] = (daily_data["daily_aqi"].shift(7))
daily_data["aqi_lag_14d"] = (daily_data["daily_aqi"].shift(14))
daily_data["aqi_lag_21d"] = (daily_data["daily_aqi"].shift(21))
daily_data["aqi_lag_30d"] = (daily_data["daily_aqi"].shift(30))

# AQI Rolling Features
daily_data["aqi_rolling_3d"] = (daily_data["daily_aqi"].shift(1).rolling(3).mean())
daily_data["aqi_rolling_7d"] = (daily_data["daily_aqi"].shift(1).rolling(7).mean())
daily_data["aqi_rolling_14d"] = (daily_data["daily_aqi"].shift(1).rolling(14).mean())
daily_data["aqi_rolling_21d"] = (daily_data["daily_aqi"].shift(1).rolling(21).mean())
daily_data["aqi_rolling_30d"] = (daily_data["daily_aqi"].shift(1).rolling(30).mean())

# Pm_25 Lag Features
daily_data["pm25_lag_1d"] = (daily_data["pm2_5"].shift(1))
daily_data["pm25_lag_3d"] = (daily_data["pm2_5"].shift(3))
daily_data["pm25_lag_7d"] = (daily_data["pm2_5"].shift(7))

# Pm_25 Rolling Features
daily_data["pm25_rolling_3d"] = (daily_data["pm2_5"].shift(1).rolling(3).mean())
daily_data["pm25_rolling_7d"] = (daily_data["pm2_5"].shift(1).rolling(7).mean())
daily_data["pm25_rolling_14d"] = (daily_data["pm2_5"].shift(1).rolling(14).mean())

# Wheather chnage rate features
daily_data["temp_change"] = (daily_data["temperature"].diff())
daily_data["humidity_change"] = (daily_data["humidity"].diff())
daily_data["pressure_change"] = (daily_data["pressure"].diff())
daily_data["wind_change"] = (daily_data["wind_speed"].diff())

# Temprature Rolling features
daily_data["temperature_rolling_3d"] = (daily_data["temperature"].shift(1).rolling(3).mean())
daily_data["temperature_rolling_7d"] = (daily_data["temperature"].shift(1).rolling(7).mean())

# Day1, Day2, Day3 Targets Features
daily_data["target_aqi_day1"] = (daily_data["daily_aqi"].shift(-1))
daily_data["target_aqi_day2"] = (daily_data["daily_aqi"].shift(-2))
daily_data["target_aqi_day3"] = (daily_data["daily_aqi"].shift(-3))

feature_columns = [
    "daily_aqi",
    "pm2_5",
    "pm10",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "co",
    "no2",
    "so2",
    "ozone",
    "day",
    "month",
    "weekday",
    "aqi_change_rate",
    "aqi_lag_1d",
    "aqi_lag_2d",
    "aqi_lag_3d",
    "aqi_lag_7d",
    "aqi_lag_14d",
    "aqi_lag_21d",
    "aqi_lag_30d",
    "aqi_rolling_3d",
    "aqi_rolling_7d",
    "aqi_rolling_14d",
    "aqi_rolling_21d",
    "aqi_rolling_30d",
    "pm25_lag_1d",
    "pm25_lag_3d",
    "pm25_lag_7d",
    "pm25_rolling_3d",
    "pm25_rolling_7d",
    "pm25_rolling_14d",
    "temp_change",
    "humidity_change",
    "pressure_change",
    "wind_change",
    "temperature_rolling_3d",
    "temperature_rolling_7d"
]

# 
daily_data = daily_data.dropna(
    subset=feature_columns
).reset_index(drop=True)


daily_data = daily_data.drop_duplicates(
    subset=["date"]
).reset_index(drop=True)

print("Feature engineered data:")
print(daily_data.tail())

print("\nLATEST FEATURE ENGINEERED ROWS:")

print(
    daily_data[
        [
            "date",
            "daily_aqi",
            "target_aqi_day1",
            "target_aqi_day2",
            "target_aqi_day3"
        ]
    ].tail(10)
)

daily_feature_group = feature_store.get_feature_group(
    name="aqi_daily",
    version=1
)

if daily_feature_group is None:

    raise Exception("Feature Group 'aqi_daily' version 1 was not found.")

print("Daily Feature Group loaded!")


existing_daily_df = daily_feature_group.read()
print("Existing daily data loaded!")

print("Existing daily shape:",existing_daily_df.shape)

existing_daily_df["date"] = pd.to_datetime(
    existing_daily_df["date"]
)

latest_existing_date = existing_daily_df["date"].max()

update_start_date = (
    latest_existing_date - pd.Timedelta(days=7)
)

data_to_update = daily_data[
    daily_data["date"] >= update_start_date
].copy()


print("\nRecent records selected for update:")

print(
    data_to_update[
        [
            "date",
            "daily_aqi",
            "target_aqi_day1",
            "target_aqi_day2",
            "target_aqi_day3"
        ]
    ]
)


print(
    "\nNumber of records to update/insert:",
    len(data_to_update)
)


print(
    "Latest date already in Daily Feature Group:",
    latest_existing_date
)

if len(data_to_update) == 0:

    print(
        "\nNo daily records found to update."
    )

else:

    print(
        "\nUpdating Daily Feature Group..."
    )

    daily_feature_group.insert(data_to_update,
        write_options={
            "wait_for_job": True,
            "operation": "upsert"
        })

    print(
        "\nDaily Feature Group updated successfully!"
    )


print("\nPipeline finished successfully!")
