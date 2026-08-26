import pandas as pd
import numpy as np
import hopsworks
import os

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

print(hourly_data.head())

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

daily_data["target_aqi_day1"] = (daily_data["daily_aqi"].shift(-1))
daily_data["target_aqi_day2"] = (daily_data["daily_aqi"].shift(-2))
daily_data["target_aqi_day3"] = (daily_data["daily_aqi"].shift(-3))


daily_data = (daily_data.dropna().reset_index(drop=True))

daily_data = (daily_data.drop_duplicates().reset_index(drop=True))

print("Feature engineered data:")
print(daily_data.tail())

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


existing_daily_df["date"] = pd.to_datetime(existing_daily_df["date"])


existing_dates = set(existing_daily_df["date"].dt.strftime("%Y-%m-%d"))


daily_data_dates = (daily_data["date"].dt.strftime("%Y-%m-%d"))


new_data = daily_data[~daily_data_dates.isin(existing_dates)].copy()



if len(new_data) == 0:
    print("No new daily records to insert.")

else:
    print("New daily records found!")
    print(new_data)

    print("Number of new daily rows:",len(new_data))

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


    print("\nNEW ROWS THAT WILL BE INSERTED:")
    print(
        new_data[
            [
                "date",
                "daily_aqi",
                "target_aqi_day1",
                "target_aqi_day2",
                "target_aqi_day3"
            ]
        ]
    )
     daily_feature_group.insert(new_data,
        write_options={"wait_for_job": True})

    print("New daily data successfully pushed to Hopsworks!")


print("\nDaily feature pipeline finished successfully!")
