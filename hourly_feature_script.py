import requests
import pandas as pd
import hopsworks
import os
import time

def get_geocodeinfo(name):

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={name}&count=10&language=en&format=json"

    for i in range(3):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print("Geocoding error:", e)
            if i < 2:
                time.sleep(5)

    raise Exception("Geocoding API failed")


def get_AQI_info(lat, lon, start_date, end_date):

    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi,us_aqi"

    for i in range(3):
        try:
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print("AQI API error:", e)
            if i < 2:
                time.sleep(5)

    raise Exception("AQI API failed")


def get_weatherinfo(lat, lon, start_date, end_date):

    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"

    for i in range(3):
        try:
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print("Weather API error:", e)
            if i < 2:
                time.sleep(5)

    raise Exception("Weather API failed")


project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)

print("Connected to Hopsworks!")

feature_store = project.get_feature_store()

print("Feature Store loaded!")


feature_group = feature_store.get_feature_group(
    name="aqi_hourly",
    version=1
)

if feature_group is None:
    raise Exception("Feature Group 'aqi_hourly' version 1 was not found.")

print("Hourly Feature Group loaded!")

existing_df = feature_group.read()

print("Existing hourly data loaded!")
print("Existing hourly shape:", existing_df.shape)

existing_df["time"] = pd.to_datetime(existing_df["time"])

existing_df = existing_df.sort_values("time").reset_index(drop=True)

latest_existing_time = existing_df["time"].max()

print(
    "Latest hour already in Feature Group:",
    latest_existing_time
)


# start from the next missing hour
start_time = latest_existing_time + pd.Timedelta(hours=1)

current_time = pd.Timestamp.utcnow().floor("h").tz_localize(None)

if start_time > current_time:
    print("No new data available.")
    print("\nPipeline finished successfully!")
    raise SystemExit(0)

start_date = start_time.strftime("%Y-%m-%d")
end_date = current_time.strftime("%Y-%m-%d")

print("Downloading hourly data from:", start_date)
print("Downloading hourly data until:", end_date)

city_name = "Lahore"

geocode_info = get_geocodeinfo(city_name)

if "results" not in geocode_info:
    raise Exception("City not found!")

latitude = geocode_info["results"][0]["latitude"]
longitude = geocode_info["results"][0]["longitude"]

print("City:", city_name)
print("Latitude:", latitude)
print("Longitude:", longitude)


weather = get_weatherinfo(
    latitude,
    longitude,
    start_date,
    end_date
)

aqi = get_AQI_info(
    latitude,
    longitude,
    start_date,
    end_date
)


aqi_hourly = aqi["hourly"]
weather_hourly = weather["hourly"]


weather_df = pd.DataFrame({
    "time": weather_hourly["time"],
    "temperature": weather_hourly["temperature_2m"],
    "humidity": weather_hourly["relative_humidity_2m"],
    "pressure": weather_hourly["surface_pressure"],
    "wind_speed": weather_hourly["wind_speed_10m"]
})


aqi_df = pd.DataFrame({
    "time": aqi_hourly["time"],
    "pm2_5": aqi_hourly["pm2_5"],
    "pm10": aqi_hourly["pm10"],
    "co": aqi_hourly["carbon_monoxide"],
    "no2": aqi_hourly["nitrogen_dioxide"],
    "so2": aqi_hourly["sulphur_dioxide"],
    "ozone": aqi_hourly["ozone"],
    "european_aqi": aqi_hourly["european_aqi"],
    "us_aqi": aqi_hourly["us_aqi"]
})


weather_df["time"] = pd.to_datetime(weather_df["time"])
aqi_df["time"] = pd.to_datetime(aqi_df["time"])


# combine both datasets using time
hourly_data = pd.merge(
    weather_df,
    aqi_df,
    on="time",
    how="inner"
)


hourly_data = hourly_data[
    (hourly_data["time"] > latest_existing_time) &
    (hourly_data["time"] <= current_time)
].copy()


hourly_data = hourly_data.dropna()
hourly_data = hourly_data.drop_duplicates(subset=["time"])

hourly_data = hourly_data.sort_values("time").reset_index(drop=True)

print("HOURLY DATA CREATED")

print(hourly_data.head())
print(hourly_data.tail())

print("\nHourly data shape:", hourly_data.shape)

print("New records found:",len(hourly_data))

if len(hourly_data) > 0:

    print("Inserting new hourly data...")

    feature_group.insert(
        hourly_data,
        write_options={
            "wait_for_job": True
        }
    )

    print("NEW HOURLY DATA INSERTED SUCCESSFULLY!")

else:

    print("NO NEW DATA TO INSERT")


print("\nPipeline finished successfully!")
