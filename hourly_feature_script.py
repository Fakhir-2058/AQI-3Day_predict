import requests
import pandas as pd
import hopsworks
import os

def get_geocodeinfo(name):

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={name}&count=10&language=en&format=json"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return response.json()


def get_AQI_info(lat, lon, start_date, end_date):

    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi,us_aqi"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    return response.json()


def get_weatherinfo(lat, lon, start_date, end_date):

    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    return response.json()

project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"]
)

print("Connected to Hopsworks!")

feature_store = project.get_feature_store()

print("Feature Store loaded!")



#  LOAD HOURLY FEATURE GROUP


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
print("Latest hour already in Feature Group:",latest_existing_time)

start_date = (latest_existing_time- pd.Timedelta(days=3)).strftime("%Y-%m-%d")
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")


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



# today = datetime.now().date()

# start_date = today - timedelta(days=1)
# end_date = today

# print("Fetching data from:")
# print("Start date:", start_date)
# print("End date:", end_date)


weather = get_weatherinfo(latitude,longitude,start_date,end_date)
aqi = get_AQI_info(latitude,longitude,start_date,end_date)

# print("\nAQI data downloaded successfully!")
# print("Weather data downloaded successfully!")


aqi_hourly = aqi["hourly"]
weather_hourly = weather["hourly"]

times = weather_hourly["time"]

combined_data = []


for i in range(len(times)):

    row = {
        "time": times[i],

        "temperature": weather_hourly["temperature_2m"][i],
        "humidity": weather_hourly["relative_humidity_2m"][i],
        "pressure": weather_hourly["surface_pressure"][i],
        "wind_speed": weather_hourly["wind_speed_10m"][i],
        "pm2_5": aqi_hourly["pm2_5"][i],
        "pm10": aqi_hourly["pm10"][i],
        "co": aqi_hourly["carbon_monoxide"][i],
        "no2": aqi_hourly["nitrogen_dioxide"][i],
        "so2": aqi_hourly["sulphur_dioxide"][i],
        "ozone": aqi_hourly["ozone"][i],
        "european_aqi": aqi_hourly["european_aqi"][i],
        "us_aqi": aqi_hourly["us_aqi"][i]
    }

    combined_data.append(row)



hourly_data = pd.DataFrame(combined_data)

hourly_data["time"] = pd.to_datetime(hourly_data["time"])

hourly_data = hourly_data.drop_duplicates(subset=["time"])

hourly_data = hourly_data.sort_values("time").reset_index(drop=True)


print("HOURLY DATA CREATED")

print(hourly_data.head())
print(hourly_data.tail())

print("\nHourly data shape:", hourly_data.shape)


# print("Connecting to Hopsworks...")

# project = hopsworks.login(
#     api_key_value="bdnP4kgJVin2ySbl.2NCORdoWh3Olg6ehDusL6WbJMcwmg1dQBT4owqLC0AeKalMNozAmNEDcFpFsIWoW"
# )
# print("Connected to Hopsworks!")

# feature_store = project.get_feature_store()

# print("Feature Store loaded!")




# feature_group = feature_store.get_or_create_feature_group(

#     name="aqi_hourly",
#     version=1,
#     primary_key=["time"],
#     description=("Hourly Lahore AQI, pollutant and weather data automatically collected from Open-Meteo")
# )

# print("Hourly Feature Group loaded!")


# print("\nChecking existing hourly records...")

# try:
#     existing_df = feature_group.read()
#     print("Existing records:", len(existing_df))

# except Exception as e:
#     print("No existing data found.")
#     print("This is the first run.")

#     existing_df = pd.DataFrame()


# if not existing_df.empty:

#     existing_df["time"] = pd.to_datetime(existing_df["time"])
#     existing_times = set(existing_df["time"])
#     new_data = hourly_data[~hourly_data["time"].isin(existing_times)].copy()

# else:
#     new_data = hourly_data.copy()



# new_data = new_data.drop_duplicates(subset=["time"])

existing_times = set(
    existing_df["time"]
)

new_data = hourly_data[~hourly_data["time"].isin(existing_times)].copy()


print("New records found:", len(new_data))

if len(new_data) > 0:

    print("Inserting new hourly data...")

    feature_group.insert(new_data,
        write_options={
            "wait_for_job": True
        }
    )
    print("NEW HOURLY DATA INSERTED SUCCESSFULLY!")

else:
    print("NO NEW DATA TO INSERT")
  


print("\nPipeline finished successfully!")