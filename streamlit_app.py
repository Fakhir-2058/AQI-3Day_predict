import streamlit as st
import requests
import pandas as pd
import time

API_URL = st.secrets.get(
    "API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="Lahore AQI Dashboard",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Lahore Air Quality Dashboard")
st.caption("AI-powered 3-day AQI prediction")

# auto refresh
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 300:
    st.session_state.last_refresh = time.time()
    st.rerun()


@st.cache_data(ttl=60)
def get_dashboard():

    response = requests.get(
        f"{API_URL}/dashboard",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


try:
    data = get_dashboard()

except Exception as e:

    st.error("Unable to connect to FastAPI.")

    st.info(
        "Make sure FastAPI is running and API_URL is correct."
    )

    st.stop()


# location

location = data["location"]

st.subheader("📍 Location")

st.write(
    f"**{location['city']}**  "
    f"Latitude: `{location['latitude']}`  "
    f"Longitude: `{location['longitude']}`"
)

st.divider()


# current aqi

current = data["current"]

st.subheader("🌫️ Current AQI")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current AQI",
        current["aqi"]
    )

with col2:
    st.metric(
        "Status",
        current["status"]
    )

with col3:
    st.write("Health Advice")
    st.write(current["health_advice"])


st.divider()


# predictions

st.subheader("🔮 3-Day AQI Prediction")

predictions = data["predictions"]

cols = st.columns(3)

days = [
    ("Day 1", predictions["day_1"]),
    ("Day 2", predictions["day_2"]),
    ("Day 3", predictions["day_3"])
]

for col, (day, prediction) in zip(cols, days):

    with col:

        st.markdown(f"### {day}")

        st.metric(
            prediction["date"],
            prediction["aqi"]
        )

        st.write(
            f"**Status:** {prediction['status']}"
        )

        st.info(
            prediction["health_advice"]
        )


st.divider()


# seven day statistics

st.subheader("📊 Last 7 Days AQI")

history = data["last_7_days"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Minimum",
        history["min"]
    )

with col2:
    st.metric(
        "Maximum",
        history["max"]
    )

with col3:
    st.metric(
        "Average",
        history["average"]
    )


st.divider()


# 24 hour trend

st.subheader("📈 Last 24 Hours Observed AQI")

observed = pd.DataFrame(
    data["last_24_hours"]
)

if not observed.empty:

    observed["time"] = pd.to_datetime(
        observed["time"]
    )

    observed = observed.set_index("time")

    st.line_chart(
        observed["aqi"],
        height=350
    )


st.divider()


# pollutants

st.subheader("🧪 Current Pollutants")

pollutants = data["pollutants"]

cols = st.columns(6)

pollutant_names = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "co": "CO",
    "no2": "NO₂",
    "so2": "SO₂",
    "ozone": "O₃"
}

for col, (key, label) in zip(
    cols,
    pollutant_names.items()
):

    with col:

        st.metric(
            label,
            pollutants[key]
        )


st.divider()


# weather

st.subheader("🌤️ Current Weather")

weather = data["weather"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Temperature",
        f"{weather['temperature']} °C"
    )

with col2:
    st.metric(
        "Humidity",
        f"{weather['humidity']} %"
    )

with col3:
    st.metric(
        "Pressure",
        f"{weather['pressure']} hPa"
    )

with col4:
    st.metric(
        "Wind Speed",
        f"{weather['wind_speed']} km/h"
    )


st.divider()


# shap

st.subheader("🔍 AQI Feature Importance")

st.caption(
    "Top features influencing each prediction."
)

shap_data = data.get("shap", {})

col1, col2, col3 = st.columns(3)

for col, day, title in [
    (col1, "day_1", "Day 1"),
    (col2, "day_2", "Day 2"),
    (col3, "day_3", "Day 3")
]:

    with col:

        st.markdown(f"### {title}")

        values = shap_data.get(day, [])

        if values:

            shap_df = pd.DataFrame(values)

            shap_df = shap_df.set_index(
                "feature"
            )

            st.bar_chart(
                shap_df["importance"]
            )

        else:

            st.warning(
                "SHAP information unavailable."
            )


st.divider()


# models

st.subheader("🤖 Current Prediction Models")

models = data["models"]

model_df = pd.DataFrame([
    {
        "Prediction": "Day 1",
        "Model": models["day_1"]["name"],
        "Version": models["day_1"]["version"]
    },
    {
        "Prediction": "Day 2",
        "Model": models["day_2"]["name"],
        "Version": models["day_2"]["version"]
    },
    {
        "Prediction": "Day 3",
        "Model": models["day_3"]["name"],
        "Version": models["day_3"]["version"]
    }
])

st.dataframe(
    model_df,
    use_container_width=True,
    hide_index=True
)


st.divider()


# refresh

col1, col2 = st.columns([1, 5])

with col1:

    if st.button("🔄 Refresh Data"):

        st.cache_data.clear()
        st.rerun()

with col2:

    st.caption(
        "Dashboard automatically refreshes every 5 minutes."
    )


st.caption(
    f"Latest available data: "
    f"{data['latest_available_date']}"
)
