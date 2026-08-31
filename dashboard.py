
import streamlit as st
import requests
import pandas as pd
from datetime import datetime



# Page Configuration

st.set_page_config(
    page_title="Lahore AQI Prediction System",
    page_icon="🌍",
    layout="wide",
)

API_BASE = "http://127.0.0.1:8000"



# Sidebar

with st.sidebar:
    st.header("⚙️ Dashboard Settings")

    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    refresh_minutes = st.slider(
        "Auto-refresh interval (minutes)", min_value=1, max_value=60, value=15
    )

    if auto_refresh:
        # Plain HTML meta-refresh tag — no extra package needed
        st.markdown(
            f'<meta http-equiv="refresh" content="{refresh_minutes * 60}">',
            unsafe_allow_html=True,
        )

    if st.button("🔄 Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    history_days = st.slider("History window (days)", min_value=7, max_value=90, value=30)

    st.divider()
    st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")



# Data Fetching 


@st.cache_data(ttl=60)
def fetch_predictions():
    r = requests.get(f"{API_BASE}/predict", timeout=120)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def fetch_models():
    r = requests.get(f"{API_BASE}/models", timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def fetch_history(days):
    r = requests.get(f"{API_BASE}/history", params={"days": days}, timeout=60)
    r.raise_for_status()
    return r.json()


try:
    data = fetch_predictions()
except Exception:
    st.error("Unable to connect to the FastAPI backend.")
    st.info("Make sure your FastAPI server is running at " + API_BASE)
    st.stop()

try:
    models_data = fetch_models()
except Exception:
    models_data = {"models": {}}

try:
    history_data = fetch_history(history_days)
except Exception:
    history_data = {"history": []}



latest_date = data["latest_available_date"]
pipeline_updated = data.get("pipeline_updated", False)
current_aqi = data.get("current_aqi", {})
current_conditions = data.get("current_conditions", {})

day1 = data["predictions"]["day_1"]
day2 = data["predictions"]["day_2"]
day3 = data["predictions"]["day_3"]



#Title + Pipline


title_col, status_col = st.columns([4, 1])
with title_col:
    st.title("🌍 Lahore AQI Prediction System")
    st.caption("AI-based air quality forecasting dashboard")
with status_col:
    if pipeline_updated:
        st.success("New pipeline data")
    else:
        st.info("Cached data")



# Hazard Alert 

all_days = [
    ("Today", latest_date, current_aqi),
    ("Day 1", day1["date"], day1),
    ("Day 2", day2["date"], day2),
    ("Day 3", day3["date"], day3),
]
hazardous_days = [(label, date) for label, date, d in all_days if d.get("hazardous_alert")]
 
if len(hazardous_days) == len(all_days):
    st.error(
        "⚠️ Hazardous air quality expected today and for the next 3 days "
        f"({latest_date} to {day3['date']}). Limit outdoor exposure and follow health guidance below."
    )
elif hazardous_days:
    day_list = ", ".join(f"{label} ({date})" for label, date in hazardous_days)
    st.error(
        f"⚠️ Hazardous air quality alert for: **{day_list}**. "
        "Limit outdoor exposure and follow health guidance below."
    )
else:
    st.success("✅ No hazardous AQI levels detected in the forecast window.")


# Current condition


st.subheader("Current Conditions")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Date", latest_date)
c2.metric("Current AQI", current_aqi.get("aqi", "N/A"), current_aqi.get("category", ""))
c3.metric(
    "Temperature (°C)",
    f"{current_conditions.get('temperature', 0):.1f}" if current_conditions.get("temperature") is not None else "N/A",
)
c4.metric(
    "Humidity (%)",
    f"{current_conditions.get('humidity', 0):.1f}" if current_conditions.get("humidity") is not None else "N/A",
)
c5.metric(
    "Wind speed (km/h)",
    f"{current_conditions.get('wind_speed', 0):.1f}" if current_conditions.get("wind_speed") is not None else "N/A",
)

with st.expander("Health advice"):
    st.write(current_aqi.get("advice", "No advice available."))



# 3-DAY Predictions

st.subheader("3-Day AQI Prediction")

pred_cols = st.columns(3)
for col, (label, day) in zip(pred_cols, [("Day 1", day1), ("Day 2", day2), ("Day 3", day3)]):
    with col:
        st.markdown(
            f"""
            <div style="border-radius:10px;padding:16px;background-color:{day['color']}22;
                        border:1px solid {day['color']};">
                <div style="font-size:14px;color:gray;">{label} — {day['date']}</div>
                <div style="font-size:32px;font-weight:700;">{day['aqi']}</div>
                <div style="font-size:14px;font-weight:600;">{day['category']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if day.get("hazardous_alert"):
            st.caption(f"⚠️ {day['advice']}")



# Forecast + History Chart 

st.subheader("AQI Trend & Forecast")

history_df = pd.DataFrame(history_data.get("history", []))

forecast_rows = [
    {"date": latest_date, "AQI": current_aqi.get("aqi"), "type": "Historical"},
    {"date": day1["date"], "AQI": day1["aqi"], "type": "Forecast"},
    {"date": day2["date"], "AQI": day2["aqi"], "type": "Forecast"},
    {"date": day3["date"], "AQI": day3["aqi"], "type": "Forecast"},
]

if not history_df.empty:
    hist_rows = [
        {"date": r["date"], "AQI": r["daily_aqi"], "type": "Historical"}
        for _, r in history_df.iterrows()
    ]
    combined = pd.DataFrame(hist_rows[:-1] + forecast_rows)  # drop dup of latest_date from history
else:
    combined = pd.DataFrame(forecast_rows)

chart_df = combined.pivot_table(index="date", columns="type", values="AQI", aggfunc="first")
chart_df = chart_df.reindex(combined["date"].drop_duplicates().tolist())

st.line_chart(chart_df)
st.caption("Dashed appearance isn't available in the native chart — Forecast and Historical are separate lines/columns instead.")


st.subheader("Pollutant Breakdown (latest reading)")

pollutant_labels = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "co": "CO",
    "no2": "NO2",
    "so2": "SO2",
    "ozone": "Ozone",
}
pollutant_vals = {
    pollutant_labels[k]: current_conditions.get(k)
    for k in pollutant_labels
    if current_conditions.get(k) is not None
}

if pollutant_vals:
    pollutant_df = pd.DataFrame(
        {"Pollutant": list(pollutant_vals.keys()), "Concentration": list(pollutant_vals.values())}
    ).set_index("Pollutant")
    st.bar_chart(pollutant_df)
else:
    st.caption("No pollutant data available for the latest reading.")



# Model Performance

st.subheader("Model Performance")

models = models_data.get("models", {})
if models:
    rows = []
    for day_key, label in [("day1", "Day 1"), ("day2", "Day 2"), ("day3", "Day 3")]:
        m = models.get(day_key, {})
        metrics = m.get("metrics", {})
        rows.append(
            {
                "Horizon": label,
                "Model": m.get("name", "N/A"),
                "R²": metrics.get("r2", "N/A"),
                "RMSE": metrics.get("rmse", "N/A"),
                "MAE": metrics.get("mae", "N/A"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("Model metadata not available. Check that /models is returning data.")



# Predictions details and table


st.subheader("Prediction Details")

details_df = pd.DataFrame(
    {
        "Date": [day1["date"], day2["date"], day3["date"]],
        "Predicted AQI": [day1["aqi"], day2["aqi"], day3["aqi"]],
        "Category": [day1["category"], day2["category"], day3["category"]],
        "Hazardous": [day1["hazardous_alert"], day2["hazardous_alert"], day3["hazardous_alert"]],
    }
)

st.dataframe(details_df, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download forecast as CSV",
    data=details_df.to_csv(index=False),
    file_name=f"lahore_aqi_forecast_{latest_date}.csv",
    mime="text/csv",
)

if auto_refresh:
    st.caption(f"Dashboard auto-refreshes every {refresh_minutes} minute(s).")
else:
    st.caption("Auto-refresh is off — use the sidebar button or reload the page manually.")