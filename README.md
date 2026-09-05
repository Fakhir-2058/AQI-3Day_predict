# 🌫️ Lahore Air Quality Intelligence


A fully automated, self-updating 3-day Air Quality Index forecasting system for Lahore, Pakistan.

**Live Dashboard:** [https://pearls-lahore-aqi-prediction.streamlit.app/](https://pearls-lahore-aqi-prediction.streamlit.app/)  
**Full project details, architecture, and design rationale:** see [Lahore_AQI_Intelligence_Report.docx](./Lahore_AQI_Intelligence_Report.docx)

## About the Project

This project predicts Lahore's Air Quality Index (AQI) 3 days in advance, using live weather and pollution data. It runs completely on its own — collecting data, retraining models, and updating the dashboard automatically, every hour, with no manual work required.

In simple terms: it watches the air quality every hour, learns from the pattern, and tells you what the air will likely be like for the next 3 days — along with a plain-language explanation of why.

## How It Works

```
Live Weather/AQI Data → Feature Engineering → Model Training → Prediction → Dashboard
      (hourly)              (daily)             (daily)         (hourly)     (live)
```

1. **Collect** — Hourly weather & pollution data is pulled from Open-Meteo for Lahore.
2. **Engineer** — Once a day, hourly data is aggregated into daily trends (lag values, rolling averages, etc.).
3. **Train** — 5 ML algorithms are benchmarked; the best one is auto-selected per forecast day.
4. **Predict** — Every hour, the latest model generates a fresh 3-day AQI forecast.
5. **Explain** — SHAP values show which factors most influenced each prediction.
6. **Display** — Results are served via a live, auto-refreshing dashboard.

## What the Dashboard Shows

- Current AQI and health advice
- 3-day forecast cards, color-coded by severity
- 24-hour trend chart
- Pollutant and weather breakdown
- SHAP-based explanation of each forecast

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Data Source | Open-Meteo API | Free, no key required, reliable historical + live data |
| Feature Store / Model Registry | Hopsworks | Built-in versioning & upsert, avoids duplicate data |
| Machine Learning | Scikit-learn + XGBoost | Best fit for small tabular time-series data |
| Explainability | SHAP | Works across all model types used |
| Backend API | FastAPI | Lightweight, fast, auto-documented |
| Dashboard | Streamlit + Plotly | Fast interactive UI, pure Python |
| Automation | GitHub Actions | Free, no server needed, runs on schedule |
| Hosting | Vercel (API) + Streamlit Cloud (dashboard) | Free, git-integrated deployment |

## Project Structure

```
├── hourly_feature_script.py     # Fetches new hourly weather & AQI data
├── daily_feature_script.py      # Builds daily features for training
├── training_script.py           # Trains models & registers the best one
├── prediction_script.py         # Generates 3-day forecast + SHAP explanations
├── api/
│   └── index.py                 # FastAPI backend serving the dashboard data
├── streamlit_app.py             # Live dashboard (frontend)
├── data/
│   └── dashboard.json           # Auto-generated forecast data
├── .github/workflows/
│   └── aqi_pipeline.yml         # Automation schedule (hourly + daily)
├── requirements.txt
└── requirements-automation.txt
```

## Automation Schedule

| Frequency | What Runs |
|---|---|
| Every hour | Fetch new data → generate fresh predictions |
| Once daily (overnight) | Rebuild features → retrain models → predict |
