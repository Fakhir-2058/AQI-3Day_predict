# 🌫️ Lahore Air Quality Intelligence

An automated system that predicts Lahore's Air Quality Index (AQI) for the next 3 days using live weather and pollution data, machine learning, and SHAP-based explainability — served through a FastAPI backend and displayed on a live Streamlit dashboard.

**Full project details, architecture, and design rationale:** see [`PROJECT_REPORT.md`](./PROJECT_REPORT.md)

## Tech Stack
Python · pandas/numpy · scikit-learn · XGBoost · SHAP · Hopsworks · FastAPI · Streamlit · Plotly · GitHub Actions · Vercel

## Project Structure
```
├── hourly_feature_script.py     # Hourly data ingestion
├── daily_feature_script.py      # Daily feature engineering
├── training_script.py           # Model training & registration
├── prediction_script.py         # Forecast + SHAP + dashboard.json
├── api/index.py                 # FastAPI backend
├── streamlit_app.py             # Streamlit dashboard
├── data/dashboard.json          # Generated forecast data
└── .github/workflow/aqi_pipeline.yml  # Automation pipeline
```

## Quick Start
```bash
pip install -r requirements-automation.txt
export HOPSWORKS_API_KEY="your_api_key_here"

python hourly_feature_script.py
python daily_feature_script.py
python training_script.py
python prediction_script.py

uvicorn api.index:app --reload      # API
streamlit run streamlit_app.py      # Dashboard
```

## License
Add your preferred license here (e.g., MIT).
