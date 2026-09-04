import os
import time as time_module
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.environ.get("AQI_API_URL", "https://aqi-3-day-predict-mu.vercel.app/").rstrip("/")
REFRESH_SECONDS = int(os.environ.get("AQI_REFRESH_SECONDS", "60"))

STATUS_COLORS = {
    "Good": "#22c55e",
    "Moderate": "#eab308",
    "Unhealthy for Sensitive Groups": "#f59e0b",
    "Unhealthy": "#f43f5e",
    "Very Unhealthy": "#b78cff",
    "Hazardous": "#dc143c",
}

AQI_BANDS = [
    (0, 50, "Good", "#22c55e"),
    (51, 100, "Moderate", "#eab308"),
    (101, 150, "Sensitive", "#f59e0b"),
    (151, 200, "Unhealthy", "#f43f5e"),
    (201, 300, "Very Unhealthy", "#b78cff"),
    (301, 500, "Hazardous", "#dc143c"),
]

PLOTLY_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "doubleClick": "reset",
}

st.set_page_config(
    page_title="Lahore Air Quality Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0B0F17;
    --bg2: #0d131d;
    --card: rgba(21, 29, 42, 0.90);
    --card2: rgba(10, 25, 40, 0.94);
    --blue: #6366f1;
    --blue2: #818cf8;
    --yellow: #4338ca;
    --yellow2: #a5b4fc;
    --text: #e8eef7;
    --muted: #94A3B8;
    --border: rgba(255, 255, 255, 0.07);
}

html, body, [class*="css"] {
    font-family: "Outfit", sans-serif !important;
}

.stApp {
    background:
        radial-gradient(1200px 600px at 50% -12%, rgba(99,102,241,.05), transparent 60%),
        linear-gradient(180deg, #0B0F17 0%, #0d1017 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(6,19,33,.65);
    backdrop-filter: blur(14px);
}

[data-testid="stToolbar"] {
    display: none;
}

.block-container {
    padding-top: 3.75rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero-wrap {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
    padding: 24px 28px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(15,43,65,.94), rgba(9,25,41,.88));
    box-shadow: 0 1px 2px rgba(0,0,0,.4);
}

.kicker {
    letter-spacing: .17em;
    text-transform: uppercase;
    color: var(--blue);
    font-size: 11px;
    font-weight: 800;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 4px 0 7px;
    letter-spacing: -.02em;
}

.hero-sub {
    color: var(--muted);
    font-size: .9rem;
    line-height: 1.65;
}

.hero-sub b {
    color: #c9d9e8;
}

div[data-testid="stButton"] > button {
    background: rgba(99,102,241,.10);
    border: 1px solid rgba(99,102,241,.38);
    color: #c7d2fe;
    border-radius: 999px;
    font-weight: 700;
    padding: 8px 16px;
    transition: background .15s ease, border-color .15s ease, color .15s ease;
}

div[data-testid="stButton"] > button:hover {
    background: rgba(99,102,241,.20);
    border-color: rgba(99,102,241,.60);
    color: #e0e7ff;
}

div[data-testid="stButton"] > button:active {
    background: rgba(99,102,241,.28);
}

div[data-testid="stButton"] > button:focus:not(:active) {
    color: #e0e7ff;
    border-color: rgba(99,102,241,.60);
}

.alert-banner {
    border-radius: 16px;
    padding: 13px 18px;
    margin: 10px 0 18px;
    font-weight: 650;
    border: 1px solid;
}

.kpi-card,
.forecast-card,
.tile,
.map-card,
.info-card {
    background: rgba(19,26,38,.92);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,.35);
}

.kpi-card {
    padding: 18px 19px;
    margin-bottom: 12px;
    min-height: 118px;
}

.kpi-label {
    color: #8fa7bc;
    font-size: 10px;
    letter-spacing: .13em;
    text-transform: uppercase;
    font-weight: 800;
}

.kpi-value {
    font-size: 1.95rem;
    font-weight: 700;
    margin-top: 7px;
    line-height: 1.15;
}

.kpi-hint {
    color: #8eb9d7;
    font-size: .78rem;
    margin-top: 7px;
}

.health-card {
    min-height: 125px;
}

.health-value {
    color: var(--blue);
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 7px;
    line-height: 1.15;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 650;
    margin: 18px 0 4px;
    color: #e8eef7;
}

.section-subtitle {
    color: #728ba1;
    font-size: .78rem;
    margin-bottom: 10px;
}

.forecast-card {
    min-height: 218px;
    padding: 20px;
    border-top: 4px solid var(--blue);
}

.forecast-day {
    color: #91a6ba;
    font-size: 10px;
    letter-spacing: .14em;
    font-weight: 800;
}

.aqi-num {
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1;
    margin-top: 15px;
}

.forecast-status {
    font-weight: 800;
    margin-top: 8px;
}

.advice {
    color: #c4d1dd;
    font-size: .87rem;
    line-height: 1.5;
    margin-top: 12px;
}

.map-card {
    padding: 5px;
    overflow: hidden;
}

.tile {
    padding: 18px;
    margin-bottom: 12px;
}

.tile-row {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    padding: 10px 2px;
    border-bottom: 1px solid rgba(148,163,184,.11);
    font-size: .88rem;
}

.tile-row:last-child {
    border-bottom: 0;
}

.tile-row b {
    text-align: right;
}

.info-card {
    padding: 18px;
    margin-top: 8px;
}

.info-title {
    color: var(--yellow2);
    font-weight: 800;
    font-size: .95rem;
    margin-bottom: 6px;
}

.info-text {
    color: #9db0c1;
    font-size: .82rem;
    line-height: 1.55;
}

[data-testid="stPlotlyChart"] {
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,.06);
    background: rgba(5,16,27,.28);
    padding: 4px;
    /* Let a single vertical finger swipe scroll the page instead of the
       chart/map hijacking it. Panning/zooming the chart itself still works
       with a drag that isn't a plain vertical scroll (e.g. pinch, or a
       deliberate multi-touch gesture on the map). */
    touch-action: pan-y !important;
}

[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container {
    touch-action: pan-y !important;
}

div[data-testid="stVerticalBlock"] > div:has(.forecast-card) {
    min-width: 0;
}

@media (max-width: 900px) {
    .hero-title { font-size: 1.75rem; }
    .block-container { padding-left: .7rem; padding-right: .7rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def aqi_color(status: str) -> str:
    return STATUS_COLORS.get(status, "#6366f1")


def band_color_for_value(value) -> str:
    """Map a raw AQI number onto the AQI_BANDS scale and return its color,
    so KPI tiles (Current AQI, 7-day min/max/avg) always show the color
    that matches where that number actually sits on the AQI scale."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "#94A3B8"
    for lo, hi, _name, color in AQI_BANDS:
        if lo <= v <= hi:
            return color
    return AQI_BANDS[-1][3] if v > AQI_BANDS[-1][1] else AQI_BANDS[0][3]


def plotly_theme(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,16,27,.48)",
        font=dict(color="#dbe7f3", family="Outfit, sans-serif"),
        height=height,
        margin=dict(l=18, r=18, t=48, b=22),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            gridcolor="rgba(148,163,184,.13)",
            zeroline=False,
            color="#9db0c1",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,.13)",
            zeroline=False,
            color="#9db0c1",
        ),
    )
    return fig


def kpi_card(label: str, value, hint: str = "", accent: str = "#6366f1") -> str:
    return (
        f'<div class="kpi-card" style="border-left:5px solid {accent};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{accent}">{value}</div>'
        f'<div class="kpi-hint">{hint}</div>'
        f'</div>'
    )


def status_banner(status: str, advice: str) -> None:
    color = aqi_color(status)
    st.markdown(
        f'<div class="alert-banner" style="background:{color}16;border-color:{color};color:{color}">'
        f'ALERT · {status} — {advice}</div>',
        unsafe_allow_html=True,
    )


def render_live_chip(refresh_seconds: int) -> None:
    widget_id = f"chip_{int(time_module.time() * 1000)}"
    html = f"""
    <div style="display:flex;justify-content:flex-end;align-items:center;height:100%;">
      <div id="{widget_id}" style="
          display:inline-flex;align-items:center;gap:9px;
          background:rgba(34,197,94,.09);
          border:1px solid rgba(34,197,94,.30);
          color:#86efac;border-radius:999px;
          padding:7px 16px;font-weight:700;white-space:nowrap;
          font-family:'Outfit',sans-serif;font-size:14px;">
          <span style="width:9px;height:9px;border-radius:50%;background:#22c55e;
          animation:pulse 1.6s infinite;display:inline-block;"></span>
          <span id="{widget_id}_text">Live · next refresh in {refresh_seconds}s</span>
      </div>
    </div>
    <style>
      @keyframes pulse {{
        70% {{ box-shadow: 0 0 0 10px rgba(34,197,94,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0); }}
      }}
    </style>
    <script>
    (function () {{
        var remaining = {refresh_seconds};
        var el = document.getElementById("{widget_id}_text");
        var timer = setInterval(function () {{
            remaining -= 1;
            if (remaining <= 0) {{
                if (el) {{ el.innerText = "Refreshing…"; }}
                clearInterval(timer);
            }} else if (el) {{
                el.innerText = "Live · next refresh in " + remaining + "s";
            }}
        }}, 1000);
    }})();
    </script>
    """
    components.html(html, height=42)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_dashboard() -> tuple[dict, str]:
    """Fetch the dashboard payload and stamp it with the LOCAL time this
    fetch actually happened. We no longer trust `meta.generated_at` from the
    API for the "Updated" label, since that field was showing a stale/garbage
    date (e.g. 2018) -- this timestamp always reflects the real refresh."""
    response = requests.get(f"{API_URL}/api/dashboard", timeout=20)
    response.raise_for_status()
    fetched_at = datetime.now().strftime("%d %b %Y · %H:%M:%S")
    return response.json(), fetched_at


def gauge_chart(aqi: float, status: str, reference_aqi: float | None = None) -> go.Figure:
    """Enhanced AQI gauge: finer ticks, glow ring, delta vs 7-day average,
    and a compact category legend under the arc."""
    color = aqi_color(status)
    aqi = float(aqi)
    axis_max = 300 if aqi <= 260 else max(320, aqi * 1.15)

    delta_block = None
    if reference_aqi is not None:
        delta_block = {
            "reference": float(reference_aqi),
            "valueformat": ".0f",
            "increasing": {"color": "#f43f5e"},
            "decreasing": {"color": "#22c55e"},
            "font": {"size": 14, "color": "#9db0c1"},
        }

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta" if delta_block else "gauge+number",
            value=aqi,
            delta=delta_block,
            number={
                "suffix": " AQI",
                "font": {"size": 40, "color": color, "family": "Outfit, sans-serif"},
            },
            gauge={
                "axis": {
                    "range": [0, axis_max],
                    "tickcolor": "#6d84a0",
                    "tickfont": {"color": "#93a4bb", "size": 11},
                    "dtick": 50,
                    "ticklen": 6,
                },
                "bar": {"color": color, "thickness": 0.30},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(34,197,94,.32)"},
                    {"range": [50, 100], "color": "rgba(234,179,8,.32)"},
                    {"range": [100, 150], "color": "rgba(245,158,11,.32)"},
                    {"range": [150, 200], "color": "rgba(244,63,94,.38)"},
                    {"range": [200, axis_max], "color": "rgba(183,140,255,.36)"},
                ],
                "threshold": {
                    "line": {"color": "#e2e8f0", "width": 3},
                    "thickness": 0.9,
                    "value": aqi,
                },
            },
            title={
                "text": f"<b>{status}</b>",
                "font": {"size": 18, "color": color},
            },
            domain={"x": [0, 1], "y": [0.08, 1]},
        )
    )

    legend = " · ".join(
        f'<span style="color:{c}">●</span> {name}' for _, _, name, c in AQI_BANDS[:5]
    )
    fig.update_layout(
        annotations=[
            dict(
                x=0.5,
                y=-0.02,
                xref="paper",
                yref="paper",
                text=legend,
                showarrow=False,
                font=dict(size=9.5, color="#7c8fa4"),
            )
        ]
    )
    return plotly_theme(fig, 360)


def trend_chart(hours: list[dict]) -> go.Figure:
    hour_df = pd.DataFrame(hours)
    hour_df["time"] = pd.to_datetime(hour_df["time"])

    fig = go.Figure()
    bands = [
        (0, 50, "rgba(34,197,94,.07)", "Good"),
        (50, 100, "rgba(234,179,8,.07)", "Moderate"),
        (100, 150, "rgba(245,158,11,.08)", "Sensitive"),
        (150, 200, "rgba(244,63,94,.09)", "Unhealthy"),
    ]

    y_max = max(200, float(hour_df["aqi"].max()) * 1.12)

    for low, high, fill, label in bands:
        fig.add_hrect(y0=low, y1=min(high, y_max), fillcolor=fill, line_width=0)
        fig.add_annotation(
            x=hour_df["time"].iloc[0],
            y=(low + min(high, y_max)) / 2,
            text=label,
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color="rgba(200,215,230,.35)"),
        )

    fig.add_trace(
        go.Scatter(
            x=hour_df["time"],
            y=hour_df["aqi"],
            mode="lines",
            line=dict(color="rgba(99,102,241,.28)", width=10, shape="spline"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hour_df["time"],
            y=hour_df["aqi"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=3.5, shape="spline"),
            marker=dict(
                size=7,
                color="#e0f2fe",
                line=dict(color="#0f766e", width=1),
            ),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,.10)",
            name="US AQI",
            hovertemplate="%{x|%H:%M}<br>AQI %{y:.2f}<extra></extra>",
        )
    )

    # Highlight the 24h peak so a viewer can spot the worst hour at a glance.
    peak_idx = hour_df["aqi"].idxmax()
    peak_time = hour_df["time"].iloc[peak_idx]
    peak_val = float(hour_df["aqi"].iloc[peak_idx])
    fig.add_trace(
        go.Scatter(
            x=[peak_time],
            y=[peak_val],
            mode="markers+text",
            marker=dict(size=11, color="#e2e8f0", line=dict(color="#0B0F17", width=2)),
            text=[f"Peak {peak_val:.0f}"],
            textposition="top center",
            textfont=dict(size=11, color="#cbd5e1"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Observed US AQI · Last 24 Hours",
        yaxis_title="AQI",
        yaxis=dict(range=[0, y_max]),
    )
    return plotly_theme(fig, 390)


def shap_chart(items: list[dict], title: str) -> None:
    if not items:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-title">Interpretability unavailable</div>'
            '<div class="info-text">No SHAP values were supplied for this forecast model.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    frame = pd.DataFrame(items)

    if "importance" not in frame.columns or "feature" not in frame.columns:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-title">Interpretability unavailable</div>'
            '<div class="info-text">The API returned SHAP data in an unsupported format.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0)
    frame = frame[frame["importance"].abs() > 1e-12].copy()

    if frame.empty:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-title">SHAP not available for this horizon</div>'
            '<div class="info-text">This model did not provide non-zero SHAP explanations. '
            'The forecast is still valid; only the feature-level explanation is unavailable.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    frame = frame.sort_values("importance", ascending=True)

    fig = px.bar(
        frame,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#312e81", "#6366f1", "#a5b4fc", "#f43f5e"],
        title=title,
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(plotly_theme(fig, 380), use_container_width=True, config=PLOTLY_CONFIG)


def pollutant_chart(pollutants: dict) -> go.Figure:
    frame = pd.DataFrame(
        {
            "pollutant": list(pollutants.keys()),
            "value": list(pollutants.values()),
        }
    )
    fig = px.bar(
        frame,
        x="pollutant",
        y="value",
        color="value",
        color_continuous_scale=["#312e81", "#6366f1", "#a5b4fc", "#f43f5e"],
        title="Current Pollutant Mix",
    )
    fig.update_layout(coloraxis_showscale=False)
    return plotly_theme(fig, 340)


def weather_chart(weather: dict) -> go.Figure:
    labels = [key.replace("_", " ").title() for key in weather]
    values = list(weather.values())

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=["#6366f1", "#818cf8", "#a5b4fc", "#94A3B8"][:len(values)]
            ),
        )
    )
    fig.update_layout(title="Weather Snapshot")
    return plotly_theme(fig, 340)


def map_chart(lat: float, lon: float, status: str) -> go.Figure:
    color = aqi_color(status)

    fig = go.Figure()

    fig.add_trace(
        go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker=dict(size=90, color="rgba(99,102,241,.14)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker=dict(size=24, color=color),
            text=[f"Lahore Monitoring Zone · {status}"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker=dict(size=11, color="#e2e8f0"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=lat, lon=lon),
            zoom=10.3,
        ),
        title="Lahore Monitoring Zone",
    )
    return plotly_theme(fig, 390)


def map_chart_fallback(lat: float, lon: float, status: str) -> go.Figure:
    color = aqi_color(status)
    fig = go.Figure(
        go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker=dict(size=28, color=color),
            text=[f"Lahore Monitoring Zone · {status}"],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=lat, lon=lon),
            zoom=10.3,
        ),
        title="Lahore Monitoring Zone",
    )
    return plotly_theme(fig, 390)


def render(data: dict, fetched_at: str) -> None:
    location = data.get("location") or {}
    current = data.get("current") or {}
    predictions = data.get("predictions") or {}
    history = data.get("last_7_days") or {}
    hours = data.get("last_24_hours") or []
    pollutants = data.get("pollutants") or {}
    weather = data.get("weather") or {}
    models = data.get("models") or {}
    shap = data.get("shap") or {}

    status = current.get("status", "Unknown")
    color = aqi_color(status)
    city = location.get("city", "Lahore")
    lat = location.get("latitude")
    lon = location.get("longitude")

    latest_date = data.get("latest_available_date", "—")

    generated_display = fetched_at

    st.markdown(
        f"""
<div class="hero-wrap">
  <div>
    <div class="kicker">Pakistan · Punjab · Air Quality Monitoring</div>
    <div class="hero-title">Lahore Air Quality Intelligence</div>
    <div class="hero-sub">
      <b>{city} Monitoring Zone</b><br>
      Coordinates: {lat}, {lon}<br>
      Latest data: {latest_date}<br>
      Dashboard updated: {generated_display}
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    status_banner(status, current.get("health_advice", ""))

    g1, g2, g3, g4, g5 = st.columns([1.25, 1, 1, 1, 1])

    with g1:
        if current.get("aqi") is not None:
            st.plotly_chart(
                gauge_chart(current["aqi"], status, history.get("average")),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

    current_aqi_val = current.get("aqi", "—")
    with g2:
        st.markdown(
            kpi_card(
                "Current AQI",
                current_aqi_val,
                status,
                band_color_for_value(current_aqi_val),
            ),
            unsafe_allow_html=True,
        )
        health_action = (
            "Limit outdoor<br>activity"
            if float(current.get("aqi", 0) or 0) > 100
            else "Normal<br>activity"
        )
        st.markdown(
            f'<div class="kpi-card health-card" style="border-left:5px solid #6366f1;">'
            f'<div class="kpi-label">Health Action</div>'
            f'<div class="health-value">{health_action}</div>'
            f'<div class="kpi-hint">Based on latest daily AQI</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    min_val = history.get("min", "—")
    with g3:
        st.markdown(
            kpi_card(
                "7-Day Minimum",
                min_val,
                "Best recent day",
                band_color_for_value(min_val),
            ),
            unsafe_allow_html=True,
        )

    max_val = history.get("max", "—")
    with g4:
        st.markdown(
            kpi_card(
                "7-Day Maximum",
                max_val,
                "Worst recent day",
                band_color_for_value(max_val),
            ),
            unsafe_allow_html=True,
        )

    avg_val = history.get("average", "—")
    with g5:
        st.markdown(
            kpi_card(
                "7-Day Average",
                avg_val,
                "Rolling week",
                band_color_for_value(avg_val),
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">3-Day AQI Forecast · Health Advisory</div>'
        '<div class="section-subtitle">AI-powered AQI outlook for the next three days</div>',
        unsafe_allow_html=True,
    )

    forecast_cols = st.columns(max(len(predictions), 1))

    for column, (key, pred) in zip(forecast_cols, predictions.items()):
        pred_status = pred.get("status", "Unknown")
        pred_aqi_val = pred.get("aqi", "—")
        pred_color = band_color_for_value(pred_aqi_val)

        with column:
            st.markdown(
                f"""
<div class="forecast-card" style="border-top-color:{pred_color}">
  <div class="forecast-day">{key.replace("_", " ").upper()} · {pred.get("date", "")}</div>
  <div class="aqi-num" style="color:{pred_color}">{pred_aqi_val}</div>
  <div class="forecast-status" style="color:{pred_color}">{pred_status}</div>
  <div class="advice">{pred.get("health_advice", "")}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    left, right = st.columns((1.55, 1))

    with left:
        if hours:
            st.plotly_chart(
                trend_chart(hours),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.info("No hourly observations in the API payload.")

    with right:
        if lat is not None and lon is not None:
            try:
                scatter_map_available = getattr(go, "Scattermap", None) is not None
                if scatter_map_available:
                    fig = map_chart(float(lat), float(lon), status)
                else:
                    fig = map_chart_fallback(float(lat), float(lon), status)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            except Exception:
                st.plotly_chart(
                    map_chart_fallback(float(lat), float(lon), status),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
        else:
            st.info("Coordinates missing from API payload.")

    st.markdown(
        '<div class="section-title">Pollutants · Weather · Models</div>',
        unsafe_allow_html=True,
    )

    poll_col, weather_col, model_col = st.columns(3)

    with poll_col:
        if pollutants:
            st.plotly_chart(
                pollutant_chart(pollutants),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.info("No pollutant fields in dashboard.json.")

    with weather_col:
        if weather:
            st.plotly_chart(
                weather_chart(weather),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.info("No weather fields in dashboard.json.")

    with model_col:
        if models:
            rows = "".join(
                f'<div class="tile-row">'
                f'<span>{key.replace("_", " ").title()}</span>'
                f'<b>{value.get("name", "—")} v{value.get("version", "—")}</b>'
                f'</div>'
                for key, value in models.items()
            )

            st.markdown(
                f'<div class="tile">'
                f'<div class="kpi-label">Serving Models</div>'
                f'{rows}'
                f'</div>',
                unsafe_allow_html=True,
            )

            scale_rows = "".join(
                f'<div class="tile-row">'
                f'<span>{lo}–{hi}</span>'
                f'<b style="color:{c}">{name}</b>'
                f'</div>'
                for lo, hi, name, c in AQI_BANDS
            )

            st.markdown(
                f'<div class="tile">'
                f'<div class="kpi-label">US AQI Scale</div>'
                f'{scale_rows}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No model metadata in dashboard.json.")

    st.markdown(
        '<div class="section-title">Model Explainability · SHAP Drivers</div>'
        '<div class="section-subtitle">Feature contribution available from the serving model</div>',
        unsafe_allow_html=True,
    )

    if shap:
        shap_cols = st.columns(max(len(shap), 1))
        for column, (key, items) in zip(shap_cols, shap.items()):
            with column:
                shap_chart(items or [], key.replace("_", " ").title())
    else:
        st.markdown(
            '<div class="info-card">'
            '<div class="info-title">SHAP data not supplied</div>'
            '<div class="info-text">The API payload does not contain SHAP explanations.</div>'
            '</div>',
            unsafe_allow_html=True,
        )


@st.fragment(run_every=timedelta(seconds=REFRESH_SECONDS))
def live_dashboard() -> None:
    spacer, chip_col, action = st.columns([7.5, 2.1, 1.4])

    with chip_col:
        st.markdown('<div style="margin-top: 2px;">', unsafe_allow_html=True)
        render_live_chip(REFRESH_SECONDS)
        st.markdown('</div>', unsafe_allow_html=True)

    with action:
        st.markdown('<div style="margin-bottom:8px;">', unsafe_allow_html=True)
        if st.button("↻ Refresh now", key="manual_refresh_btn", use_container_width=True):
            fetch_dashboard.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    try:
        data, fetched_at = fetch_dashboard()
        render(data, fetched_at)
    except requests.RequestException as exc:
        st.error(
            f"Could not reach FastAPI at {API_URL}. "
            f"Start the API, then refresh. Details: {exc}"
        )
    except Exception as exc:
        st.error(f"Dashboard rendering error: {exc}")

live_dashboard()




# import os
# from datetime import timedelta

# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import requests
# import streamlit as st

# API_URL = os.environ.get("AQI_API_URL", "https://aqi-3-day-predict-mu.vercel.app/").rstrip("/")
# REFRESH_SECONDS = int(os.environ.get("AQI_REFRESH_SECONDS", "60"))

# STATUS_COLORS = {
#     "Good": "#22c55e",
#     "Moderate": "#eab308",
#     "Unhealthy for Sensitive Groups": "#f59e0b",
#     "Unhealthy": "#f43f5e",
#     "Very Unhealthy": "#b78cff",
#     "Hazardous": "#dc143c",
# }

# AQI_BANDS = [
#     (0, 50, "Good", "#22c55e"),
#     (51, 100, "Moderate", "#eab308"),
#     (101, 150, "Sensitive", "#f59e0b"),
#     (151, 200, "Unhealthy", "#f43f5e"),
#     (201, 300, "Very Unhealthy", "#b78cff"),
#     (301, 500, "Hazardous", "#dc143c"),
# ]

# st.set_page_config(
#     page_title="Lahore Air Quality Intelligence",
#     page_icon="🌫️",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# st.markdown(
#     """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

# :root {
#     --bg: #0B0F17;
#     --bg2: #0d131d;
#     --card: rgba(21, 29, 42, 0.90);
#     --card2: rgba(10, 25, 40, 0.94);
#     --blue: #6366f1;
#     --blue2: #818cf8;
#     --yellow: #4338ca;
#     --yellow2: #a5b4fc;
#     --text: #e8eef7;
#     --muted: #94A3B8;
#     --border: rgba(255, 255, 255, 0.07);
# }

# html, body, [class*="css"] {
#     font-family: "Outfit", sans-serif !important;
# }

# .stApp {
#     background:
#         radial-gradient(1200px 600px at 50% -12%, rgba(99,102,241,.05), transparent 60%),
#         linear-gradient(180deg, #0B0F17 0%, #0d1017 100%);
#     color: var(--text);
# }

# [data-testid="stHeader"] {
#     background: rgba(6,19,33,.65);
#     backdrop-filter: blur(14px);
# }

# [data-testid="stToolbar"] {
#     display: none;
# }

# .block-container {
#     padding-top: 3.75rem;
#     padding-bottom: 3rem;
#     max-width: 1450px;
# }

# .hero-wrap {
#     display: flex;
#     justify-content: space-between;
#     align-items: center;
#     gap: 20px;
#     flex-wrap: wrap;
#     padding: 24px 28px;
#     margin-bottom: 12px;
#     border: 1px solid rgba(255,255,255,.08);
#     border-radius: 24px;
#     background: linear-gradient(135deg, rgba(15,43,65,.94), rgba(9,25,41,.88));
#     box-shadow: 0 1px 2px rgba(0,0,0,.4);
# }

# .kicker {
#     letter-spacing: .17em;
#     text-transform: uppercase;
#     color: var(--blue);
#     font-size: 11px;
#     font-weight: 800;
# }

# .hero-title {
#     font-size: 2.2rem;
#     font-weight: 800;
#     margin: 4px 0 7px;
#     letter-spacing: -.02em;
# }

# .hero-sub {
#     color: var(--muted);
#     font-size: .9rem;
#     line-height: 1.65;
# }

# .hero-sub b {
#     color: #c9d9e8;
# }

# .live-chip {
#     display: inline-flex;
#     align-items: center;
#     gap: 9px;
#     background: rgba(34,197,94,.09);
#     border: 1px solid rgba(34,197,94,.30);
#     color: #86efac;
#     border-radius: 999px;
#     padding: 10px 16px;
#     font-weight: 700;
#     white-space: nowrap;
# }

# .pulse {
#     width: 9px;
#     height: 9px;
#     border-radius: 50%;
#     background: #22c55e;
#     box-shadow: 0 0 0 0 rgba(34,197,94,.65);
#     animation: pulse 1.6s infinite;
# }

# @keyframes pulse {
#     70% { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
# }

# div[data-testid="stButton"] > button {
#     background: rgba(99,102,241,.10);
#     border: 1px solid rgba(99,102,241,.38);
#     color: #c7d2fe;
#     border-radius: 999px;
#     font-weight: 700;
#     padding: 8px 16px;
#     transition: background .15s ease, border-color .15s ease, color .15s ease;
# }

# div[data-testid="stButton"] > button:hover {
#     background: rgba(99,102,241,.20);
#     border-color: rgba(99,102,241,.60);
#     color: #e0e7ff;
# }

# div[data-testid="stButton"] > button:active {
#     background: rgba(99,102,241,.28);
# }

# div[data-testid="stButton"] > button:focus:not(:active) {
#     color: #e0e7ff;
#     border-color: rgba(99,102,241,.60);
# }

# .alert-banner {
#     border-radius: 16px;
#     padding: 13px 18px;
#     margin: 10px 0 18px;
#     font-weight: 650;
#     border: 1px solid;
# }

# .kpi-card,
# .forecast-card,
# .tile,
# .map-card,
# .info-card {
#     background: rgba(19,26,38,.92);
#     border: 1px solid var(--border);
#     border-radius: 20px;
#     box-shadow: 0 1px 2px rgba(0,0,0,.35);
# }

# .kpi-card {
#     padding: 18px 19px;
#     margin-bottom: 12px;
#     min-height: 118px;
# }

# .kpi-label {
#     color: #8fa7bc;
#     font-size: 10px;
#     letter-spacing: .13em;
#     text-transform: uppercase;
#     font-weight: 800;
# }

# .kpi-value {
#     font-size: 1.95rem;
#     font-weight: 700;
#     margin-top: 7px;
#     line-height: 1.15;
# }

# .kpi-hint {
#     color: #8eb9d7;
#     font-size: .78rem;
#     margin-top: 7px;
# }

# .health-card {
#     min-height: 125px;
# }

# .health-value {
#     color: var(--blue);
#     font-size: 1.55rem;
#     font-weight: 700;
#     margin-top: 7px;
#     line-height: 1.15;
# }

# .section-title {
#     font-size: 1.2rem;
#     font-weight: 650;
#     margin: 18px 0 4px;
#     color: #e8eef7;
# }

# .section-subtitle {
#     color: #728ba1;
#     font-size: .78rem;
#     margin-bottom: 10px;
# }

# .forecast-card {
#     min-height: 218px;
#     padding: 20px;
#     border-top: 4px solid var(--blue);
# }

# .forecast-day {
#     color: #91a6ba;
#     font-size: 10px;
#     letter-spacing: .14em;
#     font-weight: 800;
# }

# .aqi-num {
#     font-size: 2.35rem;
#     font-weight: 800;
#     line-height: 1;
#     margin-top: 15px;
# }

# .forecast-status {
#     font-weight: 800;
#     margin-top: 8px;
# }

# .advice {
#     color: #c4d1dd;
#     font-size: .87rem;
#     line-height: 1.5;
#     margin-top: 12px;
# }

# .map-card {
#     padding: 5px;
#     overflow: hidden;
# }

# .tile {
#     padding: 18px;
#     margin-bottom: 12px;
# }

# .tile-row {
#     display: flex;
#     justify-content: space-between;
#     gap: 14px;
#     padding: 10px 2px;
#     border-bottom: 1px solid rgba(148,163,184,.11);
#     font-size: .88rem;
# }

# .tile-row:last-child {
#     border-bottom: 0;
# }

# .tile-row b {
#     text-align: right;
# }

# .info-card {
#     padding: 18px;
#     margin-top: 8px;
# }

# .info-title {
#     color: var(--yellow2);
#     font-weight: 800;
#     font-size: .95rem;
#     margin-bottom: 6px;
# }

# .info-text {
#     color: #9db0c1;
#     font-size: .82rem;
#     line-height: 1.55;
# }

# [data-testid="stPlotlyChart"] {
#     border-radius: 20px;
#     border: 1px solid rgba(255,255,255,.06);
#     background: rgba(5,16,27,.28);
#     padding: 4px;
# }

# div[data-testid="stVerticalBlock"] > div:has(.forecast-card) {
#     min-width: 0;
# }

# @media (max-width: 900px) {
#     .hero-title { font-size: 1.75rem; }
#     .block-container { padding-left: .7rem; padding-right: .7rem; }
# }
# </style>
# """,
#     unsafe_allow_html=True,
# )


# def aqi_color(status: str) -> str:
#     return STATUS_COLORS.get(status, "#6366f1")


# def plotly_theme(fig: go.Figure, height: int = 360) -> go.Figure:
#     fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(5,16,27,.48)",
#         font=dict(color="#dbe7f3", family="Outfit, sans-serif"),
#         height=height,
#         margin=dict(l=18, r=18, t=48, b=22),
#         legend=dict(bgcolor="rgba(0,0,0,0)"),
#         xaxis=dict(
#             gridcolor="rgba(148,163,184,.13)",
#             zeroline=False,
#             color="#9db0c1",
#         ),
#         yaxis=dict(
#             gridcolor="rgba(148,163,184,.13)",
#             zeroline=False,
#             color="#9db0c1",
#         ),
#     )
#     return fig


# def kpi_card(label: str, value, hint: str = "", accent: str = "#6366f1") -> str:
#     return (
#         f'<div class="kpi-card" style="border-left:5px solid {accent};">'
#         f'<div class="kpi-label">{label}</div>'
#         f'<div class="kpi-value" style="color:{accent}">{value}</div>'
#         f'<div class="kpi-hint">{hint}</div>'
#         f'</div>'
#     )


# def status_banner(status: str, advice: str) -> None:
#     color = aqi_color(status)
#     st.markdown(
#         f'<div class="alert-banner" style="background:{color}16;border-color:{color};color:{color}">'
#         f'ALERT · {status} — {advice}</div>',
#         unsafe_allow_html=True,
#     )


# @st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
# def fetch_dashboard() -> dict:
#     response = requests.get(f"{API_URL}/api/dashboard", timeout=20)
#     response.raise_for_status()
#     return response.json()


# def gauge_chart(aqi: float, status: str, reference_aqi: float | None = None) -> go.Figure:
#     """Enhanced AQI gauge: finer ticks, glow ring, delta vs 7-day average,
#     and a compact category legend under the arc."""
#     color = aqi_color(status)
#     aqi = float(aqi)
#     axis_max = 300 if aqi <= 260 else max(320, aqi * 1.15)

#     delta_block = None
#     if reference_aqi is not None:
#         delta_block = {
#             "reference": float(reference_aqi),
#             "valueformat": ".0f",
#             "increasing": {"color": "#f43f5e"},
#             "decreasing": {"color": "#22c55e"},
#             "font": {"size": 14, "color": "#9db0c1"},
#         }

#     fig = go.Figure(
#         go.Indicator(
#             mode="gauge+number+delta" if delta_block else "gauge+number",
#             value=aqi,
#             delta=delta_block,
#             number={
#                 "suffix": " AQI",
#                 "font": {"size": 40, "color": color, "family": "Outfit, sans-serif"},
#             },
#             gauge={
#                 "axis": {
#                     "range": [0, axis_max],
#                     "tickcolor": "#6d84a0",
#                     "tickfont": {"color": "#93a4bb", "size": 11},
#                     "dtick": 50,
#                     "ticklen": 6,
#                 },
                
#                 "bar": {"color": color, "thickness": 0.30},
#                 "bgcolor": "rgba(0,0,0,0)",
#                 "borderwidth": 0,
#                 "steps": [
#                     {"range": [0, 50], "color": "rgba(34,197,94,.32)"},
#                     {"range": [50, 100], "color": "rgba(234,179,8,.32)"},
#                     {"range": [100, 150], "color": "rgba(245,158,11,.32)"},
#                     {"range": [150, 200], "color": "rgba(244,63,94,.38)"},
#                     {"range": [200, axis_max], "color": "rgba(183,140,255,.36)"},
#                 ],
#                 "threshold": {
#                     "line": {"color": "#e2e8f0", "width": 3},
#                     "thickness": 0.9,
#                     "value": aqi,
#                 },
#             },
#             title={
#                 "text": f"<b>{status}</b>",
#                 "font": {"size": 18, "color": color},
#             },
#             domain={"x": [0, 1], "y": [0.08, 1]},
#         )
#     )

#     legend = " · ".join(
#         f'<span style="color:{c}">●</span> {name}' for _, _, name, c in AQI_BANDS[:5]
#     )
#     fig.update_layout(
#         annotations=[
#             dict(
#                 x=0.5,
#                 y=-0.02,
#                 xref="paper",
#                 yref="paper",
#                 text=legend,
#                 showarrow=False,
#                 font=dict(size=9.5, color="#7c8fa4"),
#             )
#         ]
#     )
#     return plotly_theme(fig, 360)


# def trend_chart(hours: list[dict]) -> go.Figure:
#     hour_df = pd.DataFrame(hours)
#     hour_df["time"] = pd.to_datetime(hour_df["time"])

#     fig = go.Figure()
#     bands = [
#         (0, 50, "rgba(34,197,94,.07)", "Good"),
#         (50, 100, "rgba(234,179,8,.07)", "Moderate"),
#         (100, 150, "rgba(245,158,11,.08)", "Sensitive"),
#         (150, 200, "rgba(244,63,94,.09)", "Unhealthy"),
#     ]

#     y_max = max(200, float(hour_df["aqi"].max()) * 1.12)

#     for low, high, fill, label in bands:
#         fig.add_hrect(y0=low, y1=min(high, y_max), fillcolor=fill, line_width=0)
#         fig.add_annotation(
#             x=hour_df["time"].iloc[0],
#             y=(low + min(high, y_max)) / 2,
#             text=label,
#             showarrow=False,
#             xanchor="left",
#             font=dict(size=9, color="rgba(200,215,230,.35)"),
#         )

    
#     fig.add_trace(
#         go.Scatter(
#             x=hour_df["time"],
#             y=hour_df["aqi"],
#             mode="lines",
#             line=dict(color="rgba(99,102,241,.28)", width=10, shape="spline"),
#             hoverinfo="skip",
#             showlegend=False,
#         )
#     )

#     fig.add_trace(
#         go.Scatter(
#             x=hour_df["time"],
#             y=hour_df["aqi"],
#             mode="lines+markers",
#             line=dict(color="#6366f1", width=3.5, shape="spline"),
#             marker=dict(
#                 size=7,
#                 color="#e0f2fe",
#                 line=dict(color="#0f766e", width=1),
#             ),
#             fill="tozeroy",
#             fillcolor="rgba(99,102,241,.10)",
#             name="US AQI",
#             hovertemplate="%{x|%H:%M}<br>AQI %{y:.2f}<extra></extra>",
#         )
#     )

#     # Highlight the 24h peak so a viewer can spot the worst hour at a glance.
#     peak_idx = hour_df["aqi"].idxmax()
#     peak_time = hour_df["time"].iloc[peak_idx]
#     peak_val = float(hour_df["aqi"].iloc[peak_idx])
#     fig.add_trace(
#         go.Scatter(
#             x=[peak_time],
#             y=[peak_val],
#             mode="markers+text",
#             marker=dict(size=11, color="#e2e8f0", line=dict(color="#0B0F17", width=2)),
#             text=[f"Peak {peak_val:.0f}"],
#             textposition="top center",
#             textfont=dict(size=11, color="#cbd5e1"),
#             showlegend=False,
#             hoverinfo="skip",
#         )
#     )

#     fig.update_layout(
#         title="Observed US AQI · Last 24 Hours",
#         yaxis_title="AQI",
#         yaxis=dict(range=[0, y_max]),
#     )
#     return plotly_theme(fig, 390)


# def shap_chart(items: list[dict], title: str) -> None:
#     if not items:
#         st.markdown(
#             '<div class="info-card">'
#             '<div class="info-title">Interpretability unavailable</div>'
#             '<div class="info-text">No SHAP values were supplied for this forecast model.</div>'
#             '</div>',
#             unsafe_allow_html=True,
#         )
#         return

#     frame = pd.DataFrame(items)

#     if "importance" not in frame.columns or "feature" not in frame.columns:
#         st.markdown(
#             '<div class="info-card">'
#             '<div class="info-title">Interpretability unavailable</div>'
#             '<div class="info-text">The API returned SHAP data in an unsupported format.</div>'
#             '</div>',
#             unsafe_allow_html=True,
#         )
#         return

#     frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0)
#     frame = frame[frame["importance"].abs() > 1e-12].copy()

#     # Do not show a misleading graph full of zeros.
#     if frame.empty:
#         st.markdown(
#             '<div class="info-card">'
#             '<div class="info-title">SHAP not available for this horizon</div>'
#             '<div class="info-text">This model did not provide non-zero SHAP explanations. '
#             'The forecast is still valid; only the feature-level explanation is unavailable.</div>'
#             '</div>',
#             unsafe_allow_html=True,
#         )
#         return

#     frame = frame.sort_values("importance", ascending=True).tail(10)

#     fig = px.bar(
#         frame,
#         x="importance",
#         y="feature",
#         orientation="h",
#         color="importance",
#         color_continuous_scale=["#312e81", "#6366f1", "#a5b4fc", "#f43f5e"],
#         title=title,
#     )
#     fig.update_layout(coloraxis_showscale=False)
#     st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)


# def pollutant_chart(pollutants: dict) -> go.Figure:
#     frame = pd.DataFrame(
#         {
#             "pollutant": list(pollutants.keys()),
#             "value": list(pollutants.values()),
#         }
#     )
#     fig = px.bar(
#         frame,
#         x="pollutant",
#         y="value",
#         color="value",
#         color_continuous_scale=["#312e81", "#6366f1", "#a5b4fc", "#f43f5e"],
#         title="Current Pollutant Mix",
#     )
#     fig.update_layout(coloraxis_showscale=False)
#     return plotly_theme(fig, 340)


# def weather_chart(weather: dict) -> go.Figure:
#     labels = [key.replace("_", " ").title() for key in weather]
#     values = list(weather.values())

#     fig = go.Figure(
#         go.Bar(
#             x=values,
#             y=labels,
#             orientation="h",
#             marker=dict(
#                 color=["#6366f1", "#818cf8", "#a5b4fc", "#94A3B8"][:len(values)]
#             ),
#         )
#     )
#     fig.update_layout(title="Weather Snapshot")
#     return plotly_theme(fig, 340)


# def map_chart(lat: float, lon: float, status: str) -> go.Figure:
#     color = aqi_color(status)

#     fig = go.Figure()

#     fig.add_trace(
#         go.Scattermap(
#             lat=[lat],
#             lon=[lon],
#             mode="markers",
#             marker=dict(size=90, color="rgba(99,102,241,.14)"),
#             hoverinfo="skip",
#             showlegend=False,
#         )
#     )

#     fig.add_trace(
#         go.Scattermap(
#             lat=[lat],
#             lon=[lon],
#             mode="markers",
#             marker=dict(size=24, color=color),
#             text=[f"Lahore Monitoring Zone · {status}"],
#             hovertemplate="%{text}<extra></extra>",
#             showlegend=False,
#         )
#     )

#     fig.add_trace(
#         go.Scattermap(
#             lat=[lat],
#             lon=[lon],
#             mode="markers",
#             marker=dict(size=11, color="#e2e8f0"),
#             hoverinfo="skip",
#             showlegend=False,
#         )
#     )

#     fig.update_layout(
#         map=dict(
#             style="carto-darkmatter",
#             center=dict(lat=lat, lon=lon),
#             zoom=10.3,
#         ),
#         title="Lahore Monitoring Zone",
#     )
#     return plotly_theme(fig, 390)


# def map_chart_fallback(lat: float, lon: float, status: str) -> go.Figure:
#     color = aqi_color(status)
#     fig = go.Figure(
#         go.Scattermapbox(
#             lat=[lat],
#             lon=[lon],
#             mode="markers",
#             marker=dict(size=28, color=color),
#             text=[f"Lahore Monitoring Zone · {status}"],
#             hovertemplate="%{text}<extra></extra>",
#         )
#     )
#     fig.update_layout(
#         mapbox=dict(
#             style="carto-darkmatter",
#             center=dict(lat=lat, lon=lon),
#             zoom=10.3,
#         ),
#         title="Lahore Monitoring Zone",
#     )
#     return plotly_theme(fig, 390)


# def render(data: dict) -> None:
#     location = data.get("location") or {}
#     current = data.get("current") or {}
#     predictions = data.get("predictions") or {}
#     history = data.get("last_7_days") or {}
#     hours = data.get("last_24_hours") or []
#     pollutants = data.get("pollutants") or {}
#     weather = data.get("weather") or {}
#     models = data.get("models") or {}
#     shap = data.get("shap") or {}
#     meta = data.get("meta") or {}

#     status = current.get("status", "Unknown")
#     color = aqi_color(status)
#     city = location.get("city", "Lahore")
#     lat = location.get("latitude")
#     lon = location.get("longitude")

#     latest_date = data.get("latest_available_date", "—")
#     generated = meta.get("generated_at", "—")


#     if generated != "—":
#         try:
#             generated_display = pd.to_datetime(generated).strftime("%d %b %Y · %H:%M UTC")
#         except Exception:
#             generated_display = generated
#     else:
#         generated_display = "—"

#     st.markdown(
#         f"""
# <div class="hero-wrap">
#   <div>
#     <div class="kicker">Pakistan · Punjab · Air Quality Monitoring</div>
#     <div class="hero-title">Lahore Air Quality Intelligence</div>
#     <div class="hero-sub">
#       <b>{city} Monitoring Zone</b><br>
#       Coordinates: {lat}, {lon}<br>
#       Latest data: {latest_date}<br>
#       Updated: {generated_display}
#     </div>
#   </div>
#   <div class="live-chip"><span class="pulse"></span>Live · refresh {REFRESH_SECONDS}s</div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )

#     status_banner(status, current.get("health_advice", ""))

#     g1, g2, g3, g4, g5 = st.columns([1.25, 1, 1, 1, 1])

#     with g1:
#         if current.get("aqi") is not None:
#             st.plotly_chart(
#                 gauge_chart(current["aqi"], status, history.get("average")),
#                 use_container_width=True,
#             )

#     with g2:
#         st.markdown(
#             kpi_card("Current AQI", current.get("aqi", "—"), status, color),
#             unsafe_allow_html=True,
#         )
#         health_action = (
#             "Limit outdoor<br>activity"
#             if float(current.get("aqi", 0) or 0) > 100
#             else "Normal<br>activity"
#         )
#         st.markdown(
#             f'<div class="kpi-card health-card" style="border-left:5px solid #6366f1;">'
#             f'<div class="kpi-label">Health Action</div>'
#             f'<div class="health-value">{health_action}</div>'
#             f'<div class="kpi-hint">Based on latest daily AQI</div>'
#             f'</div>',
#             unsafe_allow_html=True,
#         )

#     with g3:
#         st.markdown(
#             kpi_card(
#                 "7-Day Minimum",
#                 history.get("min", "—"),
#                 "Best recent day",
#                 "#22c55e",
#             ),
#             unsafe_allow_html=True,
#         )

#     with g4:
#         st.markdown(
#             kpi_card(
#                 "7-Day Maximum",
#                 history.get("max", "—"),
#                 "Worst recent day",
#                 "#f43f5e",
#             ),
#             unsafe_allow_html=True,
#         )

#     with g5:
#         st.markdown(
#             kpi_card(
#                 "7-Day Average",
#                 history.get("average", "—"),
#                 "Rolling week",
#                 "#94A3B8",
#             ),
#             unsafe_allow_html=True,
#         )

#     st.markdown(
#         '<div class="section-title">3-Day AQI Forecast · Health Advisory</div>'
#         '<div class="section-subtitle">AI-powered AQI outlook for the next three days</div>',
#         unsafe_allow_html=True,
#     )

#     forecast_cols = st.columns(max(len(predictions), 1))

#     for column, (key, pred) in zip(forecast_cols, predictions.items()):
#         pred_status = pred.get("status", "Unknown")
#         pred_color = aqi_color(pred_status)

#         with column:
#             st.markdown(
#                 f"""
# <div class="forecast-card" style="border-top-color:{pred_color}">
#   <div class="forecast-day">{key.replace("_", " ").upper()} · {pred.get("date", "")}</div>
#   <div class="aqi-num" style="color:{pred_color}">{pred.get("aqi", "—")}</div>
#   <div class="forecast-status" style="color:{pred_color}">{pred_status}</div>
#   <div class="advice">{pred.get("health_advice", "")}</div>
# </div>
# """,
#                 unsafe_allow_html=True,
#             )

#     left, right = st.columns((1.55, 1))

#     with left:
#         if hours:
#             st.plotly_chart(
#                 trend_chart(hours),
#                 use_container_width=True,
#             )
#         else:
#             st.info("No hourly observations in the API payload.")

#     with right:
#         if lat is not None and lon is not None:
#             try:
#                 scatter_map_available = getattr(go, "Scattermap", None) is not None
#                 if scatter_map_available:
#                     fig = map_chart(float(lat), float(lon), status)
#                 else:
#                     fig = map_chart_fallback(float(lat), float(lon), status)
#                 st.plotly_chart(fig, use_container_width=True)
#             except Exception:
#                 st.plotly_chart(
#                     map_chart_fallback(float(lat), float(lon), status),
#                     use_container_width=True,
#                 )
#         else:
#             st.info("Coordinates missing from API payload.")

#     st.markdown(
#         '<div class="section-title">Pollutants · Weather · Models</div>',
#         unsafe_allow_html=True,
#     )

#     poll_col, weather_col, model_col = st.columns(3)

#     with poll_col:
#         if pollutants:
#             st.plotly_chart(
#                 pollutant_chart(pollutants),
#                 use_container_width=True,
#             )
#         else:
#             st.info("No pollutant fields in dashboard.json.")

#     with weather_col:
#         if weather:
#             st.plotly_chart(
#                 weather_chart(weather),
#                 use_container_width=True,
#             )
#         else:
#             st.info("No weather fields in dashboard.json.")

#     with model_col:
#         if models:
#             rows = "".join(
#                 f'<div class="tile-row">'
#                 f'<span>{key.replace("_", " ").title()}</span>'
#                 f'<b>{value.get("name", "—")} v{value.get("version", "—")}</b>'
#                 f'</div>'
#                 for key, value in models.items()
#             )

#             st.markdown(
#                 f'<div class="tile">'
#                 f'<div class="kpi-label">Serving Models</div>'
#                 f'{rows}'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )

#             scale_rows = "".join(
#                 f'<div class="tile-row">'
#                 f'<span>{lo}–{hi}</span>'
#                 f'<b style="color:{c}">{name}</b>'
#                 f'</div>'
#                 for lo, hi, name, c in AQI_BANDS
#             )

#             st.markdown(
#                 f'<div class="tile">'
#                 f'<div class="kpi-label">US AQI Scale</div>'
#                 f'{scale_rows}'
#                 f'</div>',
#                 unsafe_allow_html=True,
#             )
#         else:
#             st.info("No model metadata in dashboard.json.")

#     st.markdown(
#         '<div class="section-title">Model Explainability · SHAP Drivers</div>'
#         '<div class="section-subtitle">Feature contribution available from the serving model</div>',
#         unsafe_allow_html=True,
#     )

#     if shap:
#         shap_cols = st.columns(max(len(shap), 1))
#         for column, (key, items) in zip(shap_cols, shap.items()):
#             with column:
#                 shap_chart(items or [], key.replace("_", " ").title())
#     else:
#         st.markdown(
#             '<div class="info-card">'
#             '<div class="info-title">SHAP data not supplied</div>'
#             '<div class="info-text">The API payload does not contain SHAP explanations.</div>'
#             '</div>',
#             unsafe_allow_html=True,
#         )


# @st.fragment(run_every=timedelta(seconds=REFRESH_SECONDS))
# def live_dashboard() -> None:
#     spacer, action = st.columns([9, 1.4])
#     with action:
#         st.markdown('<div style="margin-bottom:8px;">', unsafe_allow_html=True)
#         if st.button("↻ Refresh now", key="manual_refresh_btn", use_container_width=True):
#             fetch_dashboard.clear()
#         st.markdown('</div>', unsafe_allow_html=True)

#     try:
#         data = fetch_dashboard()
#         render(data)
#     except requests.RequestException as exc:
#         st.error(
#             f"Could not reach FastAPI at {API_URL}. "
#             f"Start the API, then refresh. Details: {exc}"
#         )
#     except Exception as exc:
#         st.error(f"Dashboard rendering error: {exc}")


# live_dashboard()
