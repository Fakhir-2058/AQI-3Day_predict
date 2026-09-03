import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Lahore AQI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dashboard.json"


def read_dashboard() -> dict:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard.json not found")
    try:
        return json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid JSON: {exc}") from exc


@app.get("/")
def root():
    return {"service": "lahore-aqi-api", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "has_data": DATA_PATH.exists(),
    }


@app.get("/api/dashboard")
def dashboard():
    payload = read_dashboard()
    mtime = datetime.fromtimestamp(DATA_PATH.stat().st_mtime, tz=timezone.utc)
    payload["meta"] = {
        "generated_at": mtime.isoformat(),
        "refresh_seconds": 60,
        "source": "dashboard.json",
    }
    return JSONResponse(
        content=payload,
        headers={"Cache-Control": "no-store, max-age=0"},
    )
