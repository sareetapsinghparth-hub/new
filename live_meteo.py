"""Live weather pull for the SkyGuard map.

You do not connect a feed to Cursor. Streamlit on this PC calls Open-Meteo
over the internet (no API key). For a real IMD AWS, point LIVE_AWS_URL at
your station HTTPS/MQTT adapter instead.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from features import STATION_GEO, STATION_ORDER

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"


def fetch_open_meteo_network(timeout: float = 12.0) -> pd.DataFrame:
    """Current T / RH / pressure for the four map stations."""
    rows = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for station, geo in STATION_GEO.items():
        params = urllib.parse.urlencode(
            {
                "latitude": geo["lat"],
                "longitude": geo["lon"],
                "current": "temperature_2m,relative_humidity_2m,surface_pressure",
                "timezone": "UTC",
            }
        )
        req = urllib.request.Request(
            f"{OPEN_METEO}?{params}",
            headers={"User-Agent": "SkyGuardAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        cur = payload.get("current") or {}
        rows.append(
            {
                "timestamp": pd.to_datetime(cur.get("time", now.isoformat())),
                "station": station,
                "station_id": station,
                "zone": geo["zone"],
                "temperature_c": cur.get("temperature_2m"),
                "pressure_hpa": cur.get("surface_pressure"),
                "humidity_pct": cur.get("relative_humidity_2m"),
                "is_anomaly": False,
            }
        )
    return pd.DataFrame(rows)


def fetch_open_meteo_safe() -> tuple[pd.DataFrame | None, str]:
    try:
        return fetch_open_meteo_network(), "ok"
    except urllib.error.URLError as exc:
        return None, f"Open-Meteo unreachable ({exc}). This PC needs internet; no API key is required."
    except Exception as exc:
        return None, f"Live fetch failed: {exc}"
