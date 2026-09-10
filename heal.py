"""Self-healing outputs on top of the anomaly log: imputed values + maintenance forecast."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from features import SENSOR_COLS, normalize_station_frame

ROLL_N = 12
TYPE_WEIGHT = {
    "drift": 2.0,
    "frozen": 2.0,
    "missing": 1.5,
    "spike": 1.0,
    "false_spike": 0.5,
    "multivariate": 1.2,
    "none": 0.0,
}


def _ts(val) -> pd.Timestamp:
    return pd.Timestamp(val)


def _station_id(row: Mapping[str, Any]) -> str:
    if "station_id" in row and pd.notna(row.get("station_id")):
        return str(row["station_id"])
    return str(row.get("station", ""))


def _rolling_median(prior: pd.DataFrame, col: str) -> float | None:
    if prior is None or len(prior) == 0 or col not in prior.columns:
        return None
    series = pd.to_numeric(prior[col], errors="coerce").dropna()
    if len(series) == 0:
        return None
    return float(series.iloc[-ROLL_N:].median())


def _neighbor_mean(network: pd.DataFrame | None, ts: pd.Timestamp, station: str, col: str) -> float | None:
    if network is None or len(network) == 0:
        return None
    net = normalize_station_frame(network)
    net["timestamp"] = pd.to_datetime(net["timestamp"])
    peers = net.loc[(net["timestamp"] == ts) & (net["station"].astype(str) != str(station))]
    if col not in peers.columns:
        return None
    vals = pd.to_numeric(peers[col], errors="coerce").dropna()
    if len(vals) == 0:
        return None
    return float(vals.mean())


def suggest_values(
    row: Mapping[str, Any] | pd.Series,
    history: pd.DataFrame | None,
    network: pd.DataFrame | None,
    *,
    anomaly: bool,
    anomaly_type: str,
) -> dict[str, Any]:
    """Corrected T/P/RH: rolling median for spike/frozen/drift/missing; neighbors for multivariate."""
    row_s = row if isinstance(row, pd.Series) else pd.Series(row)
    station = _station_id(row_s)
    ts = _ts(row_s.get("timestamp"))
    reported = {c: (None if pd.isna(row_s.get(c)) else float(row_s.get(c))) for c in SENSOR_COLS}
    empty = {
        "suggested_temperature_c": None,
        "suggested_pressure_hpa": None,
        "suggested_humidity_pct": None,
        "impute_method": "none",
    }
    if not anomaly or anomaly_type in {"none", ""}:
        return empty

    hist = pd.DataFrame() if history is None or len(history) == 0 else normalize_station_frame(history)
    if len(hist):
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        if "station" in hist.columns:
            hist = hist.loc[hist["station"].astype(str) == station]
        prior = hist.loc[hist["timestamp"] < ts]
    else:
        prior = pd.DataFrame()

    spatial = anomaly_type == "multivariate"
    method = "neighbor_mean" if spatial else "rolling_median"
    suggested: dict[str, float | None] = {}
    for col in SENSOR_COLS:
        value = None
        if spatial:
            value = _neighbor_mean(network, ts, station, col)
            if value is None:
                value = _rolling_median(prior, col)
                if value is not None:
                    method = "rolling_median_fallback"
        else:
            value = _rolling_median(prior, col)
            if value is None:
                value = _neighbor_mean(network, ts, station, col)
                if value is not None:
                    method = "neighbor_mean_fallback"
        if value is not None:
            value = round(float(value), 2)
        suggested[col] = value

    if all(v is None for v in suggested.values()):
        method = "unavailable"

    return {
        "suggested_temperature_c": suggested["temperature_c"],
        "suggested_pressure_hpa": suggested["pressure_hpa"],
        "suggested_humidity_pct": suggested["humidity_pct"],
        "impute_method": method,
        "reported_temperature_c": reported["temperature_c"],
        "reported_pressure_hpa": reported["pressure_hpa"],
        "reported_humidity_pct": reported["humidity_pct"],
    }


def forecast_maintenance(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Rule on the anomaly log: rising daily weighted counts → maintenance window."""
    stable = {
        "maintenance_status": "stable",
        "maintenance_days": None,
        "maintenance_message": "stable - no maintenance trend",
    }
    if not results:
        return stable

    rows = []
    for r in results:
        ts = _ts(r.get("timestamp"))
        atype = str(r.get("anomaly_type") or "none")
        flagged = bool(r.get("anomaly"))
        w = TYPE_WEIGHT.get(atype, 1.0) if flagged else 0.0
        rows.append({"date": ts.normalize(), "weight": w, "flag": int(flagged and atype != "none")})
    frame = pd.DataFrame(rows)
    daily = frame.groupby("date", sort=True).agg(weight=("weight", "sum"), flags=("flag", "sum"))
    if len(daily) == 0:
        return stable

    tail = daily.tail(3)["weight"].to_numpy(dtype=float)
    latest = float(tail[-1])
    if len(tail) >= 3 and tail[2] > tail[1] > tail[0] and latest >= 5:
        slope = max(float(tail[2] - tail[1]), 1.0)
        days = int(np.clip(round(12.0 / slope * 3.0), 2, 14))
        return {
            "maintenance_status": "degrading",
            "maintenance_days": days,
            "maintenance_message": f"degrading - maintenance recommended within {days} days",
        }
    if len(tail) >= 2 and tail[-1] > tail[-2] and latest >= 8:
        days = int(np.clip(round(18.0 / max(latest - float(tail[-2]), 1.0)), 3, 14))
        return {
            "maintenance_status": "degrading",
            "maintenance_days": days,
            "maintenance_message": f"degrading - maintenance recommended within {days} days",
        }
    if latest >= 10:
        return {
            "maintenance_status": "watch",
            "maintenance_days": 14,
            "maintenance_message": "watch - elevated anomaly load; inspect within 14 days",
        }
    return stable


def update_daily_forecast(daily: dict, timestamp, anomaly: bool, anomaly_type: str) -> dict[str, Any]:
    """Running maintenance forecast for batch scoring (no per-row DataFrame rebuild)."""
    day = _ts(timestamp).normalize()
    if anomaly and anomaly_type != "none":
        daily[day] = float(daily.get(day, 0.0) + TYPE_WEIGHT.get(anomaly_type, 1.0))
    dates = sorted(daily)
    tail = [float(daily[d]) for d in dates[-3:]] if dates else []
    stable = {
        "maintenance_status": "stable",
        "maintenance_days": None,
        "maintenance_message": "stable - no maintenance trend",
    }
    if not tail:
        return stable
    latest = tail[-1]
    if len(tail) >= 3 and tail[2] > tail[1] > tail[0] and latest >= 5:
        slope = max(float(tail[2] - tail[1]), 1.0)
        days = int(np.clip(round(12.0 / slope * 3.0), 2, 14))
        return {
            "maintenance_status": "degrading",
            "maintenance_days": days,
            "maintenance_message": f"degrading - maintenance recommended within {days} days",
        }
    if len(tail) >= 2 and tail[-1] > tail[-2] and latest >= 8:
        days = int(np.clip(round(18.0 / max(latest - float(tail[-2]), 1.0)), 3, 14))
        return {
            "maintenance_status": "degrading",
            "maintenance_days": days,
            "maintenance_message": f"degrading - maintenance recommended within {days} days",
        }
    if latest >= 10:
        return {
            "maintenance_status": "watch",
            "maintenance_days": 14,
            "maintenance_message": "watch - elevated anomaly load; inspect within 14 days",
        }
    return stable
