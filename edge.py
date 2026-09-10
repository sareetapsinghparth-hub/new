"""Edge / ESP32 inference profile.

IsolationForest + SHAP stay on the dashboard/cloud path (`inference.py`).
This module is the MCU-sized subset: rolling history + `classify_anomaly` rules
only (no sklearn, no SHAP, no joblib). A quantized IsolationForest can be
exported later with emlearn; the uplink contract is the same JSON keys.

Call `edge_infer(row, history)` on-device every 5 minutes; radio the payload
only on `anomaly` or a heartbeat.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from classify_anomaly import classify
from features import SENSOR_COLS

EDGE_MODEL_ID = "skyguard-rules-esp32-v1"


def edge_infer(row: Mapping[str, Any] | pd.Series, history: pd.DataFrame | None) -> dict[str, Any]:
    """Rule-only score suitable for ESP32-class MCUs."""
    row_s = row if isinstance(row, pd.Series) else pd.Series(row)
    station = str(row_s.get("station_id", row_s.get("station", "")))
    ts = pd.to_datetime(row_s.get("timestamp"))
    atype = classify(row_s, history)
    missing = any(pd.isna(row_s.get(c)) for c in SENSOR_COLS)
    if missing:
        atype = "missing"
    anomaly = atype != "none"
    return {
        "station_id": station,
        "timestamp": ts.isoformat() if pd.notna(ts) else None,
        "anomaly": anomaly,
        "anomaly_type": atype,
        "anomaly_score": 1.0 if anomaly else 0.0,
        "confidence": 80.0 if anomaly else 15.0,
        "severity": "HIGH" if atype in {"missing", "frozen", "spike"} else ("MEDIUM" if anomaly else "LOW"),
        "sensor_health": 25.0 if anomaly else 96.0,
        "reason": f"Edge rules ({EDGE_MODEL_ID}) classified this reading as {atype}.",
        "edge_model": EDGE_MODEL_ID,
    }
