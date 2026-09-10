"""Shared AWS feature engineering for training and in-process inference."""

from __future__ import annotations

import pandas as pd

SENSOR_COLS = ["temperature_c", "pressure_hpa", "humidity_pct"]

STATION_ORDER = ["AWS-E01", "AWS-N01", "AWS-S01", "AWS-W01"]

# Schematic NCR cluster for the live map (zone layout, not surveyed siting).
STATION_GEO = {
    "AWS-N01": {"lat": 28.704, "lon": 77.209, "zone": "North"},
    "AWS-S01": {"lat": 28.527, "lon": 77.209, "zone": "South"},
    "AWS-E01": {"lat": 28.613, "lon": 77.359, "zone": "East"},
    "AWS-W01": {"lat": 28.613, "lon": 77.078, "zone": "West"},
}


def normalize_station_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a `station` column exists (accepts `station_id`)."""
    out = df.copy()
    if "station" not in out.columns and "station_id" in out.columns:
        out["station"] = out["station_id"]
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group-by-station diffs (1/3), rolling mean/std (window=6), hour, T/RH and T/P ratios."""
    out = normalize_station_frame(df)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["station", "timestamp"]).reset_index(drop=True)
    out[SENSOR_COLS] = out.groupby("station", sort=False)[SENSOR_COLS].ffill().bfill()
    grouped = out.groupby("station", sort=False)

    series = {
        "temp": "temperature_c",
        "pressure": "pressure_hpa",
        "humidity": "humidity_pct",
    }
    for prefix, col in series.items():
        out[f"d{prefix}_1"] = grouped[col].diff(1)
        out[f"d{prefix}_3"] = grouped[col].diff(3)
        out[f"{prefix}_roll_mean_6"] = grouped[col].transform(
            lambda s: s.rolling(window=6, min_periods=6).mean()
        )
        out[f"{prefix}_roll_std_6"] = grouped[col].transform(
            lambda s: s.rolling(window=6, min_periods=6).std()
        )

    out["hour_of_day"] = out["timestamp"].dt.hour
    humidity = out["humidity_pct"].replace(0, pd.NA)
    pressure = out["pressure_hpa"].replace(0, pd.NA)
    out["temp_humidity_ratio"] = out["temperature_c"] / humidity
    out["temp_pressure_ratio"] = out["temperature_c"] / pressure
    return out


def feature_columns() -> list[str]:
    cols: list[str] = []
    for prefix in ("temp", "pressure", "humidity"):
        cols.extend(
            [
                f"d{prefix}_1",
                f"d{prefix}_3",
                f"{prefix}_roll_mean_6",
                f"{prefix}_roll_std_6",
            ]
        )
    cols.extend(["hour_of_day", "temp_humidity_ratio", "temp_pressure_ratio"])
    return cols


def features_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    engineered = engineer_features(df)
    cols = feature_columns()
    keep = engineered.dropna(subset=cols + SENSOR_COLS).copy()
    return keep, keep[cols]
