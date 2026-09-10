"""Rule-based AWS anomaly typing for SkyGuard.

Thresholds are calibrated on the SkyGuard AWS dataset (5-minute cadence,
four stations). Typical inlier |ΔT| p99 ≈ 1.4 °C (max ~2 °C), |ΔP| p99 ≈ 1.5 hPa,
|ΔRH| p99 ≈ 5.6 %. Labeled spikes jump ~20 °C; false spikes revert on the next
reading; frozen temperature stays bit-identical for ≥15 samples; drift is a
slow multi-sample offset; missing rows are all-NaN; multivariate is a T–RH
correlation break while each reading stays in a physically plausible range.

Check order: missing → frozen → spike/false_spike → drift → multivariate.

false_spike: if `history` contains a later timestamp for the same station than
`row`, that next reading is used to test reversion. In streaming calls where
history is only the past, a jump is reported as `spike` (no look-ahead).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from features import SENSOR_COLS, normalize_station_frame

# Absolute jump vs previous / rolling mean that exceeds inlier noise.
SPIKE_ABS = {
    "temperature_c": 4.0,
    "pressure_hpa": 3.5,
    "humidity_pct": 15.0,
}
SPIKE_ROLL_K = 4.0
FALSE_SPIKE_REVERT_FRAC = 0.45

FROZEN_N = 4
FROZEN_ATOL = 1e-6

DRIFT_N = 18
DRIFT_MIN_CUMUL = {
    "temperature_c": 2.5,
    "pressure_hpa": 2.2,
    "humidity_pct": 8.0,
}
DRIFT_SIGN_FRAC = 0.55
DRIFT_BASELINE_GAP = {
    "temperature_c": 3.0,
    "pressure_hpa": 2.5,
    "humidity_pct": 8.0,
}

# Residual |RH - RH_hat(T)| from history OLS; labeled multivariate ≈ 42 % RH.
MULTIVARIATE_RH_RESIDUAL = 20.0
PLAUSIBLE = {
    "temperature_c": (5.0, 50.0),
    "pressure_hpa": (980.0, 1040.0),
    "humidity_pct": (5.0, 100.0),
}


def _station_id(row: Mapping[str, Any]) -> str:
    if "station_id" in row and pd.notna(row["station_id"]):
        return str(row["station_id"])
    return str(row.get("station", ""))


def _as_series(row: Mapping[str, Any] | pd.Series) -> pd.Series:
    return row if isinstance(row, pd.Series) else pd.Series(row)


def _station_history(history: pd.DataFrame | None, station: str) -> pd.DataFrame:
    if history is None or len(history) == 0:
        return pd.DataFrame()
    hist = normalize_station_frame(history)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    if "station" in hist.columns and station:
        hist = hist.loc[hist["station"].astype(str) == station]
    return hist.sort_values("timestamp")


def _is_missing(row: pd.Series) -> bool:
    vals = [row.get(c) for c in SENSOR_COLS]
    return any(v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) for v in vals)


def _is_plausible(row: pd.Series) -> bool:
    for col, (lo, hi) in PLAUSIBLE.items():
        v = row.get(col)
        if v is None or pd.isna(v):
            return False
        if not (lo <= float(v) <= hi):
            return False
    return True


def _prior_and_next(
    hist: pd.DataFrame, ts: pd.Timestamp
) -> tuple[pd.DataFrame, pd.Series | None]:
    prior = hist.loc[hist["timestamp"] < ts]
    later = hist.loc[hist["timestamp"] > ts]
    nxt = later.iloc[0] if len(later) else None
    return prior, nxt


def _rolling_ref(prior: pd.DataFrame, col: str, current: float) -> tuple[float, float]:
    series = prior[col].dropna() if col in prior.columns else pd.Series(dtype=float)
    if len(series) >= 6:
        mean = float(series.iloc[-6:].mean())
        std = float(series.iloc[-6:].std(ddof=0))
        return mean, max(std, 1e-6)
    if len(series):
        return float(series.iloc[-1]), 1.0
    return current, 1.0


def _is_spike_value(value: float, prev: float | None, roll_mean: float, roll_std: float, col: str) -> bool:
    abs_th = SPIKE_ABS[col]
    vs_roll = abs(value - roll_mean) >= max(abs_th, SPIKE_ROLL_K * roll_std)
    vs_prev = prev is not None and abs(value - prev) >= abs_th
    return bool(vs_roll or vs_prev)


def _spike_sensor(row: pd.Series, prior: pd.DataFrame) -> str | None:
    for col in SENSOR_COLS:
        val = row.get(col)
        if val is None or pd.isna(val):
            continue
        val = float(val)
        prev = float(prior[col].iloc[-1]) if len(prior) and pd.notna(prior[col].iloc[-1]) else None
        mean, std = _rolling_ref(prior, col, val)
        if _is_spike_value(val, prev, mean, std, col):
            return col
    return None


def _reverted(row: pd.Series, prior: pd.DataFrame, nxt: pd.Series, col: str) -> bool:
    cur = float(row[col])
    nxt_v = nxt.get(col)
    if nxt_v is None or pd.isna(nxt_v):
        return False
    nxt_v = float(nxt_v)
    mean, _ = _rolling_ref(prior, col, cur)
    prev = float(prior[col].iloc[-1]) if len(prior) and pd.notna(prior[col].iloc[-1]) else mean
    baseline = prev
    jump = abs(cur - baseline)
    if jump < SPIKE_ABS[col]:
        return False
    back_to_baseline = abs(nxt_v - baseline) <= max(SPIKE_ABS[col] * FALSE_SPIKE_REVERT_FRAC, 2.0)
    left_the_spike = abs(nxt_v - cur) >= jump * 0.55
    return bool(back_to_baseline and left_the_spike)


def _is_frozen(row: pd.Series, prior: pd.DataFrame) -> bool:
    need = FROZEN_N - 1
    if len(prior) < need:
        return False
    tail = prior.iloc[-need:]
    for col in SENSOR_COLS:
        cur = row.get(col)
        if cur is None or pd.isna(cur):
            continue
        window = list(tail[col].tolist()) + [cur]
        if any(v is None or pd.isna(v) for v in window):
            continue
        arr = np.asarray(window, dtype=float)
        if np.allclose(arr, arr[0], atol=FROZEN_ATOL, rtol=0):
            return True
    return False


def _is_drift(row: pd.Series, prior: pd.DataFrame) -> bool:
    if _spike_sensor(row, prior):
        return False
    combined = pd.concat([prior, pd.DataFrame([row])], ignore_index=True)
    if len(combined) < 8:
        return False

    window = combined.tail(DRIFT_N)
    for col in SENSOR_COLS:
        vals = pd.to_numeric(window[col], errors="coerce").dropna()
        if len(vals) < 8:
            continue
        diffs = np.diff(vals.to_numpy(dtype=float))
        if np.any(np.abs(diffs) >= SPIKE_ABS[col]):
            continue
        total = float(vals.iloc[-1] - vals.iloc[0])
        if abs(total) < DRIFT_MIN_CUMUL[col]:
            continue
        sign = np.sign(total)
        if sign == 0:
            continue
        frac = float(np.mean(np.sign(diffs) == sign))
        if frac >= DRIFT_SIGN_FRAC:
            return True

    if len(combined) >= 24:
        recent = combined.tail(6)
        older = combined.iloc[-24:-12]
        for col in SENSOR_COLS:
            r = pd.to_numeric(recent[col], errors="coerce").mean()
            o = pd.to_numeric(older[col], errors="coerce").mean()
            if pd.isna(r) or pd.isna(o):
                continue
            if abs(float(r) - float(o)) >= DRIFT_BASELINE_GAP[col]:
                if _spike_sensor(row, prior) is None:
                    return True
    return False


def _is_multivariate(row: pd.Series, prior: pd.DataFrame) -> bool:
    if not _is_plausible(row):
        return False
    if _spike_sensor(row, prior):
        return False
    hist = prior.dropna(subset=SENSOR_COLS)
    if len(hist) < 20:
        return False
    t = hist["temperature_c"].to_numpy(dtype=float)
    rh = hist["humidity_pct"].to_numpy(dtype=float)
    if np.std(t) < 1e-3:
        return False
    slope, intercept = np.polyfit(t, rh, 1)
    pred = float(slope * float(row["temperature_c"]) + intercept)
    residual = abs(float(row["humidity_pct"]) - pred)
    t_mean, t_std = _rolling_ref(prior, "temperature_c", float(row["temperature_c"]))
    individually_ok = abs(float(row["temperature_c"]) - t_mean) < max(2.5, 3.0 * t_std)
    return bool(residual >= MULTIVARIATE_RH_RESIDUAL and individually_ok)


def frozen_mask(df: pd.DataFrame) -> pd.Series:
    """Vectorized stuck-sensor detector (any channel identical for FROZEN_N samples)."""
    frame = normalize_station_frame(df)
    flags = pd.Series(False, index=frame.index)
    for col in SENSOR_COLS:

        def _run_len(s: pd.Series) -> pd.Series:
            changed = s.ne(s.shift()) | s.isna() | s.shift().isna()
            return s.groupby(changed.cumsum()).cumcount() + 1

        run = frame.groupby("station", sort=False)[col].transform(_run_len)
        flags = flags | ((run >= FROZEN_N) & frame[col].notna())
    return flags


def classify(row: Mapping[str, Any] | pd.Series, history: pd.DataFrame | None) -> str:
    """Return one of spike, false_spike, frozen, drift, missing, multivariate."""
    row_s = _as_series(row)
    ts = pd.to_datetime(row_s.get("timestamp"))
    station = _station_id(row_s)
    hist = _station_history(history, station)
    prior, nxt = _prior_and_next(hist, ts) if len(hist) else (pd.DataFrame(), None)

    if _is_missing(row_s):
        return "missing"
    if _is_frozen(row_s, prior):
        return "frozen"

    spike_col = _spike_sensor(row_s, prior)
    if spike_col is not None:
        if nxt is not None and _reverted(row_s, prior, nxt, spike_col):
            return "false_spike"
        return "spike"

    if _is_drift(row_s, prior):
        return "drift"
    if _is_multivariate(row_s, prior):
        return "multivariate"
    return "none"
