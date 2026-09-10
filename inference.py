"""In-process SkyGuard inference (IsolationForest + rules + SHAP + LIME + healing).

Streamlit calls these functions directly. In production a POST /sensor-data
handler would return the same JSON dict as `run_inference` — this module is
not a FastAPI app.

Edge / ESP32: `edge.edge_infer` is the MCU profile (rules only). Cloud keeps
IsolationForest + SHAP. Same JSON keys.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from classify_anomaly import classify, frozen_mask
from edge import EDGE_MODEL_ID, edge_infer
from features import SENSOR_COLS, engineer_features, feature_columns, normalize_station_frame
from heal import forecast_maintenance, suggest_values, update_daily_forecast

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "skyguard_isoforest.joblib"

FEATURE_LABELS = {
    "dtemp_1": "1-step temperature change",
    "dtemp_3": "3-step temperature change",
    "temp_roll_mean_6": "6-step temperature rolling mean",
    "temp_roll_std_6": "6-step temperature rolling std",
    "dpressure_1": "1-step pressure change",
    "dpressure_3": "3-step pressure change",
    "pressure_roll_mean_6": "6-step pressure rolling mean",
    "pressure_roll_std_6": "6-step pressure rolling std",
    "dhumidity_1": "1-step humidity change",
    "dhumidity_3": "3-step humidity change",
    "humidity_roll_mean_6": "6-step humidity rolling mean",
    "humidity_roll_std_6": "6-step humidity rolling std",
    "hour_of_day": "hour of day",
    "temp_humidity_ratio": "temperature/humidity ratio",
    "temp_pressure_ratio": "temperature/pressure ratio",
}

TYPE_REASON = {
    "spike": "A sensor reading jumped well beyond its recent rolling mean.",
    "false_spike": "A large jump appeared but the next reading reverted toward the baseline.",
    "frozen": "A sensor repeated the same value across consecutive readings.",
    "drift": "A slow monotonic offset accumulated without a single spike-sized jump.",
    "missing": "One or more sensor fields were null or NaN.",
    "multivariate": "Individual T/P/RH look plausible but their relationship is inconsistent.",
    "none": "IsolationForest treated this observation as an inlier.",
}

# IsolationForest decision_function: higher = more normal; predict == -1 iff score < 0.
# confidence maps that score through a logistic so ~0 (the sklearn threshold) ≈ 50,
# strongly negative scores approach 100, strongly positive scores approach 0:
#   confidence = 100 / (1 + exp(12 * decision_function))
CONFIDENCE_K = 12.0


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    payload = joblib.load(MODEL_PATH)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unexpected model payload at {MODEL_PATH}")
    return payload


def load_model():
    return load_bundle()["model"]


def _feature_names() -> list[str]:
    names = load_bundle().get("feature_columns") or feature_columns()
    return list(names)


@lru_cache(maxsize=1)
def _explainer():
    """Cache TreeExplainer (or fallback) for the loaded IsolationForest."""
    model = load_model()
    try:
        import shap

        try:
            return ("tree", shap.TreeExplainer(model))
        except Exception:
            background = _background_matrix(n=40)
            return ("generic", shap.Explainer(model.decision_function, background))
    except Exception:
        return ("permute", None)


def _background_matrix(n: int = 40) -> np.ndarray:
    path = ROOT / "data" / "aws_training.csv"
    cols = _feature_names()
    if path.exists():
        from features import features_matrix

        raw = pd.read_csv(path)
        _, X = features_matrix(raw)
        if len(X):
            return X[cols].to_numpy(dtype=float)[:n]
    return np.zeros((n, len(cols)))


def _pretty(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def _confidence_from_decision(decision: float) -> float:
    """Map IsolationForest decision_function to a 0–100 anomaly confidence."""
    conf = 100.0 / (1.0 + np.exp(CONFIDENCE_K * float(decision)))
    return float(np.clip(conf, 0.0, 100.0))


def _severity(confidence: float, anomaly: bool) -> str:
    if not anomaly:
        return "LOW"
    if confidence >= 80:
        return "HIGH"
    if confidence >= 60:
        return "MEDIUM"
    return "LOW"


def _sensor_health(confidence: float, anomaly: bool, anomaly_type: str) -> float:
    if anomaly_type == "missing":
        return 12.0
    if not anomaly:
        return float(np.clip(100.0 - 0.15 * confidence, 88.0, 100.0))
    return float(np.clip(100.0 - 0.78 * confidence, 8.0, 85.0))


def _row_meta(row: Mapping[str, Any] | pd.Series) -> tuple[str, str]:
    s = row if isinstance(row, pd.Series) else pd.Series(row)
    station = s.get("station_id", s.get("station", ""))
    ts = pd.to_datetime(s.get("timestamp"))
    return str(station), ts.isoformat()


def explain(model, feature_row, feature_names) -> dict[str, Any]:
    """Top-3 |SHAP| attributions and a plain-English sentence.

    Tries shap.TreeExplainer first (cached at module load). If TreeExplainer
    cannot explain this IsolationForest/sklearn combo, falls back to
    shap.Explainer on decision_function, then to a mean-impute permutation
    of each feature (same top-3 contract).
    """
    names = list(feature_names)
    x = np.asarray(feature_row, dtype=float).reshape(1, -1)
    x_df = pd.DataFrame(x, columns=names)
    kind, explainer = _explainer()
    values = None

    if kind == "tree" and explainer is not None:
        try:
            sv = explainer.shap_values(x_df, check_additivity=False)
            values = np.asarray(sv, dtype=float).reshape(-1)
        except Exception:
            values = None
    if values is None and kind == "generic" and explainer is not None:
        try:
            sv = explainer(x_df)
            values = np.asarray(getattr(sv, "values", sv), dtype=float).reshape(-1)
        except Exception:
            values = None
    if values is None:
        background = _background_matrix(n=1).reshape(-1)
        if background.shape[0] != x.shape[1]:
            background = np.nanmean(np.vstack([x, x]), axis=0)
        base = float(model.decision_function(x_df)[0])
        values = np.zeros(x.shape[1], dtype=float)
        for i in range(x.shape[1]):
            xp = x_df.copy()
            xp.iloc[0, i] = background[i] if i < len(background) else 0.0
            values[i] = base - float(model.decision_function(xp)[0])

    values = np.asarray(values, dtype=float).reshape(-1)[: len(names)]
    ranked = sorted(
        zip(names, values.tolist()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:3]
    top_name, top_val = ranked[0]
    sentence = (
        f"{_pretty(top_name)} was the primary reason this observation was flagged, "
        f"contributing {top_val:.4f}."
    )
    return {
        "top_features": [
            {"feature": n, "shap": float(v), "label": _pretty(n)} for n, v in ranked
        ],
        "reason": sentence,
    }


@lru_cache(maxsize=1)
def _lime_explainer():
    from lime.lime_tabular import LimeTabularExplainer

    background = _background_matrix(n=160)
    names = _feature_names()
    return LimeTabularExplainer(
        background,
        feature_names=names,
        class_names=["inlier", "anomaly"],
        mode="classification",
        discretize_continuous=False,
    )


def _if_anomaly_proba(X) -> np.ndarray:
    model = load_model()
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    cols = _feature_names()
    x_df = pd.DataFrame(arr, columns=cols)
    decision = model.decision_function(x_df)
    p_anom = 1.0 / (1.0 + np.exp(CONFIDENCE_K * decision))
    p_anom = np.clip(p_anom, 1e-6, 1.0 - 1e-6)
    return np.column_stack([1.0 - p_anom, p_anom])


def explain_lime(feature_row, feature_names) -> dict[str, Any]:
    """Top-3 LIME weights for the anomaly class and a plain-English sentence."""
    names = list(feature_names)
    x = np.asarray(feature_row, dtype=float).reshape(-1)
    try:
        explainer = _lime_explainer()
        exp = explainer.explain_instance(
            x,
            _if_anomaly_proba,
            num_features=3,
            labels=(1,),
            num_samples=180,
        )
        pairs = exp.as_list(label=1)
        ranked = []
        for label, weight in pairs:
            ranked.append((str(label), float(weight)))
        if not ranked:
            raise RuntimeError("empty LIME explanation")
        top_name, top_val = ranked[0]
        sentence = (
            f"LIME: {top_name} was the strongest local driver of the anomaly score "
            f"(weight {top_val:.4f})."
        )
        return {
            "top_features": [{"feature": n, "lime": float(v)} for n, v in ranked],
            "reason": sentence,
        }
    except Exception as exc:
        return {
            "top_features": [],
            "reason": f"LIME explanation unavailable ({exc}).",
        }


def attach_xai(result: dict[str, Any], history: pd.DataFrame) -> dict[str, Any]:
    """Add SHAP + LIME sentences onto a result using the station history frame."""
    out = dict(result)
    if history is None or len(history) == 0:
        out.setdefault("reason", TYPE_REASON["none"])
        out["lime_reason"] = "LIME needs history to engineer features."
        return out
    cols = _feature_names()
    engineered = engineer_features(history)
    feat = engineered.iloc[-1]
    x = feat[cols]
    if pd.isna(x).any():
        out["lime_reason"] = "LIME skipped — not enough rolling history yet."
        return out
    shap_ex = explain(load_model(), x, cols)
    lime_ex = explain_lime(x, cols)
    if out.get("anomaly"):
        out["reason"] = shap_ex["reason"]
    else:
        out["reason"] = shap_ex["reason"].replace(
            "was the primary reason this observation was flagged",
            "contributed most to the IsolationForest score",
        )
    out["lime_reason"] = lime_ex["reason"]
    out["shap_top_features"] = shap_ex.get("top_features", [])
    out["lime_top_features"] = lime_ex.get("top_features", [])
    return out


def _combine(history: pd.DataFrame | None, row: Mapping[str, Any] | pd.Series) -> pd.DataFrame:
    row_s = row if isinstance(row, pd.Series) else pd.Series(row)
    row_df = pd.DataFrame([row_s])
    if history is None or len(history) == 0:
        return normalize_station_frame(row_df)
    hist = normalize_station_frame(history)
    combined = pd.concat([hist, row_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.sort_values(["station", "timestamp"])
    combined = combined.drop_duplicates(subset=["station", "timestamp"], keep="last")
    return combined.reset_index(drop=True)


def _result_dict(
    *,
    station_id: str,
    timestamp: str,
    anomaly: bool,
    anomaly_score: float,
    anomaly_type: str,
    confidence: float,
    severity: str,
    sensor_health: float,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "station_id": station_id,
        "timestamp": timestamp,
        "anomaly": bool(anomaly),
        "anomaly_score": float(anomaly_score),
        "anomaly_type": anomaly_type,
        "confidence": float(round(confidence, 2)),
        "severity": severity,
        "sensor_health": float(round(sensor_health, 2)),
        "reason": reason,
        "lime_reason": extra.get("lime_reason") if extra else "LIME runs on the displayed observation.",
        "edge_model": EDGE_MODEL_ID,
    }
    if extra:
        payload.update(extra)
    return payload


def _healing_fields(
    row: Mapping[str, Any] | pd.Series,
    history: pd.DataFrame | None,
    network: pd.DataFrame | None,
    result: dict[str, Any],
    log: list[dict[str, Any]],
) -> dict[str, Any]:
    extra = suggest_values(
        row,
        history,
        network,
        anomaly=bool(result["anomaly"]),
        anomaly_type=str(result["anomaly_type"]),
    )
    extra.update(forecast_maintenance(log))
    return extra


def run_inference(
    row: Mapping[str, Any] | pd.Series,
    history: pd.DataFrame | None,
    network: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Score one reading. `history` is prior rows (and optionally later rows for typing).
    `network` is optional same-timestamp peer stations for spatial imputation.
    """
    model = load_model()
    cols = _feature_names()
    station_id, ts_iso = _row_meta(row)
    combined = _combine(history, row)
    engineered = engineer_features(combined)
    feat_row = engineered.iloc[-1]
    X = feat_row[cols]
    ready = not pd.isna(X).any()

    x_df = pd.DataFrame([X.to_numpy(dtype=float)], columns=cols) if ready else None
    if ready:
        decision = float(model.decision_function(x_df)[0])
        pred = int(model.predict(x_df)[0])
        anomaly = pred == -1
    else:
        decision = 0.0
        anomaly = False

    raw = row if isinstance(row, pd.Series) else pd.Series(row)
    rule_type = classify(raw, combined)
    if rule_type == "missing" or any(pd.isna(raw.get(c)) for c in SENSOR_COLS):
        anomaly = True
        atype = "missing"
    elif rule_type == "frozen":
        anomaly = True
        atype = "frozen"
    elif anomaly:
        atype = rule_type
    else:
        atype = "none"

    confidence = _confidence_from_decision(decision)
    if atype == "missing":
        confidence = max(confidence, 90.0)
        decision = min(decision, -0.15)
        anomaly = True
    elif atype == "frozen":
        confidence = max(confidence, 75.0)

    severity = _severity(confidence, anomaly)
    health = _sensor_health(confidence, anomaly, atype)
    # anomaly_score: higher means more anomalous (negated sklearn decision_function).
    anomaly_score = float(-decision)

    reason = TYPE_REASON.get(atype, TYPE_REASON["multivariate"])
    if ready and anomaly and atype != "missing":
        explained = explain(model, X, cols)
        reason = explained["reason"]
    elif not ready and atype != "missing":
        reason = "Insufficient history to engineer rolling features (need 6 samples)."

    result = _result_dict(
        station_id=station_id,
        timestamp=ts_iso,
        anomaly=anomaly,
        anomaly_score=anomaly_score,
        anomaly_type=atype,
        confidence=confidence,
        severity=severity,
        sensor_health=health,
        reason=reason,
    )
    edge = edge_infer(raw, combined)
    result["edge_anomaly"] = bool(edge["anomaly"])
    result["edge_anomaly_type"] = edge["anomaly_type"]
    result.update(
        _healing_fields(
            raw,
            combined,
            network if network is not None else combined,
            result,
            [result],
        )
    )
    return result


def run_inference_batch(
    df: pd.DataFrame,
    *,
    decision_threshold: float = 0.0,
    shap_event_limit: int = 5,
    network: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Vectorized IsolationForest scoring for a station (or mixed) frame.

    Feature engineering runs once. SHAP runs only for the latest scorable row
    plus up to `shap_event_limit` most-recent anomaly events.
    `decision_threshold` is applied to sklearn `decision_function` (default 0
    matches IsolationForest.predict). Higher sensitivity in the UI should pass
    a higher threshold (more negative scores still flag; a positive threshold
    flags more points).
    """
    model = load_model()
    cols = _feature_names()
    raw = normalize_station_frame(df).copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values(["station", "timestamp"]).reset_index(drop=True)

    engineered = engineer_features(raw)
    X = engineered[cols]
    ready_mask = ~X.isna().any(axis=1)
    n = len(raw)
    decision = np.zeros(n, dtype=float)
    if ready_mask.any():
        decision[ready_mask.to_numpy()] = model.decision_function(X.loc[ready_mask])

    missing_mask = raw[SENSOR_COLS].isna().any(axis=1).to_numpy()
    stuck_mask = frozen_mask(raw).to_numpy()
    anomaly_mask = (decision < decision_threshold) & ready_mask.to_numpy()
    anomaly_mask = anomaly_mask | missing_mask | stuck_mask

    shap_targets: set[int] = set()
    anom_idx = np.flatnonzero(anomaly_mask & ready_mask.to_numpy())
    for i in anom_idx[-shap_event_limit:]:
        shap_targets.add(int(i))
    ready_idx = np.flatnonzero(ready_mask.to_numpy())
    if len(ready_idx) and bool(anomaly_mask[int(ready_idx[-1])]):
        shap_targets.add(int(ready_idx[-1]))

    shap_cache: dict[int, str] = {}
    for i in shap_targets:
        if not ready_mask.iat[i]:
            continue
        explained = explain(model, X.iloc[i], cols)
        shap_cache[i] = explained["reason"]

    net = network if network is not None else raw
    empty_heal = {
        "suggested_temperature_c": None,
        "suggested_pressure_hpa": None,
        "suggested_humidity_pct": None,
        "impute_method": "none",
        "maintenance_status": "stable",
        "maintenance_days": None,
        "maintenance_message": "stable - no maintenance trend",
    }
    results: list[dict[str, Any]] = []
    daily_by_station: dict[str, dict] = {}
    for i in range(n):
        row = raw.iloc[i]
        station_id = str(row.get("station_id", row.get("station", "")))
        ts_iso = pd.Timestamp(row["timestamp"]).isoformat()
        is_anom = bool(anomaly_mask[i])
        if missing_mask[i]:
            rule = "missing"
            atype = "missing"
            is_anom = True
        elif stuck_mask[i]:
            rule = "frozen"
            atype = "frozen"
            is_anom = True
        elif is_anom:
            rule = classify(row, raw)
            atype = rule if rule != "none" else "multivariate"
        else:
            rule = "none"
            atype = "none"
        conf = _confidence_from_decision(float(decision[i]))
        if atype == "missing":
            conf = max(conf, 90.0)
        elif atype == "frozen":
            conf = max(conf, 75.0)
        severity = _severity(conf, is_anom)
        health = _sensor_health(conf, is_anom, atype)
        if is_anom and atype != "missing" and i in shap_cache:
            reason = shap_cache[i]
        else:
            reason = TYPE_REASON.get(atype, TYPE_REASON["none"])
        rec = _result_dict(
            station_id=station_id,
            timestamp=ts_iso,
            anomaly=is_anom,
            anomaly_score=float(-decision[i]),
            anomaly_type=atype,
            confidence=conf,
            severity=severity,
            sensor_health=health,
            reason=reason,
        )
        rec["edge_anomaly"] = rule != "none"
        rec["edge_anomaly_type"] = rule
        rec.update(empty_heal)
        if is_anom:
            rec.update(
                suggest_values(
                    row,
                    raw.iloc[: i + 1],
                    net,
                    anomaly=True,
                    anomaly_type=atype,
                )
            )
        daily_by_station.setdefault(station_id, {})
        rec.update(
            update_daily_forecast(
                daily_by_station[station_id],
                rec["timestamp"],
                bool(rec["anomaly"]),
                str(rec["anomaly_type"]),
            )
        )
        results.append(rec)
    return results
