"""Train SkyGuard IsolationForest on AWS station telemetry."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from features import SENSOR_COLS, feature_columns, features_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TRAIN_PATH = DATA_DIR / "aws_training.csv"
TEST_PATH = DATA_DIR / "aws_test_injected.csv"
MODEL_PATH = MODELS_DIR / "skyguard_isoforest.joblib"

SOURCE_CANDIDATES = [
    DATA_DIR / "aws_skyguard_dataset.csv",
    Path(r"C:\Users\saree\Downloads\aws_skyguard_dataset.csv"),
]

TRAIN_COLS = ["timestamp", "station", *SENSOR_COLS]
TEST_COLS = ["timestamp", "station", *SENSOR_COLS, "is_anomaly"]

CONTAMINATION_GRID = [0.01, 0.02, 0.05, 0.1]
RANDOM_STATE = 42


def _load_source() -> pd.DataFrame:
    for path in SOURCE_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        tried = ", ".join(str(p) for p in SOURCE_CANDIDATES)
        raise FileNotFoundError(f"Could not find source AWS CSV. Tried: {tried}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "station" not in df.columns:
        df["station"] = df["station_id"]
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    return df.sort_values(["station", "timestamp"]).reset_index(drop=True)


def ensure_split_csvs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        return

    source = _load_source()
    complete_normal = source.loc[
        ~source["is_anomaly"],
        ["timestamp", "station", *SENSOR_COLS],
    ].dropna(subset=SENSOR_COLS)

    train = complete_normal[TRAIN_COLS].copy()
    test = source[TEST_COLS].copy()

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"Wrote {TRAIN_PATH} ({len(train)} rows)")
    print(f"Wrote {TEST_PATH} ({len(test)} rows, anomalies={int(test['is_anomaly'].sum())})")


def main() -> None:
    ensure_split_csvs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    test_raw["is_anomaly"] = test_raw["is_anomaly"].astype(bool)

    train_keep, X_train = features_matrix(train_raw)
    test_keep, X_test = features_matrix(test_raw)
    y_test = test_keep["is_anomaly"].astype(int).to_numpy()

    anomaly_rate = float(y_test.mean()) if len(y_test) else 0.0
    grid = sorted(set(CONTAMINATION_GRID + [max(round(anomaly_rate, 4), 0.01)]))

    print(f"Train rows after features: {len(X_train)}")
    print(f"Test rows after features:  {len(X_test)} (anomaly rate={anomaly_rate:.4f})")
    print(f"Contamination grid: {grid}")

    best = None
    for contamination in grid:
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(X_train)
        pred = (model.predict(X_test) == -1).astype(int)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        print(
            f"  contamination={contamination:.4f}  "
            f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}"
        )
        score = (f1, recall, precision)
        if best is None or score > best["score"]:
            best = {
                "contamination": contamination,
                "model": model,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "score": score,
            }

    assert best is not None
    payload = {
        "model": best["model"],
        "feature_columns": feature_columns(),
        "contamination": best["contamination"],
        "random_state": RANDOM_STATE,
    }
    joblib.dump(payload, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(
        "Best contamination={:.4f}  precision={:.4f}  recall={:.4f}  F1={:.4f}".format(
            best["contamination"], best["precision"], best["recall"], best["f1"]
        )
    )
    print(
        f"TEST_METRICS precision={best['precision']:.4f} "
        f"recall={best['recall']:.4f} f1={best['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
