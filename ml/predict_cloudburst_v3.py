#!/usr/bin/env python3
"""
predict_cloudburst_v3.py

Usage:
    # Predict for the last row in the CSV:
    python predict_cloudburst_v3.py --csv synthetic_cloudburst_data_v3.csv

    # Predict for a specific row index (0-based):
    python predict_cloudburst_v3.py --csv synthetic_cloudburst_data_v3.csv --row 4999

    # Or call predict_row(...) from other Python code and pass a pandas Series + history DataFrame.

What it does:
 - Loads calibrated ensemble and label encoder created by train_cloudburst_model_v3.py
 - Computes missing rolling features from history rows (if needed)
 - Provides synthetic fallbacks for radar_dbz and satellite_precip if missing
 - Averages calibrated model probabilities and applies best-F1 threshold to set alert
 - Prints and optionally saves JSON output
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta

# ---------- CONFIG ----------
CALIBRATED_ENSEMBLE_PATH = "calibrated_ensemble_v3.pkl"
ENCODER_PATH = "location_label_encoder_v3.pkl"
THRESHOLDS_PATH = "thresholds_v3.json"
FEATURE_ORDER_PATH = "features_v3.txt"  # optional, not required but helpful
OUTPUT_JSON = "predict_output.json"

# Which threshold to use for alert: we use best_f1 from thresholds_v3.json (Option A)
THRESHOLD_KEY = "best_f1_threshold"

# Rolling windows (minutes)
ROLL_5 = 5
ROLL_15 = 15

# ---------- UTILITIES ----------

def load_artifacts():
    if not os.path.exists(CALIBRATED_ENSEMBLE_PATH):
        raise FileNotFoundError(f"Missing {CALIBRATED_ENSEMBLE_PATH}. Run training first.")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Missing {ENCODER_PATH}. Run training first.")
    if not os.path.exists(THRESHOLDS_PATH):
        raise FileNotFoundError(f"Missing {THRESHOLDS_PATH}. Run training first.")

    models = joblib.load(CALIBRATED_ENSEMBLE_PATH)   # list of CalibratedClassifierCV
    le = joblib.load(ENCODER_PATH)
    with open(THRESHOLDS_PATH, "r") as f:
        thresholds = json.load(f)
    return models, le, thresholds

def synthetic_radar_dbz(rainfall_mm):
    """
    Deterministic synthetic estimate for radar_dbz when radar is absent.
    We use a monotonic mapping: DBZ roughly scales with log10 of rain rate.
    Output in dBZ scale-like float (not meteorologically perfect).
    """
    # safe mapping: small rainfall -> small dbz, larger rainfall -> larger dbz
    return float(10.0 * np.log10(max(0.01, rainfall_mm) + 1.0))

def synthetic_satellite_precip(rainfall_mm):
    """
    Estimate satellite precipitation intensity in 0..1 from rainfall_mm.
    Deterministic scaling capped at 1.0.
    """
    return float(min(1.0, rainfall_mm / 20.0))

def compute_rolling_features(history_df, current_ts, location, rainfall_col="rainfall_mm", pressure_col="pressure", humidity_col="humidity"):
    """
    history_df: DataFrame containing prior rows (same CSV) ideally sorted by timestamp asc
    current_ts: pd.Timestamp for current sample
    location: location_id string
    Returns: dict of rolling features:
        rain_last_5min, rain_last_15min, pressure_drop_15min, humidity_change
    """
    # filter same location and timestamps strictly before current_ts
    hist_loc = history_df[(history_df["location_id"] == location) & (history_df["timestamp"] < current_ts)].copy()
    if hist_loc.empty:
        return {"rain_last_5min": 0.0, "rain_last_15min": 0.0, "pressure_drop_15min": 0.0, "humidity_change": 0.0}

    # ensure timestamp dtype
    if hist_loc["timestamp"].dtype == object:
        hist_loc["timestamp"] = pd.to_datetime(hist_loc["timestamp"])

    # last 5 and 15 minutes windows
    t5 = current_ts - pd.Timedelta(minutes=ROLL_5)
    t15 = current_ts - pd.Timedelta(minutes=ROLL_15)

    last5 = hist_loc[hist_loc["timestamp"] >= t5]
    last15 = hist_loc[hist_loc["timestamp"] >= t15]

    rain_5 = float(last5[rainfall_col].sum()) if not last5.empty else 0.0
    rain_15 = float(last15[rainfall_col].sum()) if not last15.empty else 0.0

    # pressure_drop_15min = current_pressure - pressure 15 min ago (negative drop => decrease)
    pressure_drop = 0.0
    humidity_change = 0.0
    # approximate pressure 15min ago as the earliest value in last15 (if exists) otherwise compare to last available
    if not last15.empty:
        # find value closest to t15 (earliest in window)
        pressure_15 = float(last15.iloc[0][pressure_col])
        humidity_15 = float(last15.iloc[0][humidity_col]) if humidity_col in last15.columns else float(last15.iloc[0].get(humidity_col, 0.0))
        # we will subtract (current - old)
        pressure_current = float(history_df[(history_df["location_id"]==location) & (history_df["timestamp"]==current_ts)][pressure_col].iloc[0]) if ((history_df["location_id"]==location) & (history_df["timestamp"]==current_ts)).any() else float(last15.iloc[-1][pressure_col])
        humidity_current = float(history_df[(history_df["location_id"]==location) & (history_df["timestamp"]==current_ts)][humidity_col].iloc[0]) if ((history_df["location_id"]==location) & (history_df["timestamp"]==current_ts)).any() else float(last15.iloc[-1][humidity_col])
        pressure_drop = pressure_current - pressure_15
        humidity_change = humidity_current - humidity_15
    else:
        # fallback: compare to last available record
        if not hist_loc.empty:
            pressure_drop = float(hist_loc.iloc[-1][pressure_col]) - float(hist_loc.iloc[0][pressure_col])  # crude
            humidity_change = float(hist_loc.iloc[-1][humidity_col]) - float(hist_loc.iloc[0][humidity_col])

    return {
        "rain_last_5min": rain_5,
        "rain_last_15min": rain_15,
        "pressure_drop_15min": float(pressure_drop),
        "humidity_change": float(humidity_change)
    }

def prepare_input_row(row, history_df, le, thresholds):
    """
    row: pandas Series (the row to predict) - should include timestamp & location_id
    history_df: full DataFrame used to compute rolling features (including the row)
    le: label encoder to transform location_id
    thresholds: dict loaded from thresholds_v3.json
    Returns: (model_input_df, meta) where model_input_df is a DataFrame with columns matching features,
             and meta contains helpful info (row_index, timestamp, location, threshold used)
    """

    # required fields & defaults
    ts = pd.to_datetime(row["timestamp"])
    location = str(row["location_id"])

    # copy values or set defaults when missing
    rainfall = float(row.get("rainfall_mm", 0.0) if not pd.isna(row.get("rainfall_mm", np.nan)) else 0.0)
    humidity = float(row.get("humidity", 0.0) if not pd.isna(row.get("humidity", np.nan)) else 0.0)
    pressure = float(row.get("pressure", 0.0) if not pd.isna(row.get("pressure", np.nan)) else 0.0)
    temperature = float(row.get("temperature", 0.0) if not pd.isna(row.get("temperature", np.nan)) else 0.0)
    wind_gust = float(row.get("wind_gust", 0.0) if not pd.isna(row.get("wind_gust", np.nan)) else 0.0)

    # satellite & radar: use given if present else synthetic fallback
    if "satellite_precip" in row and not pd.isna(row["satellite_precip"]):
        satellite_precip = float(row["satellite_precip"])
    else:
        satellite_precip = synthetic_satellite_precip(rainfall)

    if "radar_dbz" in row and not pd.isna(row["radar_dbz"]):
        radar_dbz = float(row["radar_dbz"])
    else:
        radar_dbz = synthetic_radar_dbz(rainfall)

    # rolling features: if present in row, use them; otherwise compute from history
    if "rain_last_5min" in row and not pd.isna(row["rain_last_5min"]):
        rain_last_5min = float(row["rain_last_5min"])
    else:
        rf = compute_rolling_features(history_df, ts, location)
        rain_last_5min = rf["rain_last_5min"]

    if "rain_last_15min" in row and not pd.isna(row["rain_last_15min"]):
        rain_last_15min = float(row["rain_last_15min"])
    else:
        rain_last_15min = rf["rain_last_15min"] if 'rf' in locals() else compute_rolling_features(history_df, ts, location)["rain_last_15min"]

    if "pressure_drop_15min" in row and not pd.isna(row["pressure_drop_15min"]):
        pressure_drop_15min = float(row["pressure_drop_15min"])
    else:
        pressure_drop_15min = rf["pressure_drop_15min"] if 'rf' in locals() else compute_rolling_features(history_df, ts, location)["pressure_drop_15min"]

    if "humidity_change" in row and not pd.isna(row["humidity_change"]):
        humidity_change = float(row["humidity_change"])
    else:
        humidity_change = rf["humidity_change"] if 'rf' in locals() else compute_rolling_features(history_df, ts, location)["humidity_change"]

    # forecast_30min: simple synthetic estimate = rainfall in next 30min predicted from recent trend
    if "forecast_30min" in row and not pd.isna(row["forecast_30min"]):
        forecast_30min = float(row["forecast_30min"])
    else:
        # naive predictor: average of last 15 minutes * 2 (as 30min ahead) capped
        avg_recent = rain_last_15min / max(1, ROLL_15)
        forecast_30min = float(min(100.0, avg_recent * 30.0))

    # encode location
    try:
        location_enc = int(le.transform([location])[0])
    except Exception:
        # unseen location: add new mapping (append)
        # NOTE: LabelEncoder doesn't support incremental add easily; fallback: map unseen to -1
        # but LightGBM expects numeric; using -1 is acceptable if model saw similar mapping during training? It may not.
        # Safer: map unseen to 0 and warn.
        location_enc = 0
        print(f"WARNING: location_id '{location}' not found in encoder. Using location_id_enc=0 as fallback.")

    # Build model input (single-row dataframe)
    model_input = pd.DataFrame([{
        "location_id_enc": location_enc,
        "rainfall_mm": rainfall,
        "humidity": humidity,
        "pressure": pressure,
        "temperature": temperature,
        "wind_gust": wind_gust,
        "satellite_precip": satellite_precip,
        "radar_dbz": radar_dbz,
        "rain_last_5min": rain_last_5min,
        "rain_last_15min": rain_last_15min,
        "pressure_drop_15min": pressure_drop_15min,
        "humidity_change": humidity_change,
        "forecast_30min": forecast_30min
    }])

    meta = {
        "timestamp": str(ts),
        "location_id": location,
        "threshold_used": float(thresholds.get(THRESHOLD_KEY, 0.5))
    }

    return model_input, meta

def ensemble_predict(models, X_df):
    """
    models: list of calibrated sklearn estimators (CalibratedClassifierCV)
    X_df: DataFrame with one or more rows
    Returns: averaged probability for positive class
    """
    probs = np.zeros((len(X_df), len(models)))
    for i, m in enumerate(models):
        probs[:, i] = m.predict_proba(X_df)[:, 1]
    avg = probs.mean(axis=1)
    return avg

def risk_label(prob):
    """
    Risk levels:
      prob < 0.10 => LOW
      0.10 <= prob < 0.30 => MODERATE
      0.30 <= prob < 0.60 => HIGH
      prob >= 0.60 => CRITICAL
    (These are heuristics and can be tuned.)
    """
    if prob < 0.10:
        return "LOW"
    if prob < 0.30:
        return "MODERATE"
    if prob < 0.60:
        return "HIGH"
    return "CRITICAL"

# ---------- MAIN FLOW ----------

def predict_from_csv(csv_path, row_index=None, save_json=False):
    # load artifacts
    models, le, thresholds = load_artifacts()

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    if row_index is None:
        row_index = len(df) - 1
    if row_index < 0 or row_index >= len(df):
        raise IndexError("row_index out of range")

    row = df.iloc[row_index]
    # pass full df as history for rolling computation
    model_input_df, meta = prepare_input_row(row, df, le, thresholds)

    # Align input columns to what models expect (if features_v3.txt provided)
    # Models were trained with specific columns; ensure order
    cols_expected = [
        "location_id_enc", "rainfall_mm", "humidity", "pressure", "temperature", "wind_gust",
        "satellite_precip", "radar_dbz",
        "rain_last_5min", "rain_last_15min",
        "pressure_drop_15min", "humidity_change",
        "forecast_30min"
    ]
    model_input_df = model_input_df[cols_expected]

    avg_prob = float(ensemble_predict(models, model_input_df)[0])

    # choose threshold (best F1 from thresholds)
    threshold = float(thresholds.get(THRESHOLD_KEY, 0.5))

    alert_flag = 1 if avg_prob >= threshold else 0
    label = risk_label(avg_prob)

    out = {
        "timestamp": meta["timestamp"],
        "location_id": meta["location_id"],
        "probability": round(avg_prob, 6),
        "risk_level": label,
        "alert": int(alert_flag),
        "used_threshold": threshold
    }

    print("\n------ Prediction Result ------")
    print(json.dumps(out, indent=2))
    print("--------------------------------\n")

    if save_json:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved prediction to {OUTPUT_JSON}")

    return out

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Predict cloudburst from latest CSV row (version 3).")
    p.add_argument("--csv", required=True, help="Path to CSV file (live hardware CSV)")
    p.add_argument("--row", type=int, default=None, help="Row index to predict (0-based). Default: last row")
    p.add_argument("--save", action="store_true", help="Save output JSON to disk")
    args = p.parse_args()

    result = predict_from_csv(args.csv, row_index=args.row, save_json=args.save)
    