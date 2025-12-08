#!/usr/bin/env python3
"""
predict_service_v3.py

Simple, robust continuous prediction service for SIH v3.

Features:
- Polls a CSV file (path in CONFIG) looking for new rows.
- Runs calibrated ensemble on newly appended rows (handles rolling feature computation).
- Keeps alerts ON for a hold-time (hysteresis) once risk >= HIGH/CRITICAL.
- Outputs JSON file with last prediction and optionally POSTs to a configured webhook.
- Avoids reprocessing the same row (stores last_processed_index in memory + optional file).

Run:
    venv\Scripts\activate
    python predict_service_v3.py --csv path/to/live.csv --poll-interval 10 --webhook http://frontend.example/api/push

Notes:
- Requires same artifacts created by training:
    calibrated_ensemble_v3.pkl
    location_label_encoder_v3.pkl
    thresholds_v3.json
    features_v3.txt (optional)
"""

import argparse
import json
import os
import time
import joblib
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# import predict function helpers from earlier prediction script
# If you kept the previous predict script in same folder, you can import; otherwise we reimplement minimal helpers here.
# For simplicity, we copy minimal helpers here so this file is self-contained.

CALIBRATED_ENSEMBLE_PATH = "calibrated_ensemble_v3.pkl"
ENCODER_PATH = "location_label_encoder_v3.pkl"
THRESHOLDS_PATH = "thresholds_v3.json"
OUTPUT_JSON = "predict_output.json"
LAST_PROC_STATE_FILE = "predict_service_state.json"  # stores last_processed_row index persistently (optional)

# Hysteresis config (tune these)
HOLD_TIME_MINUTES = 10           # once HIGH/CRITICAL triggered, keep alert ON for this many minutes unless cleared
RISK_CLEAR_MARGIN = 0.05         # margin below threshold before clearing (helps avoid flapping)

# Prediction windows used by rolling features (must match training assumptions)
ROLL_5 = 5
ROLL_15 = 15

# risk -> label mapping
def risk_label(prob):
    if prob < 0.10:
        return "LOW"
    if prob < 0.30:
        return "MODERATE"
    if prob < 0.60:
        return "HIGH"
    return "CRITICAL"

# Synthetic fallbacks (same logic as predict script)
def synthetic_radar_dbz(rainfall_mm):
    return float(10.0 * np.log10(max(0.01, rainfall_mm) + 1.0))

def synthetic_satellite_precip(rainfall_mm):
    return float(min(1.0, rainfall_mm / 20.0))

# compute rolling features (same function as before, simplified)
def compute_rolling_features(history_df, current_ts, location):
    hist_loc = history_df[(history_df["location_id"] == location) & (history_df["timestamp"] < current_ts)].copy()
    if hist_loc.empty:
        return {"rain_last_5min": 0.0, "rain_last_15min": 0.0, "pressure_drop_15min": 0.0, "humidity_change": 0.0}
    if hist_loc["timestamp"].dtype == object:
        hist_loc["timestamp"] = pd.to_datetime(hist_loc["timestamp"])
    t5 = current_ts - pd.Timedelta(minutes=ROLL_5)
    t15 = current_ts - pd.Timedelta(minutes=ROLL_15)
    last5 = hist_loc[hist_loc["timestamp"] >= t5]
    last15 = hist_loc[hist_loc["timestamp"] >= t15]
    rain_5 = float(last5["rainfall_mm"].sum()) if not last5.empty else 0.0
    rain_15 = float(last15["rainfall_mm"].sum()) if not last15.empty else 0.0
    if not last15.empty:
        pressure_15 = float(last15.iloc[0]["pressure"])
        humidity_15 = float(last15.iloc[0]["humidity"])
        pressure_current = float(history_df[(history_df["location_id"]==location) & (history_df["timestamp"]==current_ts)]["pressure"].iloc[0]) if ((history_df["location_id"]==location) & (history_df["timestamp"]==current_ts)).any() else float(last15.iloc[-1]["pressure"])
        humidity_current = float(history_df[(history_df["location_id"]==location) & (history_df["timestamp"]==current_ts)]["humidity"].iloc[0]) if ((history_df["location_id"]==location) & (history_df["timestamp"]==current_ts)).any() else float(last15.iloc[-1]["humidity"])
        pressure_drop = pressure_current - pressure_15
        humidity_change = humidity_current - humidity_15
    else:
        pressure_drop = float(hist_loc.iloc[-1]["pressure"]) - float(hist_loc.iloc[0]["pressure"])
        humidity_change = float(hist_loc.iloc[-1]["humidity"]) - float(hist_loc.iloc[0]["humidity"])
    return {"rain_last_5min": rain_5, "rain_last_15min": rain_15, "pressure_drop_15min": float(pressure_drop), "humidity_change": float(humidity_change)}

# prepare single-row model input (same logic)
def prepare_model_input(row, full_df, le, thresholds):
    ts = pd.to_datetime(row["timestamp"])
    loc = str(row["location_id"])
    rainfall = float(row.get("rainfall_mm", 0.0) if not pd.isna(row.get("rainfall_mm", np.nan)) else 0.0)
    humidity = float(row.get("humidity", 0.0) if not pd.isna(row.get("humidity", np.nan)) else 0.0)
    pressure = float(row.get("pressure", 0.0) if not pd.isna(row.get("pressure", np.nan)) else 0.0)
    temperature = float(row.get("temperature", 0.0) if not pd.isna(row.get("temperature", np.nan)) else 0.0)
    wind_gust = float(row.get("wind_gust", 0.0) if not pd.isna(row.get("wind_gust", np.nan)) else 0.0)

    satellite_precip = float(row["satellite_precip"]) if ("satellite_precip" in row and not pd.isna(row["satellite_precip"])) else synthetic_satellite_precip(rainfall)
    radar_dbz = float(row["radar_dbz"]) if ("radar_dbz" in row and not pd.isna(row["radar_dbz"])) else synthetic_radar_dbz(rainfall)

    # rolling
    rf = compute_rolling_features(full_df, ts, loc)
    rain_last_5min = float(row.get("rain_last_5min", rf["rain_last_5min"]))
    rain_last_15min = float(row.get("rain_last_15min", rf["rain_last_15min"]))
    pressure_drop_15min = float(row.get("pressure_drop_15min", rf["pressure_drop_15min"]))
    humidity_change = float(row.get("humidity_change", rf["humidity_change"]))

    forecast_30min = float(row.get("forecast_30min", min(100.0, (rain_last_15min / max(1, ROLL_15)) * 30.0)))

    try:
        location_enc = int(le.transform([loc])[0])
    except Exception:
        location_enc = 0

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
    return model_input, {"timestamp": str(ts), "location_id": loc}

# ensemble predict
def ensemble_predict(models, X_df):
    probs = np.zeros((len(X_df), len(models)))
    for i, m in enumerate(models):
        probs[:, i] = m.predict_proba(X_df)[:, 1]
    return probs.mean(axis=1)

# persistent small state
def load_state():
    if os.path.exists(LAST_PROC_STATE_FILE):
        try:
            return json.load(open(LAST_PROC_STATE_FILE, "r"))
        except Exception:
            return {}
    return {}

def save_state(state):
    json.dump(state, open(LAST_PROC_STATE_FILE, "w"), indent=2)

# main service loop
def run_service(csv_path, poll_interval, webhook_url=None, save_json=True, hold_minutes=HOLD_TIME_MINUTES):
    # load models & encoder & thresholds
    models = joblib.load(CALIBRATED_ENSEMBLE_PATH)
    le = joblib.load(ENCODER_PATH)
    thresholds = json.load(open(THRESHOLDS_PATH, "r"))
    threshold = float(thresholds.get("best_f1_threshold", 0.5))

    state = load_state()
    last_idx = state.get("last_processed_index", None)
    alert_active = state.get("alert_active", False)
    alert_until = pd.to_datetime(state.get("alert_until")) if state.get("alert_until") else None

    print(f"Starting predict service. Watching CSV: {csv_path}")
    print(f"Polling every {poll_interval}s. Webhook: {webhook_url}. Hold-minutes: {hold_minutes}. Alert-threshold: {threshold}")

    while True:
        try:
            if not os.path.exists(csv_path):
                print(f"[{datetime.now()}] CSV not found: {csv_path}. Waiting...")
                time.sleep(poll_interval)
                continue

            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
            if len(df) == 0:
                time.sleep(poll_interval)
                continue

            newest_idx = len(df) - 1
            if last_idx is None:
                # on first run, process last row only
                process_idx = newest_idx
            else:
                # only process new appended rows (if any)
                if newest_idx <= last_idx:
                    # nothing new
                    time.sleep(poll_interval)
                    continue
                # process each new row in order (you can change to batch if you want)
                process_idx = last_idx + 1

            # process rows until newest_idx
            while process_idx <= newest_idx:
                row = df.iloc[process_idx]
                model_input, meta = prepare_model_input(row, df, le, thresholds)
                # align columns exactly to training features
                cols_expected = [
                    "location_id_enc", "rainfall_mm", "humidity", "pressure", "temperature", "wind_gust",
                    "satellite_precip", "radar_dbz",
                    "rain_last_5min", "rain_last_15min",
                    "pressure_drop_15min", "humidity_change",
                    "forecast_30min"
                ]
                model_input = model_input[cols_expected]

                avg_prob = float(ensemble_predict(models, model_input)[0])
                label = risk_label(avg_prob)

                now = datetime.utcnow()
                # decide alert activation/clearing with hysteresis
                # If already active, don't clear until alert_until or until prob falls sufficiently low
                if alert_active:
                    # if alert_until set, check expiration first
                    if alert_until and now >= alert_until:
                        # allow clearing only if prob < (threshold - margin)
                        if avg_prob < (threshold - RISK_CLEAR_MARGIN):
                            alert_active = False
                            alert_until = None
                    else:
                        # extend alert period if still high
                        if avg_prob >= threshold:
                            alert_until = now + timedelta(minutes=hold_minutes)
                else:
                    # not active -> trigger if prob >= threshold
                    if avg_prob >= threshold:
                        alert_active = True
                        alert_until = now + timedelta(minutes=hold_minutes)

                out = {
                    "timestamp": meta["timestamp"],
                    "location_id": meta["location_id"],
                    "probability": round(avg_prob, 6),
                    "risk_level": label,
                    "alert": int(alert_active),
                    "used_threshold": threshold,
                    "alert_until": str(alert_until) if alert_until else None,
                    "processed_index": int(process_idx)
                }

                # write output json
                if save_json:
                    with open(OUTPUT_JSON, "w") as f:
                        json.dump(out, f, indent=2)

                # optionally post to webhook
                if webhook_url:
                    try:
                        resp = requests.post(webhook_url, json=out, timeout=5)
                        print(f"[{datetime.utcnow()}] POST -> {webhook_url} status {resp.status_code}")
                    except Exception as e:
                        print(f"[{datetime.utcnow()}] Failed to POST to webhook: {e}")

                print(f"[{datetime.utcnow()}] Processed row {process_idx} | loc={meta['location_id']} prob={out['probability']} risk={label} alert={out['alert']}")

                # update state
                last_idx = process_idx
                state["last_processed_index"] = last_idx
                state["alert_active"] = alert_active
                state["alert_until"] = str(alert_until) if alert_until else None
                save_state(state)

                process_idx += 1

            # end inner while
        except Exception as e:
            print(f"[{datetime.utcnow()}] ERROR in service loop: {e}")
        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run continuous predict service (v3).")
    parser.add_argument("--csv", required=True, help="Path to live CSV file produced by hardware/db.")
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between polls (default 10).")
    parser.add_argument("--webhook", default=None, help="Optional: webhook URL to POST JSON outputs.")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON output to disk.")
    parser.add_argument("--hold-minutes", type=int, default=HOLD_TIME_MINUTES, help="How long to hold alerts once triggered.")
    args = parser.parse_args()

    run_service(args.csv, args.poll_interval, webhook_url=args.webhook, save_json=not args.no_save, hold_minutes=args.hold_minutes)
