#!/usr/bin/env python3
"""
predict_cloudburst.py
Simple script to load the trained model and score example input(s).

Usage example:
    python predict_cloudburst.py --example

This script demonstrates loading the model and running a single-row prediction.
"""

import joblib
import numpy as np
import pandas as pd

# Load model and encoder
model = joblib.load("lightgbm_cloudburst_model.pkl")
le = joblib.load("location_label_encoder.pkl")

# Example single-row input (replace with real sensor values)
example = {
    "location_id": "loc_1",
    "rainfall_mm": 5.2,
    "humidity": 82.1,
    "pressure": 1009.6,
    "radar_dbz": 28.0,
    "satellite_precip": 3.4,
    "rain_last_5min": 20.1,
    "rain_last_15min": 45.2,
    "pressure_drop_15min": 2.1,
    "humidity_change": 4.3,
    "forecast_30min": 12.5
}

# Build dataframe and encode location
df = pd.DataFrame([example])
df['location_id_enc'] = le.transform(df['location_id']) if example['location_id'] in le.classes_ else 0
feature_cols = [
    "location_id_enc", "rainfall_mm", "humidity", "pressure", "radar_dbz", "satellite_precip",
    "rain_last_5min", "rain_last_15min", "pressure_drop_15min", "humidity_change", "forecast_30min"
]
X = df[feature_cols].fillna(0)

proba = model.predict_proba(X)[:,1][0]
print(f"Predicted cloudburst probability: {proba:.4f}")
if proba >= 0.5:
    print("ALERT: HIGH RISK")
elif proba >= 0.2:
    print("Warning: Elevated risk")
else:
    print("Low risk")
