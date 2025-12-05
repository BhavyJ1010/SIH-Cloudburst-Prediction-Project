#!/usr/bin/env python3
"""
generate_dataset_v3.py
Creates synthetic dataset for cloudburst prediction containing:
 - rainfall, humidity, pressure, temperature, wind_gust
 - satellite_precip (synthetic realistic)
 - radar_dbz (synthetic realistic)
 - rolling features
 - cloudburst label

Output: synthetic_cloudburst_data_v3.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

NUM_LOCATIONS = 10
ROWS_PER_LOCATION = 500        # total 5000 rows
START_DATE = datetime(2024, 7, 1, 0, 0)
CLOUDBURST_RATE = 0.02         # 2% cloudburst events

rows = []

for loc_i in range(NUM_LOCATIONS):
    loc = f"loc_{loc_i+1}"
    timestamp = START_DATE
    
    for i in range(ROWS_PER_LOCATION):

        # Base random weather
        rainfall = np.random.uniform(0, 5)
        humidity = np.random.uniform(55, 80)
        pressure = np.random.uniform(1008, 1016)
        temperature = np.random.uniform(16, 32)
        wind_gust = np.random.uniform(2, 15)

        # Satellite synthetic (Option C)
        satellite_precip = (
            0.4 * rainfall +
            0.05 * humidity +
            0.2 * wind_gust +
            np.random.uniform(0, 1)
        )

        # Basic radar dBZ synthetic (Option C)
        radar_dbz = (
            15 +
            rainfall * 2 +
            wind_gust * 0.3 +
            np.random.uniform(0, 2)
        )

        # Inject storm patterns
        if np.random.random() < 0.10:
            rainfall += np.random.uniform(5, 10)
            humidity += np.random.uniform(5, 10)
            pressure -= np.random.uniform(1, 3)
            wind_gust += np.random.uniform(5, 12)
            satellite_precip += np.random.uniform(2, 4)
            radar_dbz += np.random.uniform(8, 12)

        # Rare cloudburst event
        cloudburst = 0
        if np.random.random() < CLOUDBURST_RATE:
            cloudburst = 1
            rainfall += np.random.uniform(20, 30)
            humidity += np.random.uniform(10, 15)
            pressure -= np.random.uniform(4, 7)
            wind_gust += np.random.uniform(10, 20)
            satellite_precip += np.random.uniform(5, 10)
            radar_dbz += np.random.uniform(15, 20)

        rows.append([
            timestamp, loc,
            round(rainfall, 3), round(humidity, 3), round(pressure, 3),
            round(temperature, 3), round(wind_gust, 3),
            round(satellite_precip, 3), round(radar_dbz, 3),
            cloudburst
        ])

        timestamp += timedelta(minutes=1)

df = pd.DataFrame(rows, columns=[
    "timestamp","location_id",
    "rainfall_mm","humidity","pressure","temperature","wind_gust",
    "satellite_precip","radar_dbz","cloudburst"
])

# Rolling features
df = df.sort_values(["location_id","timestamp"]).reset_index(drop=True)

df["rain_last_5min"] = df.groupby("location_id")["rainfall_mm"].rolling(5).sum().reset_index(0,drop=True).fillna(0)
df["rain_last_15min"] = df.groupby("location_id")["rainfall_mm"].rolling(15).sum().reset_index(0,drop=True).fillna(0)
df["pressure_drop_15min"] = -df.groupby("location_id")["pressure"].diff(15).fillna(0)
df["humidity_change"] = df.groupby("location_id")["humidity"].diff(15).fillna(0)
df["forecast_30min"] = df.groupby("location_id")["rainfall_mm"].shift(-30).rolling(30).mean().reset_index(0,drop=True).fillna(0)

df.to_csv("synthetic_cloudburst_data_v3.csv", index=False)
print("Dataset saved: synthetic_cloudburst_data_v3.csv")
print(df.head())
