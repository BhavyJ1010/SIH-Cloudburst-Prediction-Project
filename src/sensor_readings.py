import sqlite3

conn = sqlite3.connect("sensor_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    location_id TEXT,
    rainfall_mm REAL,
    humidity REAL,
    pressure REAL,
    temperature REAL,
    wind_gust REAL,
    satellite_precip REAL,
    radar_dbz REAL,
    cloudburst INTEGER,
    rain_last_5min REAL,
    rain_last_15min REAL,
    pressure_drop_5min REAL,
    pressure_drop_15min REAL,
    humidity_change REAL,
    forecast_30min REAL
)
""")

conn.commit()
conn.close()

print("Database with ALL parameters created successfully!")

import sqlite3
import pandas as pd
import time

CSV_PATH = "sensor_readings.csv"
DB_PATH = "sensor_data.db"

def load_last_position():
    try:
        with open("last_pos.txt", "r") as f:
            return int(f.read().strip())
    except:
        return 0

def save_last_position(pos):
    with open("last_pos.txt", "w") as f:
        f.write(str(pos))


while True:
    print("Checking for new data...")

    df = pd.read_csv(CSV_PATH)

    last_pos = load_last_position()
    new_rows = df.iloc[last_pos:]

    if not new_rows.empty:
        print(f"Inserting {len(new_rows)} new rows into database...")

        conn = sqlite3.connect(DB_PATH)
        new_rows.to_sql("sensor_readings", conn, if_exists="append", index=False)
        conn.close()

        save_last_position(len(df))
    else:
        print("No new data found.")

    print("Waiting for next update...\n")
    time.sleep(60)
