import sqlite3

DB_PATH = "pi_cloudburst.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
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
            pressure_drop_15min REAL,
            humidity_change REAL,
            forecast_30min REAL,
            created_at TEXT
        )
        """)
        conn.commit()
