from fastapi import FastAPI
from pydantic import BaseModel
from database import get_conn, init_db
from datetime import datetime

app = FastAPI()

# Initialize database when API starts
init_db()


# -----------------------------
# Pydantic model for JSON body
# -----------------------------
class SensorData(BaseModel):
    timestamp: str
    location_id: str
    rainfall_mm: float
    humidity: float
    pressure: float
    temperature: float
    wind_gust: float
    satellite_precip: float
    radar_dbz: float
    cloudburst: int
    rain_last_5min: float
    rain_last_15min: float
    pressure_drop_15min: float
    humidity_change: float
    forecast_30min: float


# -----------------------------
# POST /ingest → Store Data
# -----------------------------
@app.post("/ingest")
async def ingest(data: SensorData):

    data = data.dict()  # Convert Pydantic object to standard dict

    with get_conn() as conn:
        c = conn.cursor()

        c.execute("""
            INSERT INTO sensor_data (
                timestamp, location_id, rainfall_mm, humidity, pressure, temperature,
                wind_gust, satellite_precip, radar_dbz, cloudburst,
                rain_last_5min, rain_last_15min, pressure_drop_15min,
                humidity_change, forecast_30min, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["timestamp"],
            data["location_id"],
            data["rainfall_mm"],
            data["humidity"],
            data["pressure"],
            data["temperature"],
            data["wind_gust"],
            data["satellite_precip"],
            data["radar_dbz"],
            data["cloudburst"],
            data["rain_last_5min"],
            data["rain_last_15min"],
            data["pressure_drop_15min"],
            data["humidity_change"],
            data["forecast_30min"],
            datetime.utcnow().isoformat()
        ))

        conn.commit()

    return {"status": "success", "message": "Data stored in database"}


# -----------------------------
# GET /latest → Fetch last row
# -----------------------------
@app.get("/latest")
async def latest():

    with get_conn() as conn:
        c = conn.cursor()

        row = c.execute(
            "SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1"
        ).fetchone()

    if not row:
        return {"error": "No data available"}

    columns = [
        "id", "timestamp", "location_id", "rainfall_mm", "humidity", "pressure",
        "temperature", "wind_gust", "satellite_precip", "radar_dbz",
        "cloudburst", "rain_last_5min", "rain_last_15min",
        "pressure_drop_15min", "humidity_change", "forecast_30min", "created_at"
    ]

    return dict(zip(columns, row))


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
async def home():
    return {"status": "Cloudburst API Running", "version": "1.0"}
