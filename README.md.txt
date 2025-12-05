# 🌧️ SIH Cloudburst Prediction Project — Machine Learning Pipeline (V3)

This repository contains all frontend, backend, hardware PI code and the **Machine Learning Cloudburst Prediction Pipeline (Version 3)** developed for Smart India Hackathon 2025.

---

## 📌 Folder Structure

SIH_clean/
│
├── cloudburst_analysis/        → Exploratory data analysis by ML analysis team  
├── PI code/                    → Raspberry Pi sensor reading + data logging  
├── src/                        → Frontend (React)  
│
└── ml/                         → Machine Learning V3 (Bhavy Jain)
     ├── generate_dataset_v3.py  
     ├── train_cloudburst_model_v3.py  
     ├── predict_cloudburst_v3.py  
     ├── predict_service_v3.py  
     ├── requirements.txt  
     └── models/
         ├── calibrated_ensemble_v3.pkl  
         ├── location_label_encoder_v3.pkl  
         ├── thresholds_v3.json  
         └── features_v3.txt  

---

## 🚀 ML V3 Components

### 1️⃣ generate_dataset_v3.py  
Creates a **synthetic weather dataset** including:
- rainfall  
- humidity  
- pressure  
- temperature  
- wind gust  
- satellite precip (synthetic fallback)  
- radar dbz (synthetic fallback)  
- rolling features  
- cloudburst labels  

Run: python ml/generate_dataset_v3.py

---

### 3️⃣ predict_cloudburst_v3.py  
Runs a **single prediction** from:
- a live CSV file  
- or manual values  

Run: python ml/train_cloudburst_model_v3.py

---

### 3️⃣ predict_cloudburst_v3.py  
Runs a **single prediction** from:
- a live CSV file  
- or manual values  

Run: python ml/predict_cloudburst_v3.py --csv path_to_file.csv

Example output:
{
"timestamp": "2024-07-01 08:19:00",
"location_id": "loc_9",
"probability": 0.000775,
"risk_level": "LOW",
"alert": 0,
"used_threshold": 0.3153
}

---

### 4️⃣ predict_service_v3.py  
This is the **real-time prediction engine** used during deployment.

It:
- Reads the live CSV file every few seconds  
- Predicts risk using calibrated ensemble  
- Applies **hysteresis** (HIGH risk persists for a while to avoid rapid drop)  
- Can push output to API / Database / Hardware  

Run: python ml/predict_service_v3.py --csv live_data.csv --interval 5

---

## 🛰️ Required CSV Format (from Hardware Team)

The system expects a CSV file with columns in this exact order:

timestamp
location_id
rainfall_mm
humidity
pressure
temperature
wind_gust
satellite_precip (optional — filled synthetically if missing)
radar_dbz (optional — filled synthetically if missing)
rain_last_5min
rain_last_15min
pressure_drop_15min
humidity_change
forecast_30min


Hardware team generates this file continuously.

---

## 🎯 Output (to API / Frontend / Alerts)

The model produces:

- probability  
- risk level (LOW / MODERATE / HIGH / CRITICAL)  
- alert flag (0 or 1)  

Used by:
- Frontend dashboard  
- Mobile app  
- IoT alert system (LED/Siren)  
- Database logging  

---






