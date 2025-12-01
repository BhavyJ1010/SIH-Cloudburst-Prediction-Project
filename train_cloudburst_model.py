#!/usr/bin/env python3
"""
train_cloudburst_model.py
Simple, beginner-friendly LightGBM training script for the synthetic cloudburst dataset.

Usage:
    python train_cloudburst_model.py

Outputs:
    - lightgbm_cloudburst_model.pkl   (saved model)
    - location_label_encoder.pkl      (label encoder for location_id)
    - prints validation AUC and Precision@K
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import os

# 1) Load dataset (update path if needed)
DATA_PATH = "synthetic_cloudburst_data.csv"  # put this file in the same folder as this script
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# 2) Basic sanity
print("Rows:", len(df), "Locations:", df['location_id'].nunique())
df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

# 3) Encode location_id (simple label encoding)
le = LabelEncoder()
df['location_id_enc'] = le.fit_transform(df['location_id'])

# 4) Features and target
feature_cols = [
    "location_id_enc",
    "rainfall_mm",
    "humidity",
    "pressure",
    "radar_dbz",
    "satellite_precip",
    "rain_last_5min",
    "rain_last_15min",
    "pressure_drop_15min",
    "humidity_change",
    "forecast_30min",
]
target_col = "cloudburst"

X = df[feature_cols].fillna(0)
y = df[target_col].astype(int)

# 5) Time-based split (80% train, 20% val)
split_idx = int(len(df) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print("Train samples:", len(X_train), "Val samples:", len(X_val))
print("Positive examples in train:", np.sum(y_train), "in val:", np.sum(y_val))

# 6) LightGBM baseline model
model = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.05,
    n_estimators=600,
    num_leaves=40,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    scale_pos_weight=(len(y_train) - y_train.sum()) / max(1, y_train.sum()),
    random_state=42,
    n_jobs=-1
)

print("Training LightGBM... (this may take a minute)")
model.fit(X_train, y_train)

# 7) Validation
val_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_proba)
print("Validation AUC: {:.4f}".format(auc))

# Precision@K (top K highest risks)
K = 100
top_k_idx = np.argsort(val_proba)[-K:]
precision_at_k = y_val.iloc[top_k_idx].mean()
print(f"Precision@{K}: {precision_at_k:.4f} (fraction of true events in top {K})")

# 8) Feature importance plot (saved to disk)
plt.figure(figsize=(8,6))
lgb.plot_importance(model, max_num_features=12, importance_type='gain')
plt.title("LightGBM Feature Importance (gain)")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved feature_importance.png")

# 9) Save model and encoder
joblib.dump(model, "lightgbm_cloudburst_model.pkl")
joblib.dump(le, "location_label_encoder.pkl")
print("Saved lightgbm_cloudburst_model.pkl and location_label_encoder.pkl")

print("Done. You can now use predict_cloudburst.py to run quick predictions on new samples.")
