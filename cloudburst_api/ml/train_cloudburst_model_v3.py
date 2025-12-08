#!/usr/bin/env python3
"""
train_cloudburst_model_v3.py

Trains an ensemble of calibrated LightGBM classifiers for cloudburst prediction.
Outputs:
 - calibrated_ensemble_v3.pkl      (list of calibrated estimators)
 - location_label_encoder_v3.pkl
 - thresholds_v3.json
 - features_v3.txt
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from lightgbm.sklearn import LGBMClassifier


# -------------------------
# Config
# -------------------------
DATA_PATH = "synthetic_cloudburst_data_v3.csv"
ENSEMBLE_SIZE = 5
SEEDS = [42, 10, 77, 123, 2024]  # length must == ENSEMBLE_SIZE
CALIBRATION_CV = 3               # cv folds for calibration (stratified)
N_ESTIMATORS = 800

FEATURE_COLS = [
    "location_id_enc",
    "rainfall_mm", "humidity", "pressure", "temperature", "wind_gust",
    "satellite_precip", "radar_dbz",
    "rain_last_5min", "rain_last_15min",
    "pressure_drop_15min", "humidity_change",
    "forecast_30min"
]
TARGET_COL = "cloudburst"


# -------------------------
# Load data
# -------------------------
print("Loading data:", DATA_PATH)
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

# Encode location_id
le = LabelEncoder()
df["location_id_enc"] = le.fit_transform(df["location_id"])

# Features & target
X = df[FEATURE_COLS].fillna(0).copy()
y = df[TARGET_COL].astype(int).copy()

print(f"Rows: {len(df)} Positives: {int(y.sum())} Negatives: {len(y)-int(y.sum())}")


# -------------------------
# Helper to compute scale_pos_weight
# -------------------------
def compute_scale_pos_weight(y_vec):
    pos = np.sum(y_vec == 1)
    neg = np.sum(y_vec == 0)
    return float(neg) / max(1.0, float(pos))


# -------------------------
# Train calibrated ensemble
# -------------------------
calibrated_models = []
ensemble_train_probs = np.zeros((len(X), ENSEMBLE_SIZE))

print("\nTraining calibrated ensemble...")

for i, seed in enumerate(SEEDS):
    print(f"\n-> Seed {seed} ({i+1}/{ENSEMBLE_SIZE})")

    params = {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.05,
        "num_leaves": 48,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "scale_pos_weight": compute_scale_pos_weight(y)
    }

    base_clf = LGBMClassifier(**params)

    # sklearn ≥1.6 uses "estimator=", not "base_estimator="
    cv = StratifiedKFold(n_splits=CALIBRATION_CV, shuffle=True, random_state=seed)

    calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=cv
    )

    calibrated.fit(X, y)

    calibrated_models.append(calibrated)

    # save in-sample predictions for ensemble evaluation
    probs = calibrated.predict_proba(X)[:, 1]
    ensemble_train_probs[:, i] = probs

    auc_i = roc_auc_score(y, probs)
    print(f"  Model {i+1} calibrated AUC: {auc_i:.4f}")


# -------------------------
# Ensemble metrics
# -------------------------
avg_calibrated_probs = ensemble_train_probs.mean(axis=1)
auc_ensemble = roc_auc_score(y, avg_calibrated_probs)

print(f"\nEnsemble calibrated AUC: {auc_ensemble:.4f}")


# -------------------------
# Threshold tuning
# -------------------------
print("\nTuning thresholds...")

# Precision@100
K = 100
sorted_idx = np.argsort(avg_calibrated_probs)[::-1]

precision_at_100 = (
    float(np.mean(y.iloc[sorted_idx[:K]]))
    if len(y) >= K else float(np.mean(y))
)

# Best F1 threshold
best_f1 = -1.0
best_f1_threshold = 0.5

for t in np.linspace(0.01, 0.99, 200):
    preds_t = (avg_calibrated_probs >= t).astype(int)
    f1 = f1_score(y, preds_t)
    if f1 > best_f1:
        best_f1 = f1
        best_f1_threshold = float(t)

thresholds = {
    "precision_at_100": precision_at_100,
    "precision_at_100_threshold_prob":
        float(avg_calibrated_probs[sorted_idx[K-1]]) if len(y) >= K else None,
    "best_f1_threshold": best_f1_threshold,
    "best_f1_value": float(best_f1),
}

print("Thresholds:", json.dumps(thresholds, indent=2))


# -------------------------
# Save results
# -------------------------
print("\nSaving artifacts...")

joblib.dump(calibrated_models, "calibrated_ensemble_v3.pkl")
joblib.dump(le, "location_label_encoder_v3.pkl")

with open("thresholds_v3.json", "w") as f:
    json.dump(thresholds, f, indent=2)

with open("features_v3.txt", "w") as f:
    for c in FEATURE_COLS:
        f.write(c + "\n")

print("\nSaved:")
print(" - calibrated_ensemble_v3.pkl")
print(" - location_label_encoder_v3.pkl")
print(" - thresholds_v3.json")
print(" - features_v3.txt")
print("\nTraining finished.\n")
