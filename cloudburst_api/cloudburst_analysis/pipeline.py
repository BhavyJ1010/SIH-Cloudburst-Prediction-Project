

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


file_path = "/Users/namrata/Namrata/cloudburst_analysis/synthetic_cloudburst_data (1).csv"

df = pd.read_csv(
    file_path,
    parse_dates=['timestamp']
)

df = df.sort_values('timestamp').reset_index(drop=True)

print("Initial shape:", df.shape)
print(df.info())
print("\nNaN counts before cleaning:")
print(df.isna().sum())


num_cols = df.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop('cloudburst', errors='ignore')


df[num_cols] = df[num_cols].interpolate(
    limit=3,
    limit_direction='both'
)


df[num_cols] = df[num_cols].ffill().bfill()

print("\nNaN counts after numeric interpolation + ffill/bfill:")
print(df.isna().sum())

for col in ['rainfall_mm', 'rain_last_5min', 'rain_last_15min']:
    if col in df.columns:
        df[col] = df[col].fillna(0)


df = df.set_index('timestamp')


for col in ['rainfall_mm', 'pressure', 'humidity']:
    if col not in df.columns:
        continue

    mean = df[col].mean()
    std = df[col].std()
    z = (df[col] - mean) / std

    outlier_col = col + '_is_outlier'

    if col == 'rainfall_mm' and 'cloudburst' in df.columns:
        
        df[outlier_col] = (z.abs() > 3) & (df['cloudburst'] == 0)
    else:
        df[outlier_col] = z.abs() > 3

    
    df.loc[df[outlier_col], col] = np.nan


df[num_cols] = df[num_cols].interpolate(
    limit=3,
    limit_direction='both'
).ffill().bfill()

print("\nNaN counts after outlier handling:")
print(df.isna().sum())


for col in ['rainfall_mm', 'pressure', 'humidity']:
    if col not in df.columns:
        continue

    med = df[col].rolling(window=3, center=True, min_periods=1).median()
    smooth = med.rolling(window=5, center=True, min_periods=1).mean()
    df[col + '_smooth'] = smooth


rain_series = df['rainfall_mm_smooth'] if 'rainfall_mm_smooth' in df.columns else df['rainfall_mm']
pressure_series = df['pressure_smooth'] if 'pressure_smooth' in df.columns else df['pressure']
humidity_series = df['humidity_smooth'] if 'humidity_smooth' in df.columns else df['humidity']


df['rain_5'] = rain_series.rolling('5min').sum()
df['rain_15'] = rain_series.rolling('15min').sum()


periods_15 = 3

df['pressure_prev_15'] = pressure_series.shift(periods_15)
df['pressure_drop_15'] = df['pressure_prev_15'] - pressure_series

df['humidity_prev_15'] = humidity_series.shift(periods_15)
df['humidity_change_15'] = humidity_series - df['humidity_prev_15']


df = df.dropna().reset_index()

print("Final shape after feature engineering:", df.shape)


df.to_csv("engineered_features.csv", index=False)
print("Saved engineered_features.csv")


plt.figure(figsize=(12, 3))
plt.plot(df['timestamp'], df['rain_15'], label='rain_15min')
plt.xlabel("Time")
plt.ylabel("Rain in last 15 min (mm)")
plt.legend()
plt.tight_layout()
plt.savefig("rain_15_timeseries.png")

if 'cloudburst' in df.columns:
    evt = df[df['cloudburst'] == 1]
    plt.figure(figsize=(12, 3))
    plt.plot(df['timestamp'], df['rain_15'], label='rain_15min')
    plt.scatter(evt['timestamp'], evt['rain_15'], label='cloudburst')
    plt.xlabel("Time")
    plt.ylabel("Rain in last 15 min (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rain_cloudburst_overlay.png")

sns.histplot(data=df, x='rain_15', hue='cloudburst',
             stat='density', common_norm=False)
plt.tight_layout()
plt.savefig("rain15_hist_by_label.png")

corr_cols = ['rain_15', 'rain_5', 'pressure_drop_15',
             'humidity_change_15', 'satellite_precip',
             'forecast_30min', 'cloudburst']
corr_cols = [c for c in corr_cols if c in df.columns]

corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f")
plt.tight_layout()
plt.savefig("corr_heatmap.png")


print("\n=== MODELING SECTION ===")

assert 'cloudburst' in df.columns, "No 'cloudburst' column found for modeling."

candidate_features = [
    'rain_15', 'rain_5',
    'pressure_drop_15', 'humidity_change_15',
    'satellite_precip', 'forecast_30min',
    'rainfall_mm_smooth', 'pressure_smooth', 'humidity_smooth',
]

feature_cols = [c for c in candidate_features if c in df.columns]
print("Using features:", feature_cols)

X = df[feature_cols].copy()
y = df['cloudburst'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])


pos = df[df['cloudburst'] == 1]

thr_rain15 = pos['rain_15'].quantile(0.75) if 'rain_15' in pos else None
thr_pdrop  = pos['pressure_drop_15'].quantile(0.5) if 'pressure_drop_15' in pos else None
thr_hchg   = pos['humidity_change_15'].quantile(0.5) if 'humidity_change_15' in pos else None

print("\nRule thresholds:")
print(" rain_15 >", thr_rain15)
print(" pressure_drop_15 >", thr_pdrop)
print(" humidity_change_15 >", thr_hchg)

def rule_predict(df_slice):
    conds = []
    if thr_rain15 is not None and 'rain_15' in df_slice:
        conds.append(df_slice['rain_15'] > thr_rain15)
    if thr_pdrop is not None and 'pressure_drop_15' in df_slice:
        conds.append(df_slice['pressure_drop_15'] > thr_pdrop)
    if thr_hchg is not None and 'humidity_change_15' in df_slice:
        conds.append(df_slice['humidity_change_15'] > thr_hchg)

    if len(conds) == 0:
        raise ValueError("No usable features for rule-based system.")

    rule = np.logical_and.reduce(conds)
    return rule.astype(int)

y_rule = rule_predict(X_test)

print("\n=== Rule-based system performance ===")
print("Accuracy:", accuracy_score(y_test, y_rule))
print("Confusion matrix:\n", confusion_matrix(y_test, y_rule))
print(classification_report(y_test, y_rule, digits=3))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

y_pred_lr = logreg.predict(X_test_scaled)
y_prob_lr = logreg.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, digits=3))
try:
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    print("ROC AUC:", auc_lr)
except Exception as e:
    print("Could not compute AUC:", e)


rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, digits=3))
try:
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    print("ROC AUC:", auc_rf)
except Exception as e:
    print("Could not compute AUC:", e)


importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Random Forest feature importance ===")
print(importances)

importances.to_csv("feature_importance_rf.csv", header=['importance'])
print("Saved feature_importance_rf.csv")
