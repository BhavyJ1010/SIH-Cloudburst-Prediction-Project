import argparse
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib



def kalman_1d(series, process_var=0.01, meas_var=1.0):
    
    series = np.asarray(series, dtype=float)
    n = len(series)
    x_hat = np.zeros(n, dtype=float)
    P = np.zeros(n, dtype=float)
    residuals = np.zeros(n, dtype=float)

    
    x_hat[0] = series[0]
    P[0] = 1.0

    Q = process_var
    R = meas_var

    for k in range(1, n):
        # Predict
        x_pred = x_hat[k-1]
        P_pred = P[k-1] + Q

        # Update
        K = P_pred / (P_pred + R)
        x_hat[k] = x_pred + K * (series[k] - x_pred)
        P[k] = (1 - K) * P_pred

        residuals[k] = series[k] - x_hat[k]

    return x_hat, residuals



def estimate_sampling_interval_seconds(df):
    """Estimate sampling interval in seconds from timestamp column."""
    dt = df['timestamp'].diff().dt.total_seconds().dropna()
    if dt.empty:
        return None
    return dt.median()


def design_lowpass(fs, cutoff_hz, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def design_highpass(fs, cutoff_hz, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def apply_filter(series, b, a):
    series = np.asarray(series, dtype=float)
    if len(series) < max(len(a), len(b)) * 3:
        
        return series
    return filtfilt(b, a, series)


def fft_noise_analysis(series, fs, name_prefix):
    """
    Compute FFT power spectrum of a sensor series and save to CSV.
    No plotting, just data for analysis.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 4:
        return

   
    x = series - np.mean(series)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)  # Hz
    spectrum = np.fft.rfft(x)
    power = (np.abs(spectrum) ** 2) / n

    fft_df = pd.DataFrame({
        "frequency_hz": freqs,
        "power": power
    })
    fft_df.to_csv(f"fft_{name_prefix}.csv", index=False)



def flag_range_errors(df):
    """Flag impossible values based on physical ranges."""
    for col in ['temp_err', 'hum_err', 'press_err', 'rain_err', 'wind_err']:
        if col not in df.columns:
            df[col] = False

   
    if 'temp_C' in df.columns:
        bad = ~df['temp_C'].between(-40, 60)
        df.loc[bad, 'temp_err'] = True
        df.loc[bad, 'temp_C'] = np.nan

    # Humidity: 0–100 % (SHT31)
    if 'humidity' in df.columns:
        bad = ~df['humidity'].between(0, 100)
        df.loc[bad, 'hum_err'] = True
        df.loc[bad, 'humidity'] = np.nan

    # Pressure: 800–1100 hPa (BMP280 typical range)
    if 'pressure' in df.columns:
        bad = ~df['pressure'].between(800, 1100)
        df.loc[bad, 'press_err'] = True
        df.loc[bad, 'pressure'] = np.nan

    # Rain: cannot be negative (tipping bucket)
    if 'rain_mm' in df.columns:
        bad = df['rain_mm'] < 0
        df.loc[bad, 'rain_err'] = True
        df.loc[bad, 'rain_mm'] = 0.0

    # Wind: 0–60 m/s (3-cup anemometer)
    if 'wind_speed' in df.columns:
        bad = ~df['wind_speed'].between(0, 60)
        df.loc[bad, 'wind_err'] = True
        df.loc[bad, 'wind_speed'] = np.nan

    return df


def flag_spike_errors(df):
    """Flag sudden impossible jumps between consecutive samples."""
    # Thresholds – tune according to sampling interval
    max_temp_step = 5.0      # °C per sample
    max_press_step = 3.0     # hPa per sample
    max_hum_step = 20.0      # % per sample
    max_wind_step = 15.0     # m/s per sample

    if 'temp_C' in df.columns:
        diff = df['temp_C'].diff().abs()
        bad = diff > max_temp_step
        df.loc[bad, 'temp_err'] = True
        df.loc[bad, 'temp_C'] = np.nan

    if 'pressure' in df.columns:
        diff = df['pressure'].diff().abs()
        bad = diff > max_press_step
        df.loc[bad, 'press_err'] = True
        df.loc[bad, 'pressure'] = np.nan

    if 'humidity' in df.columns:
        diff = df['humidity'].diff().abs()
        bad = diff > max_hum_step
        df.loc[bad, 'hum_err'] = True
        df.loc[bad, 'humidity'] = np.nan

    if 'wind_speed' in df.columns:
        diff = df['wind_speed'].diff().abs()
        bad = diff > max_wind_step
        df.loc[bad, 'wind_err'] = True
        df.loc[bad, 'wind_speed'] = np.nan

    return df


def flag_stuck_sensor(df, col, err_col, run_threshold=20):
    """
    Flag sensor stuck at identical value for many samples.
    run_threshold: number of consecutive identical readings to treat as error.
    """
    if col not in df.columns:
        return df

    same_as_prev = df[col].diff().fillna(0) == 0
    group_id = (~same_as_prev).cumsum()
    counts = same_as_prev.groupby(group_id).transform('sum') + 1
    bad = (counts >= run_threshold) & same_as_prev

    df.loc[bad, err_col] = True
    df.loc[bad, col] = np.nan
    return df


def apply_error_filters(df):
    """Apply range, spike, and stuck-sensor checks."""
    df = flag_range_errors(df)
    df = flag_spike_errors(df)

    if 'temp_C' in df.columns:
        df = flag_stuck_sensor(df, 'temp_C', 'temp_err')
    if 'humidity' in df.columns:
        df = flag_stuck_sensor(df, 'humidity', 'hum_err')
    if 'pressure' in df.columns:
        df = flag_stuck_sensor(df, 'pressure', 'press_err')
    if 'wind_speed' in df.columns:
        df = flag_stuck_sensor(df, 'wind_speed', 'wind_err')

    error_cols = [c for c in ['temp_err', 'hum_err', 'press_err', 'rain_err', 'wind_err'] if c in df.columns]
    df['any_error'] = df[error_cols].any(axis=1)

    return df



def clean_and_smooth(df):
    """Handle NaNs, apply low/high-pass, Savitzky–Golay, Kalman + residual flags, FFT."""
    df = df.copy()

   
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = num_cols.drop('cloudburst', errors='ignore')

    df[num_cols] = df[num_cols].interpolate(limit=4, limit_direction='both')
    df[num_cols] = df[num_cols].ffill().bfill()

    
    fs = estimate_sampling_interval_seconds(df)
    if fs is None or fs <= 0:
        fs_hz = 1.0   
    else:
        fs_hz = 1.0 / fs  

    
    try:
        b_lp, a_lp = design_lowpass(fs_hz, cutoff_hz=0.01, order=3)
        b_hp, a_hp = design_highpass(fs_hz, cutoff_hz=0.0001, order=3)
    except ValueError:
        
        b_lp = a_lp = b_hp = a_hp = None

   
    sensor_cols = ['temp_C', 'humidity', 'pressure', 'rain_mm', 'wind_speed']

    for col in sensor_cols:
        if col not in df.columns:
            continue

        raw = df[col].values.astype(float)

        # Low-pass
        if b_lp is not None:
            lp = apply_filter(raw, b_lp, a_lp)
            df[col + '_lp'] = lp
        else:
            lp = raw.copy()

        # High-pass (to inspect high-frequency noise component)
        if b_hp is not None:
            hp = apply_filter(raw, b_hp, a_hp)
            df[col + '_hp'] = hp

        # Savitzky–Golay on low-passed signal
        if len(lp) >= 11:
            window = 11 if len(lp) >= 11 else (len(lp) // 2) * 2 + 1
            sg = savgol_filter(lp, window_length=window, polyorder=2, mode='mirror')
        else:
            sg = lp.copy()

        df[col + '_sg'] = sg

        # Kalman on SG-smoothed signal
        kf, residuals = kalman_1d(sg, process_var=0.01, meas_var=1.0)
        df[col + '_kalman'] = kf
        df[col + '_kalman_resid'] = residuals

        # Model-based noise detection: residual outliers
        resid_std = np.std(residuals)
        if resid_std > 0:
            mask = np.abs(residuals) > 3 * resid_std
            flag_col = col + '_kalman_outlier'
            df[flag_col] = False
            df.loc[mask, flag_col] = True

        # FFT noise analysis → saves fft_<col>.csv
        fft_noise_analysis(raw, fs_hz, name_prefix=col)

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def add_features(df):
    """Create time-based and instability features from sensor data."""
    df = df.copy()
    df = df.set_index('timestamp')

    rain_col = 'rain_mm_kalman' if 'rain_mm_kalman' in df.columns else 'rain_mm'
    temp_col = 'temp_C_kalman' if 'temp_C_kalman' in df.columns else 'temp_C'
    hum_col = 'humidity_kalman' if 'humidity_kalman' in df.columns else 'humidity'
    press_col = 'pressure_kalman' if 'pressure_kalman' in df.columns else 'pressure'
    wind_col = 'wind_speed_kalman' if 'wind_speed_kalman' in df.columns else 'wind_speed'

    # Rolling rainfall sums
    if rain_col in df.columns:
        df['rain_5'] = df[rain_col].rolling('5min').sum()
        df['rain_15'] = df[rain_col].rolling('15min').sum()
        df['rain_30'] = df[rain_col].rolling('30min').sum()

    # Temperature features
    if temp_col in df.columns:
        df['temp_mean_15'] = df[temp_col].rolling('15min').mean()
        df['temp_change_15'] = df[temp_col].diff(periods=3)
        df['temp_std_15'] = df[temp_col].rolling('15min').std()

    # Humidity features
    if hum_col in df.columns:
        df['hum_mean_15'] = df[hum_col].rolling('15min').mean()
        df['hum_change_15'] = df[hum_col].diff(periods=3)
        df['hum_std_15'] = df[hum_col].rolling('15min').std()

    # Pressure features
    if press_col in df.columns:
        df['pressure_drop_15'] = df[press_col].diff(periods=3)
        df['pressure_std_30'] = df[press_col].rolling('30min').std()

    # Wind features
    if wind_col in df.columns:
        df['wind_mean_15'] = df[wind_col].rolling('15min').mean()
        df['wind_max_15'] = df[wind_col].rolling('15min').max()
        df['wind_std_15'] = df[wind_col].rolling('15min').std()
        df['wind_change_15'] = df[wind_col].diff(periods=3)

    # Dew point (SHT31) and saturation index
    if ('temp_C' in df.columns) and ('humidity' in df.columns):
        T = df['temp_C']
        RH = df['humidity'].clip(lower=1)

        a, b = 17.27, 237.7
        alpha = (a * T / (b + T)) + np.log(RH / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        df['dew_point'] = dew_point
        df['dewpoint_depression'] = T - dew_point

    df = df.dropna().reset_index()
    return df


# =========================================================
# MODEL TRAINING
# =========================================================
def train_model(df, output_model_path="cloudburst_rf_model.joblib"):
    """Train RandomForest model if 'cloudburst' label exists."""
    if 'cloudburst' not in df.columns:
        print("No 'cloudburst' column – skipping model training.")
        return

    feature_cols = [
        'rain_5', 'rain_15', 'rain_30',
        'pressure_drop_15', 'pressure_std_30',
        'hum_change_15', 'hum_mean_15', 'hum_std_15',
        'temp_change_15', 'temp_std_15', 'temp_mean_15',
        'wind_mean_15', 'wind_max_15', 'wind_std_15', 'wind_change_15',
        'dew_point', 'dewpoint_depression',
        'satellite_precip', 'forecast_30min',
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        print("No usable features for model training.")
        return

    X = df[feature_cols]
    y = df['cloudburst'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== MODEL PERFORMANCE ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))

    rf = clf.named_steps['rf']
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importances.to_csv("feature_importance_rf.csv", header=['importance'], index_label='feature')
    print("Saved feature_importance_rf.csv")

    joblib.dump(clf, output_model_path)
    print(f"Saved trained model to {output_model_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Cloudburst sensor data full pipeline (v2)")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to input CSV file with sensor readings (timestamp + sensor columns)"
    )
    args = parser.parse_args()

    # 1. Load
    df = pd.read_csv(args.csv, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print("Loaded:", args.csv)
    print("Initial shape:", df.shape)

    # 2. Hardware-like error detection
    df = apply_error_filters(df)

    # 3. Save error log
    if 'any_error' in df.columns:
        error_log = df[df['any_error'] == True].copy()
        if not error_log.empty:
            error_log.to_csv("sensor_error_log.csv", index=False)
            print("Saved sensor_error_log.csv with", len(error_log), "rows of sensor issues.")
        else:
            print("No sensor errors detected according to current rules.")
    else:
        print("No error flags present.")

    # 4. Cleaning + low/high pass + SG + Kalman + FFT
    df = clean_and_smooth(df)

    # 5. Feature engineering
    df = add_features(df)

    # 6. Save engineered dataset
    df.to_csv("engineered_final_dataset.csv", index=False)
    print("Saved engineered_final_dataset.csv with shape:", df.shape)

    # 7. Train model
    train_model(df)


if __name__ == "__main__":
    main()
