import argparse
import os
import sys
import logging
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

# --------------------------- Logging ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cloudburst")

# --------------------------- Original Functions (kept) --------------

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
    """Estimate sampling interval in seconds from timestamp column.
    If timestamps are non-uniform, returns median delta in seconds.
    """
    if 'timestamp' not in df.columns:
        return None
    dt = df['timestamp'].diff().dt.total_seconds().dropna()
    if dt.empty:
        return None
    return max(0.0, float(dt.median()))


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
        # Not enough samples to reliably apply filtfilt
        return series
    return filtfilt(b, a, series)


def fft_noise_analysis(series, fs, name_prefix, output_dir='.'):
    """
    Compute FFT power spectrum of a sensor series and save to CSV.
    No plotting, just data for analysis. Saves into output_dir.
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
    outpath = os.path.join(output_dir, f"fft_{name_prefix}.csv")
    fft_df.to_csv(outpath, index=False)
    log.info("Saved FFT to %s", outpath)


def flag_range_errors(df):
    """Flag impossible values based on physical ranges."""
    for col in ['temp_err', 'hum_err', 'press_err', 'rain_err', 'wind_err']:
        if col not in df.columns:
            df[col] = False

    if 'temp_C' in df.columns:
        bad = ~df['temp_C'].between(-40, 60)
        df.loc[bad, 'temp_err'] = True
        df.loc[bad, 'temp_C'] = np.nan

    if 'humidity' in df.columns:
        bad = ~df['humidity'].between(0, 100)
        df.loc[bad, 'hum_err'] = True
        df.loc[bad, 'humidity'] = np.nan

    if 'pressure' in df.columns:
        bad = ~df['pressure'].between(800, 1100)
        df.loc[bad, 'press_err'] = True
        df.loc[bad, 'pressure'] = np.nan

    if 'rain_mm' in df.columns:
        bad = df['rain_mm'] < 0
        df.loc[bad, 'rain_err'] = True
        df.loc[bad, 'rain_mm'] = 0.0

    if 'wind_speed' in df.columns:
        bad = ~df['wind_speed'].between(0, 60)
        df.loc[bad, 'wind_err'] = True
        df.loc[bad, 'wind_speed'] = np.nan

    return df


def flag_spike_errors(df):
    """Flag sudden impossible jumps between consecutive samples."""
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


# --------------------------- Enhancements ---------------------------

def guess_timestamp_column(df):
    """Try to find the timestamp-like column.
    Returns name or None.
    """
    candidates = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower() or 'date' in c.lower()]
    if candidates:
        return candidates[0]
    # fallback: if index looks like date
    return None


def safe_parse_timestamps(df, colname):
    """Parse timestamp column with robust errors handling."""
    try:
        df[colname] = pd.to_datetime(df[colname], errors='coerce', infer_datetime_format=True)
    except Exception:
        df[colname] = pd.to_datetime(df[colname].astype(str), errors='coerce')
    return df


def coerce_numeric_columns(df, exclude_cols=None):
    """Coerce potentially numeric columns to numeric with NaN on errors."""
    exclude_cols = exclude_cols or []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype == object:
            # try coercion
            coerced = pd.to_numeric(df[col], errors='coerce')
            # only replace if reasonable fraction converts
            non_na_fraction = coerced.notna().mean()
            if non_na_fraction > 0.3:  # arbitrary threshold
                df[col] = coerced
    return df


def resample_to_regular_intervals(df, ts_col='timestamp', rule='1S', how='interpolate'):
    """Resample dataframe to a regular time grid.
    rule: pandas offset alias like '1S', '1min', '10S'
    how: 'interpolate' or 'ffill' or 'zero' (for rain)
    """
    if ts_col not in df.columns:
        return df
    df = df.set_index(ts_col).sort_index()
    full = df.resample(rule).asfreq()

    if how == 'interpolate':
        numeric_cols = full.select_dtypes(include=[np.number]).columns
        full[numeric_cols] = full[numeric_cols].interpolate(limit=10).ffill().bfill()
    elif how == 'ffill':
        full = full.ffill().bfill()
    elif how == 'zero':
        # typically for rain counters
        full = full.fillna(0)
    full = full.reset_index()
    return full


def make_pi_compatible(df):
    """Downcast numeric types to save memory on Raspberry Pi and reduce compute.
    This mutates df in-place and returns it.
    """
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df


# --------------------------- clean_and_smooth (kept, small improvements) ----

def clean_and_smooth(df, output_dir='.', do_fft=True, pi_mode=False):
    """Handle NaNs, apply low/high-pass, Savitzky–Golay, Kalman + residual flags, FFT.

    Added arguments:
      - output_dir: where to write FFT CSVs
      - do_fft: skip FFT writes when not wanted
      - pi_mode: if True, reduce heavy operations and memory
    """
    df = df.copy()

    # detect numeric columns more robustly
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = num_cols.drop('cloudburst', errors='ignore')

    # interpolate short gaps but don't invent long stretches
    df[num_cols] = df[num_cols].interpolate(limit=4, limit_direction='both')
    df[num_cols] = df[num_cols].ffill().bfill()

    # sampling frequency estimation: returns median delta seconds
    fs = estimate_sampling_interval_seconds(df)
    if fs is None or fs <= 0:
        fs_hz = 1.0
    else:
        fs_hz = 1.0 / fs

    try:
        # For typical environmental signals, cutoff in Hz relative to sampling freq
        b_lp, a_lp = design_lowpass(fs_hz, cutoff_hz=0.01, order=3)
        b_hp, a_hp = design_highpass(fs_hz, cutoff_hz=0.0001, order=3)
    except Exception:
        b_lp = a_lp = b_hp = a_hp = None

    sensor_cols = ['temp_C', 'humidity', 'pressure', 'rain_mm', 'wind_speed']

    for col in sensor_cols:
        if col not in df.columns:
            continue

        raw = df[col].values.astype(float)

        # Low-pass
        if b_lp is not None:
            try:
                lp = apply_filter(raw, b_lp, a_lp)
            except Exception:
                lp = raw.copy()
            df[col + '_lp'] = lp
        else:
            lp = raw.copy()

        # High-pass
        if b_hp is not None:
            try:
                hp = apply_filter(raw, b_hp, a_hp)
                df[col + '_hp'] = hp
            except Exception:
                pass

        # Savitzky–Golay on low-passed signal
        if len(lp) >= 11:
            window = 11 if len(lp) >= 11 else (len(lp) // 2) * 2 + 1
            try:
                sg = savgol_filter(lp, window_length=window, polyorder=2, mode='mirror')
            except Exception:
                sg = lp.copy()
        else:
            sg = lp.copy()

        df[col + '_sg'] = sg

        # Kalman on SG-smoothed signal
        kf, residuals = kalman_1d(sg, process_var=0.01, meas_var=1.0)
        df[col + '_kalman'] = kf
        df[col + '_kalman_resid'] = residuals

        resid_std = np.std(residuals)
        if resid_std > 0:
            mask = np.abs(residuals) > 3 * resid_std
            flag_col = col + '_kalman_outlier'
            df[flag_col] = False
            df.loc[mask, flag_col] = True

        # FFT noise analysis → saves fft_<col>.csv
        if do_fft:
            try:
                fft_noise_analysis(raw, fs_hz, name_prefix=col, output_dir=output_dir)
            except Exception:
                log.exception("FFT failed for %s", col)

    if pi_mode:
        # reduce dataframe memory
        df = make_pi_compatible(df)

    return df


# --------------------------- FEATURE ENGINEERING (kept) -----------------

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


# --------------------------- MODEL TRAINING (kept, made configurable) -----

def train_model(df, output_model_path="cloudburst_rf_model.joblib", pi_mode=False, n_estimators=None, skip_if_no_label=True):
    """Train RandomForest model if 'cloudburst' label exists.

    New args:
      - pi_mode: if True, use fewer estimators and n_jobs=1
      - n_estimators: override number of estimators
      - skip_if_no_label: if True, skip training when label absent (default preserved)
    """
    if 'cloudburst' not in df.columns:
        if skip_if_no_label:
            log.info("No 'cloudburst' column – skipping model training.")
            return
        else:
            raise ValueError("'cloudburst' label required for training")

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
        log.info("No usable features for model training.")
        return

    X = df[feature_cols]
    y = df['cloudburst'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    if pi_mode:
        default_estimators = 50
        n_jobs = 1
    else:
        default_estimators = 200
        n_jobs = -1

    if n_estimators is None:
        n_estimators = default_estimators

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=int(n_estimators),
            class_weight="balanced",
            random_state=42,
            n_jobs=n_jobs
        ))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    log.info("=== MODEL PERFORMANCE ===")
    log.info("Accuracy: %s", accuracy_score(y_test, y_pred))
    log.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    log.info("Classification report:\n%s", classification_report(y_test, y_pred, digits=3))

    rf = clf.named_steps['rf']
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fi_path = os.path.join(os.getcwd(), "feature_importance_rf.csv")
    importances.to_csv(fi_path, header=['importance'], index_label='feature')
    log.info("Saved %s", fi_path)

    joblib.dump(clf, output_model_path, compress=3)
    log.info("Saved trained model to %s", output_model_path)


# --------------------------- MAIN -------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cloudburst sensor data full pipeline (v2) - Pi ready")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to input CSV file with sensor readings (timestamp + sensor columns)"
    )
    parser.add_argument("--output-dir", default='.', help="Directory to save outputs (csvs, model)")
    parser.add_argument("--resample-interval", default=None,
                        help="If provided (e.g. '1S', '10S', '1min') resample data to a regular grid")
    parser.add_argument("--resample-method", default='interpolate', choices=['interpolate','ffill','zero'],
                        help="How to fill values after resampling; choose 'zero' for rain counters")
    parser.add_argument("--pi-mode", action='store_true', help="Enable Raspberry Pi friendly mode (downcast types, fewer trees)")
    parser.add_argument("--no-fft", action='store_true', help="Skip FFT CSV outputs (saves I/O on Pi)")
    parser.add_argument("--no-train", action='store_true', help="Skip model training step")
    parser.add_argument("--n-estimators", type=int, default=None, help="Override RF n_estimators")
    parser.add_argument("--force-timestamp-col", default=None, help="Force a column name to use as timestamp")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load CSV robustly
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        log.exception("Failed to read CSV: %s", args.csv)
        sys.exit(1)

    # Try to find timestamp column
    ts_col = None
    if args.force_timestamp_col:
        ts_col = args.force_timestamp_col
        if ts_col not in df.columns:
            log.error("Forced timestamp column '%s' not in CSV", ts_col)
            sys.exit(1)
    else:
        if 'timestamp' in df.columns:
            ts_col = 'timestamp'
        else:
            guessed = guess_timestamp_column(df)
            ts_col = guessed if guessed and guessed in df.columns else None

    if ts_col is None:
        # try index as timestamp
        if df.index.dtype == 'int64' or np.issubdtype(df.index.dtype, np.datetime64):
            try:
                df = df.reset_index().rename(columns={'index': 'timestamp'})
                ts_col = 'timestamp'
            except Exception:
                log.error("Could not detect timestamp column. Please provide --force-timestamp-col")
                sys.exit(1)
        else:
            log.error("Could not detect timestamp column. Please provide --force-timestamp-col")
            log.info("Columns found: %s", list(df.columns))
            sys.exit(1)

    df = safe_parse_timestamps(df, ts_col)
    if df[ts_col].isna().all():
        log.error("All parsed timestamps are NaT. Check CSV formatting or provide --force-timestamp-col")
        sys.exit(1)

    # rename to 'timestamp' for internal consistency
    if ts_col != 'timestamp':
        df = df.rename(columns={ts_col: 'timestamp'})

    # Coerce numeric columns where possible
    df = coerce_numeric_columns(df, exclude_cols=['timestamp'])

    # Optionally resample to regular grid
    if args.resample_interval:
        log.info("Resampling to %s using method=%s", args.resample_interval, args.resample_method)
        df = resample_to_regular_intervals(df, ts_col='timestamp', rule=args.resample_interval, how=args.resample_method)

    # Sort and reset index
    df = df.sort_values('timestamp').reset_index(drop=True)
    log.info("Loaded: %s (shape=%s)", args.csv, df.shape)

    # 2. Hardware-like error detection
    df = apply_error_filters(df)

    # 3. Save error log
    if 'any_error' in df.columns:
        error_log = df[df['any_error'] == True].copy()
        if not error_log.empty:
            error_path = os.path.join(args.output_dir, "sensor_error_log.csv")
            error_log.to_csv(error_path, index=False)
            log.info("Saved sensor_error_log.csv with %d rows of sensor issues.", len(error_log))
        else:
            log.info("No sensor errors detected according to current rules.")
    else:
        log.info("No error flags present.")

    # 4. Cleaning + smoothing + FFT
    df = clean_and_smooth(df, output_dir=args.output_dir, do_fft=(not args.no_fft), pi_mode=args.pi_mode)

    # 5. Feature engineering
    df = add_features(df)

    # 6. Save engineered dataset
    engineered_path = os.path.join(args.output_dir, "engineered_final_dataset.csv")
    df.to_csv(engineered_path, index=False)
    log.info("Saved engineered_final_dataset.csv with shape: %s", df.shape)

    # 7. Train model (optional)
    if not args.no_train:
        try:
            model_path = os.path.join(args.output_dir, "cloudburst_rf_model.joblib")
            train_model(df, output_model_path=model_path, pi_mode=args.pi_mode, n_estimators=args.n_estimators)
        except Exception:
            log.exception("Model training failed")
    else:
        log.info("Skipping model training as requested (--no-train)")

    log.info("Pipeline finished. Outputs in %s", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
