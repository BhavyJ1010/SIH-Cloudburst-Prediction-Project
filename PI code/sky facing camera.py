#!/usr/bin/env python3
"""
Sky camera feature extraction for Raspberry Pi.

- Connects to phone IP camera stream
- Computes:
  - Cloud motion (Farnebäck optical flow)
  - Optical properties (brightness, contrast, dark_fraction, cloud_fraction, texture)
  - Vertical growth metrics (darkening, cloud area growth, texture growth)
  - 20-min trend metrics using 2–3 hours of history
  - Time-of-day brightness anomaly
- Appends all features to a CSV file (no database writes).

Intended as a middle ML step. Another script can read the CSV and push to DB.
"""

import cv2
import numpy as np
import time
import math
import csv
import os
from collections import deque

# =============== CONFIG ====================

# ---- Debug / logging ----
DEBUG = False   # set True only when you want console logs

# ---- Stream & sampling ----
STREAM_URL = "http://192.168.1.20:8080/video"  # <- update to your phone IP stream
SAMPLE_INTERVAL_SEC = 60          # update ~every 1 minute
MAX_CONSECUTIVE_ERRORS = 20       # after this, try to fully reopen stream

# ---- Frame size ----
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

# ---- Optical flow config ----
MIN_FLOW_MAG = 0.3
MAX_FLOW_MAG = 20.0
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=2,
    winsize=11,
    iterations=2,
    poly_n=5,
    poly_sigma=1.1,
    flags=0
)

# ---- Geometry / physical mapping ----
HFOV_DEG = 90.0           # camera horizontal FOV
CLOUD_HEIGHT_M = 2000.0   # assumed cloud base height [m]

# ---- ROI masking (use top part of frame for sky) ----
ROI_TOP_FRACTION = 0.7    # use rows [0 : h * ROI_TOP_FRACTION]

# ---- CSV logging ----
CSV_PATH = "sky_features.csv"

# ---- EMA smoothing factor (0–1), higher = more weight on new value ----
EMA_ALPHA = 0.3

# ---- Optical / brightness thresholds ----
DARK_THRESHOLD = 90
CLOUD_THRESHOLD = 200

THIN_BRIGHTNESS_NORM_MIN = 0.7
THIN_DARK_FRACTION_MAX = 0.15

MEDIUM_BRIGHTNESS_NORM_MIN = 0.4
MEDIUM_BRIGHTNESS_NORM_MAX = 0.7
MEDIUM_DARK_FRACTION_MAX = 0.40

# ---- Vertical growth thresholds ----
VG_DARKENING_RATE_MIN = 8.0              # brightness units / minute
VG_BRIGHTNESS_MAX_FOR_DARK = 170.0       # must already be relatively dark
VG_CLOUD_GROWTH_MIN = 0.08               # per minute
VG_CLOUD_FRACTION_MIN = 0.60             # must already be quite cloudy
VG_TEXTURE_GROWTH_MIN = 1.5              # per minute
VG_TEXTURE_MIN = 6.0                     # absolute texture index

# ---- Historical data / trend analysis ----
HISTORY_MAX_HOURS = 3.0          # keep ~2–3 hours of history
TREND_WINDOW_MIN = 20.0          # 20-minute trend window

# ---- Time-of-day compensation ----
USE_TIME_OF_DAY_COMP = True      # if True, compute brightness anomaly


# =============== UTILS =====================

def debug_log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def compute_meters_per_pixel(width_px, hfov_deg, cloud_height_m):
    hfov_rad = math.radians(hfov_deg)
    scene_width_m = 2 * cloud_height_m * math.tan(hfov_rad / 2.0)
    return scene_width_m / float(width_px)


def angle_to_direction(angle_deg):
    angle = (angle_deg + 360) % 360
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((angle + 22.5) % 360) / 45)
    return dirs[idx]


def expected_clear_sky_brightness(local_time_struct):
    """
    Very simple time-of-day curve for clear sky brightness.
    Peak around 13:00, near-zero at night.
    """
    hour = local_time_struct.tm_hour + local_time_struct.tm_min / 60.0

    # Map [6,18] -> [0,pi], sinusoidal daylight curve
    if hour < 6 or hour > 18:
        return 0.0
    x = (hour - 6.0) / 12.0 * math.pi
    return max(0.0, 255.0 * math.sin(x))


def compute_cloud_optical_features(gray_roi):
    g = gray_roi.astype(np.float32)

    brightness = float(g.mean())
    contrast = float(g.std())

    dark_fraction = float((g < DARK_THRESHOLD).mean())
    cloud_fraction = float((g < CLOUD_THRESHOLD).mean())

    lap = cv2.Laplacian(gray_roi, cv2.CV_32F)
    texture_index = float(np.mean(np.abs(lap)))

    brightness_norm = brightness / 255.0

    # Thickness classification based on normalized brightness + dark fraction
    if brightness_norm > THIN_BRIGHTNESS_NORM_MIN and dark_fraction < THIN_DARK_FRACTION_MAX:
        thickness = "THIN"
    elif (MEDIUM_BRIGHTNESS_NORM_MIN < brightness_norm <= MEDIUM_BRIGHTNESS_NORM_MAX
          and dark_fraction < MEDIUM_DARK_FRACTION_MAX):
        thickness = "MEDIUM"
    else:
        thickness = "THICK_CONVECTIVE"

    return {
        "brightness": brightness,
        "contrast": contrast,
        "dark_fraction": dark_fraction,
        "cloud_fraction": cloud_fraction,
        "texture_index": texture_index,
        "thickness_class": thickness,
    }


def compute_vertical_growth(prev_optical, optical, dt_min):
    if dt_min <= 0:
        dt_min = 1e-3

    darkening_rate = (prev_optical["brightness"] - optical["brightness"]) / dt_min
    cloud_area_growth_rate = (
        optical["cloud_fraction"] - prev_optical["cloud_fraction"]
    ) / dt_min
    texture_growth_rate = (
        optical["texture_index"] - prev_optical["texture_index"]
    ) / dt_min

    rapid_darkening = (
        darkening_rate > VG_DARKENING_RATE_MIN
        and optical["brightness"] < VG_BRIGHTNESS_MAX_FOR_DARK
    )
    fast_cloud_growth = (
        cloud_area_growth_rate > VG_CLOUD_GROWTH_MIN
        and optical["cloud_fraction"] > VG_CLOUD_FRACTION_MIN
    )
    texture_increase = (
        texture_growth_rate > VG_TEXTURE_GROWTH_MIN
        and optical["texture_index"] > VG_TEXTURE_MIN
    )

    vertical_growth_flag = (
        rapid_darkening
        and fast_cloud_growth
        and texture_increase
        and optical["thickness_class"] == "THICK_CONVECTIVE"
    )

    return darkening_rate, cloud_area_growth_rate, texture_growth_rate, vertical_growth_flag


def compute_flow(prev_gray_roi, gray_roi, dt_sec, m_per_px):
    if dt_sec <= 0:
        dt_sec = 1e-3

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_roi, gray_roi, None, **FARNEBACK_PARAMS
    )

    vx = flow[..., 0]
    vy = flow[..., 1]
    mag = np.sqrt(vx**2 + vy**2)
    mask = (mag > MIN_FLOW_MAG) & (mag < MAX_FLOW_MAG)

    if not np.any(mask):
        return None, None, None, None, None

    vx_valid = vx[mask]
    vy_valid = vy[mask]
    mag_valid = mag[mask]

    mean_vx = float(np.median(vx_valid))
    mean_vy = float(np.median(vy_valid))
    mean_mag = float(np.median(mag_valid))

    vx_frame = mean_vx
    vy_frame = -mean_vy  # invert y for "up is North"

    angle_rad = math.atan2(vy_frame, vx_frame)
    angle_deg = math.degrees(angle_rad)
    direction = angle_to_direction(angle_deg)

    px_per_sec = mean_mag / dt_sec
    m_per_sec = px_per_sec * m_per_px
    km_per_h = m_per_sec * 3.6

    return mean_vx, mean_vy, direction, km_per_h, px_per_sec


def ema(prev, current, alpha=EMA_ALPHA):
    if prev is None:
        return current
    return alpha * current + (1 - alpha) * prev


def init_csv(path):
    new_file = not os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow([
            "timestamp",
            "brightness",
            "contrast",
            "dark_fraction",
            "cloud_fraction",
            "texture_index",
            "thickness_class",
            "darkening_rate",
            "cloud_area_growth_rate",
            "texture_growth_rate",
            "vertical_growth_flag",
            "cloud_vx",
            "cloud_vy",
            "cloud_speed_kmh",
            "cloud_direction",
            "hour_local",
            "brightness_expected_clear",
            "brightness_anomaly",
            "trend20_brightness_slope",
            "trend20_cloud_fraction_slope",
            "trend20_dark_fraction_slope",
            "trend20_texture_slope",
        ])
    return f, writer


def compute_trend_over_window(history, now_epoch, window_min):
    """
    history: deque of dicts with keys:
        - "epoch"
        - "brightness", "cloud_fraction", "dark_fraction", "texture_index"
    Return slopes (per minute) using oldest sample inside window and current sample.
    """
    if len(history) < 2:
        return None, None, None, None

    window_sec = window_min * 60.0
    cutoff = now_epoch - window_sec

    # Find the oldest sample within the window
    oldest = None
    for item in history:
        if item["epoch"] >= cutoff:
            oldest = item
            break

    # If nothing in the last window, no trend
    if oldest is None:
        return None, None, None, None

    latest = history[-1]
    dt_min = (latest["epoch"] - oldest["epoch"]) / 60.0
    if dt_min <= 0:
        return None, None, None, None

    b_slope = (latest["brightness"] - oldest["brightness"]) / dt_min
    cf_slope = (latest["cloud_fraction"] - oldest["cloud_fraction"]) / dt_min
    df_slope = (latest["dark_fraction"] - oldest["dark_fraction"]) / dt_min
    tx_slope = (latest["texture_index"] - oldest["texture_index"]) / dt_min

    return b_slope, cf_slope, df_slope, tx_slope


def reopen_stream():
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        return None
    return cap


# =============== MAIN FUNCTION ==================

def run_sky_feature_extractor(csv_path=CSV_PATH):
    """
    Run the sky feature extraction loop and write features to a CSV file.

    This function NEVER talks to any database.
    It only appends to `csv_path` and finally returns the path (if the
    loop exits due to an error / stop).
    """
    cap = reopen_stream()
    if cap is None:
        raise RuntimeError("Could not open video stream")

    debug_log("Connected to phone sky camera.")

    csv_file, csv_writer = init_csv(csv_path)

    prev_gray_roi = None
    prev_time = None
    prev_optical = None
    m_per_px = None

    smooth_brightness = None
    smooth_cloud_fraction = None
    smooth_texture = None

    history = deque()
    history_max_sec = HISTORY_MAX_HOURS * 3600.0

    consecutive_errors = 0

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                consecutive_errors += 1
                debug_log("Failed to grab frame. Error count:", consecutive_errors)
                if consecutive_errors > MAX_CONSECUTIVE_ERRORS:
                    debug_log("Too many errors, attempting to reopen stream...")
                    cap.release()
                    time.sleep(2)
                    cap = reopen_stream()
                    if cap is None:
                        raise RuntimeError("Could not reopen video stream after errors")
                    debug_log("Stream reopened.")
                    consecutive_errors = 0
                time.sleep(1)
                continue

            consecutive_errors = 0

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape
            roi = gray[0:int(h * ROI_TOP_FRACTION), :]

            if m_per_px is None:
                m_per_px = compute_meters_per_pixel(w, HFOV_DEG, CLOUD_HEIGHT_M)

            optical = compute_cloud_optical_features(roi)

            smooth_brightness = ema(smooth_brightness, optical["brightness"])
            smooth_cloud_fraction = ema(smooth_cloud_fraction, optical["cloud_fraction"])
            smooth_texture = ema(smooth_texture, optical["texture_index"])

            now = time.time()
            now_struct = time.localtime(now)
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", now_struct)

            hour_local = now_struct.tm_hour + now_struct.tm_min / 60.0
            brightness_expected = expected_clear_sky_brightness(now_struct) if USE_TIME_OF_DAY_COMP else 0.0
            brightness_anomaly = (smooth_brightness - brightness_expected) if smooth_brightness is not None else None

            features = {
                "timestamp": timestamp_str,
                "brightness": smooth_brightness,
                "contrast": optical["contrast"],
                "dark_fraction": optical["dark_fraction"],
                "cloud_fraction": smooth_cloud_fraction,
                "texture_index": smooth_texture,
                "thickness_class": optical["thickness_class"],
                "darkening_rate": None,
                "cloud_area_growth_rate": None,
                "texture_growth_rate": None,
                "vertical_growth_flag": 0,
                "cloud_vx": None,
                "cloud_vy": None,
                "cloud_speed_kmh": None,
                "cloud_direction": None,
                "hour_local": hour_local,
                "brightness_expected_clear": brightness_expected,
                "brightness_anomaly": brightness_anomaly,
                "trend20_brightness_slope": None,
                "trend20_cloud_fraction_slope": None,
                "trend20_dark_fraction_slope": None,
                "trend20_texture_slope": None,
            }

            # Vertical growth between consecutive samples
            if prev_optical is not None and prev_time is not None:
                dt_sec = now - prev_time
                dt_min = dt_sec / 60.0

                dr, ca_gr, tx_gr, vflag = compute_vertical_growth(prev_optical, optical, dt_min)
                features["darkening_rate"] = dr
                features["cloud_area_growth_rate"] = ca_gr
                features["texture_growth_rate"] = tx_gr
                features["vertical_growth_flag"] = int(vflag)

            # Optical flow (cloud motion)
            if prev_gray_roi is not None and prev_time is not None:
                dt_sec = now - prev_time
                mean_vx, mean_vy, direction, km_per_h, px_per_sec = compute_flow(
                    prev_gray_roi, roi, dt_sec, m_per_px
                )
                if direction is not None:
                    features["cloud_vx"] = mean_vx
                    features["cloud_vy"] = mean_vy
                    features["cloud_speed_kmh"] = km_per_h
                    features["cloud_direction"] = direction

            # Update history for trend metrics
            history.append({
                "epoch": now,
                "brightness": features["brightness"],
                "cloud_fraction": features["cloud_fraction"],
                "dark_fraction": features["dark_fraction"],
                "texture_index": features["texture_index"],
            })

            # Trim history older than HISTORY_MAX_HOURS
            while history and (now - history[0]["epoch"] > history_max_sec):
                history.popleft()

            # Compute 20-minute trend slopes
            b_slope, cf_slope, df_slope, tx_slope = compute_trend_over_window(
                history, now, TREND_WINDOW_MIN
            )
            features["trend20_brightness_slope"] = b_slope
            features["trend20_cloud_fraction_slope"] = cf_slope
            features["trend20_dark_fraction_slope"] = df_slope
            features["trend20_texture_slope"] = tx_slope

            # Write to CSV
            csv_writer.writerow([
                features["timestamp"],
                features["brightness"],
                features["contrast"],
                features["dark_fraction"],
                features["cloud_fraction"],
                features["texture_index"],
                features["thickness_class"],
                features["darkening_rate"],
                features["cloud_area_growth_rate"],
                features["texture_growth_rate"],
                features["vertical_growth_flag"],
                features["cloud_vx"],
                features["cloud_vy"],
                features["cloud_speed_kmh"],
                features["cloud_direction"],
                features["hour_local"],
                features["brightness_expected_clear"],
                features["brightness_anomaly"],
                features["trend20_brightness_slope"],
                features["trend20_cloud_fraction_slope"],
                features["trend20_dark_fraction_slope"],
                features["trend20_texture_slope"],
            ])
            csv_file.flush()

            prev_gray_roi = roi.copy()
            prev_optical = optical
            prev_time = now

            # Enforce sampling interval (~1 min)
            elapsed = time.time() - loop_start
            sleep_time = SAMPLE_INTERVAL_SEC - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        if cap is not None:
            cap.release()
        csv_file.close()

    # Let caller know which CSV file has the data
    return csv_path


if __name__ == "__main__":
    # When run directly, just start the extractor and ignore the return value.
    run_sky_feature_extractor()
