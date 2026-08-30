"""
io_utils.py
===========
Utilities for loading IO-VNBD CSV files with correct encoding and column
name resolution.  Handles the messy reality of the dataset: latin-1 encoding,
leading spaces in column names, and encoding-dependent special characters
(°, ², µ) that render differently across systems.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _resolve_column(df_columns: list, partial_name: str) -> str:
    """
    Find the actual DataFrame column name that matches a partial name.
    First tries exact match, then startswith match (case-insensitive).
    This handles the ² encoding issue and trailing spaces.
    """
    # Exact match (after strip)
    stripped = {c.strip(): c for c in df_columns}
    if partial_name in stripped:
        return stripped[partial_name]

    # Startswith match (handles cases like "ACCELEROMETER X" matching
    # "ACCELEROMETER X (m/s²) ")
    partial_lower = partial_name.lower().strip()
    for col in df_columns:
        if col.strip().lower().startswith(partial_lower):
            return col

    # Contains match as last resort
    for col in df_columns:
        if partial_lower in col.strip().lower():
            return col

    raise KeyError(
        f"Column matching '{partial_name}' not found. "
        f"Available columns: {df_columns}"
    )


def load_s_file(path: str) -> pd.DataFrame:
    """
    Load an S-file (smartphone IMU data) with correct encoding and
    resolved column names.  Returns a DataFrame with clean, standardized
    column keys matching config.IMU_COLUMNS keys.
    
    Detects and unwraps any resets/overflows in the timestamp column
    where the time jumps backward.
    """
    df = pd.read_csv(path, encoding=config.CSV_ENCODING)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Build renamed DataFrame with standard keys
    renamed = pd.DataFrame()
    for key, partial in config.IMU_COLUMNS.items():
        try:
            actual_col = _resolve_column(list(df.columns), partial)
            renamed[key] = df[actual_col].values
        except KeyError:
            # Some columns are optional (e.g., GPS might be missing)
            pass

    # Timestamp unwrapping: check for negative jumps in the time column
    if "time" in renamed.columns:
        times = renamed["time"].to_numpy(dtype=float)
        unwrapped = times.copy()
        accumulated_offset = 0.0
        n_jumps = 0
        for i in range(1, len(unwrapped)):
            diff = times[i] - times[i-1]
            if diff < 0:
                # Phone clock reset/overflow detected!
                # We add the last valid time minus the reset value, plus a nominal 100ms dt
                offset = times[i-1] - times[i] + 100.0
                accumulated_offset += offset
                n_jumps += 1
            unwrapped[i] += accumulated_offset
        if n_jumps > 0:
            print(f"[io_utils] Corrected {n_jumps} time reset/overflow jump(s) in S-file: {os.path.basename(path)}. "
                  f"Total accumulated offset added: {accumulated_offset / 1000.0:.2f} seconds.")
        renamed["time"] = unwrapped

    return renamed


def load_v_file(path: str) -> pd.DataFrame:
    """
    Load a V-file (vehicle CAN-bus ground truth) with correct encoding and
    resolved column names.  Returns a DataFrame with clean, standardized
    column keys matching config.GT_COLUMNS keys.
    """
    df = pd.read_csv(path, encoding=config.CSV_ENCODING)
    df.columns = [c.strip() for c in df.columns]

    renamed = pd.DataFrame()
    for key, partial in config.GT_COLUMNS.items():
        try:
            actual_col = _resolve_column(list(df.columns), partial)
            renamed[key] = df[actual_col].values
        except KeyError:
            pass

    return renamed


def latlon_to_enu(lat: np.ndarray, lon: np.ndarray,
                  lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude arrays to local East-North-Up (ENU) meters
    relative to a reference point (lat0, lon0).
    Uses equirectangular approximation (accurate enough for <50 km drives).

    Returns (east_m, north_m) — both in meters.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    east = config.EARTH_RADIUS_M * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    north = config.EARTH_RADIUS_M * (lat_rad - lat0_rad)
    return east, north


def speed_heading_to_velocity(speed_kmh: np.ndarray,
                               heading_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert scalar speed (km/h) + heading (degrees, 0=N clockwise) to
    velocity components in the ENU frame.

    Returns (vel_east, vel_north) in m/s.
    """
    speed_ms = speed_kmh / 3.6
    heading_rad = np.radians(heading_deg)
    vel_east = speed_ms * np.sin(heading_rad)
    vel_north = speed_ms * np.cos(heading_rad)
    return vel_east, vel_north


if __name__ == "__main__":
    # Quick test: load one S-file and one V-file
    from src.discover_sessions import discover_all_sessions

    sessions = discover_all_sessions()
    if sessions:
        s = sessions[0]
        print(f"Testing with session: {s['name']}")
        print(f"\nLoading S-file: {s['s_path']}")
        imu = load_s_file(s["s_path"])
        print(f"  Columns: {list(imu.columns)}")
        print(f"  Shape: {imu.shape}")
        print(f"  First row:\n{imu.iloc[0]}\n")

        print(f"Loading V-file: {s['v_path']}")
        gt = load_v_file(s["v_path"])
        print(f"  Columns: {list(gt.columns)}")
        print(f"  Shape: {gt.shape}")
        print(f"  First row:\n{gt.iloc[0]}")
