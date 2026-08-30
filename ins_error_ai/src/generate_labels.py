"""
generate_labels.py
====================
STEP 2 of the plan: turn (INS estimate) + (ground truth) into training labels.

For every session discovered by discover_sessions.py:
    - S-file = smartphone IMU data
    - V-file = vehicle CAN-bus ground truth (speed, heading, lat/lon, wheel speeds)

This script:
    1. Runs the classical INS mechanization over the S-file to get the INS's
       own (flawed) running velocity/position estimate.
    2. Loads the matching V-file and derives ground-truth velocity components
       and position from speed+heading+lat/lon.
    3. Aligns ground truth onto IMU timestamps via interpolation.
    4. Computes error = true_value - ins_estimate at every timestep.
    5. Saves one merged, per-session CSV to data/processed/ containing:
       raw IMU columns, ins_vel_*, ins_pos_*, true_vel_*, true_pos_*,
       err_vel_*, err_pos_*
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.io_utils import load_s_file, load_v_file, latlon_to_enu, speed_heading_to_velocity
from src.ins_mechanization import run_ins_mechanization
from src.discover_sessions import discover_all_sessions


def derive_ground_truth(gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the V-file's scalar speed + heading + lat/lon, derive:
        - vel_east, vel_north (m/s) — velocity components in ENU frame
        - pos_east, pos_north (m) — position in local ENU meters from session start
        - time_s — time in seconds from session start

    Returns a new DataFrame with these derived columns plus the raw V-file time.
    """
    # Time: V-file uses "Time Since Start of Day (seconds)" — normalize to
    # session-relative by subtracting the first value
    t = gt_df["time"].to_numpy(dtype=float)
    t = t - t[0]

    # Velocity: speed (km/h) + heading (degrees) -> ENU components (m/s)
    speed = gt_df["speed"].to_numpy(dtype=float)
    heading = gt_df["heading"].to_numpy(dtype=float)
    vel_east, vel_north = speed_heading_to_velocity(speed, heading)

    # Position: lat/lon -> local ENU meters from session start
    lat = gt_df["lat"].to_numpy(dtype=float)
    lon = gt_df["lon"].to_numpy(dtype=float)
    pos_east, pos_north = latlon_to_enu(lat, lon, lat[0], lon[0])

    return pd.DataFrame({
        "gt_time_s": t,
        "true_vel_x": vel_east,
        "true_vel_y": vel_north,
        "true_pos_x": pos_east,
        "true_pos_y": pos_north,
        "true_speed_ms": speed / 3.6,
        "true_heading": heading,
    })


def align_ground_truth(ins_df: pd.DataFrame, gt_derived: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate derived ground-truth values onto the IMU's own timestamps.
    The V-file logs at 10 Hz, the S-file also at ~10 Hz, but they're not
    necessarily synchronized — interpolation handles this.
    """
    imu_t = ins_df["_t_sec"].to_numpy(dtype=float)
    gt_t = gt_derived["gt_time_s"].to_numpy(dtype=float)

    aligned = pd.DataFrame(index=ins_df.index)
    for col in ["true_vel_x", "true_vel_y", "true_pos_x", "true_pos_y",
                "true_speed_ms", "true_heading"]:
        aligned[col] = np.interp(imu_t, gt_t, gt_derived[col].to_numpy(dtype=float))
    return aligned


def build_session_labels(s_path: str, v_path: str) -> pd.DataFrame:
    """
    Full pipeline for one session: load raw -> INS -> ground truth -> error labels.
    """
    import hashlib
    # Load raw data using IO utilities
    raw_imu = load_s_file(s_path)
    raw_gt = load_v_file(v_path)

    # Derive ground truth from V-file
    gt_derived = derive_ground_truth(raw_gt)

    # Compute normalized _t_sec for raw_imu to align ground truth
    t = raw_imu["time"].to_numpy(dtype=float)
    if t.max() > 1e5:
        t = t / 1000.0
    t = t - t[0]
    
    # Temporarily create ins_df metadata for align_ground_truth
    temp_df = raw_imu.copy()
    temp_df["_t_sec"] = t
    
    # Align ground truth to IMU timestamps first
    true_df = align_ground_truth(temp_df, gt_derived)

    # Use a session-specific stable seed for randomized re-anchoring interval selection
    session_seed = int(hashlib.md5(os.path.basename(s_path).encode()).hexdigest(), 16) % 1000000

    # Step 1: Run INS mechanization WITH re-anchoring back to ground truth
    ins_df = run_ins_mechanization(raw_imu, ref_df=true_df, seed=session_seed)

    # Step 4: Merge and compute error labels
    merged = pd.concat([
        ins_df.reset_index(drop=True),
        true_df.reset_index(drop=True)
    ], axis=1)

    # THE CORE LABEL: error = true - INS estimate
    merged["err_vel_x"] = merged["true_vel_x"] - merged["ins_vel_x"]
    merged["err_vel_y"] = merged["true_vel_y"] - merged["ins_vel_y"]
    merged["err_pos_x"] = merged["true_pos_x"] - merged["ins_pos_x"]
    merged["err_pos_y"] = merged["true_pos_y"] - merged["ins_pos_y"]

    return merged


def main():
    sessions = discover_all_sessions()
    if not sessions:
        print("No sessions found. Check config.DATASET_ROOT.")
        print(f"Looking in: {config.DATASET_ROOT}")
        return

    print(f"Found {len(sessions)} sessions. Processing...\n")

    success_count = 0
    for session in sessions:
        name = session["name"]
        try:
            print(f"[processing] {name} ({session['category']})")
            labeled = build_session_labels(session["s_path"], session["v_path"])

            out_path = os.path.join(config.PROCESSED_DIR, f"{name}_labeled.csv")
            labeled.to_csv(out_path, index=False)

            rmse_vel = np.sqrt(np.mean(
                labeled["err_vel_x"]**2 + labeled["err_vel_y"]**2
            ))
            rmse_pos = np.sqrt(np.mean(
                labeled["err_pos_x"]**2 + labeled["err_pos_y"]**2
            ))
            duration = labeled["_t_sec"].iloc[-1]
            print(f"    duration: {duration:.0f}s | "
                  f"INS vel-error RMSE: {rmse_vel:.3f} m/s | "
                  f"INS pos-error RMSE: {rmse_pos:.1f} m")
            print(f"    saved -> {out_path}")
            success_count += 1
        except Exception as e:
            print(f"    [ERROR] {name}: {e}")

    print(f"\nDone. Successfully processed {success_count}/{len(sessions)} sessions.")
    print(f"Labeled files in: {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
