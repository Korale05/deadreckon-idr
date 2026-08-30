import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.discover_sessions import discover_all_sessions
from src.io_utils import load_s_file, load_v_file, latlon_to_enu, speed_heading_to_velocity
from src.ins_mechanization import run_ins_mechanization
from src.evaluate import load_ai_corrector, run_ai_correction_batch, compute_ai_corrected_trajectory, compute_drift_pct
from src.ekf import run_ekf_fusion

def run_blackout_test(session, duration_s, corrector):
    try:
        raw_imu = load_s_file(session["s_path"])
        raw_gt = load_v_file(session["v_path"])
    except Exception as e:
        print(f"Error loading {session['name']}: {e}")
        return None

    # Align GT first
    gt_t = raw_gt["time"].to_numpy(dtype=float)
    gt_t = gt_t - gt_t[0]
    gt_lat = raw_gt["lat"].to_numpy(dtype=float)
    gt_lon = raw_gt["lon"].to_numpy(dtype=float)
    gt_speed = raw_gt["speed"].to_numpy(dtype=float)
    gt_heading = raw_gt["heading"].to_numpy(dtype=float)

    gt_vel_e, gt_vel_n = speed_heading_to_velocity(gt_speed, gt_heading)
    gt_pos_e, gt_pos_n = latlon_to_enu(gt_lat, gt_lon, gt_lat[0], gt_lon[0])

    imu_t = raw_imu["time"].to_numpy(dtype=float)
    if imu_t.max() > 1e5:
        imu_t = imu_t / 1000.0
    imu_t = imu_t - imu_t[0]
    n = len(imu_t)

    true_pos_x = np.interp(imu_t, gt_t, gt_pos_e)
    true_pos_y = np.interp(imu_t, gt_t, gt_pos_n)
    true_vel_x = np.interp(imu_t, gt_t, gt_vel_e)
    true_vel_y = np.interp(imu_t, gt_t, gt_vel_n)

    # Smartphone GPS
    gps_available = np.ones(n, dtype=bool)
    gps_pos = np.zeros((n, 2))
    gps_accuracy = np.full(n, 10.0)

    if "gps_lat" in raw_imu.columns and "gps_lon" in raw_imu.columns:
        gps_lat = raw_imu["gps_lat"].to_numpy(dtype=float)
        gps_lon = raw_imu["gps_lon"].to_numpy(dtype=float)
        gps_e, gps_n = latlon_to_enu(gps_lat, gps_lon, gt_lat[0], gt_lon[0])
        gps_pos[:, 0] = gps_e
        gps_pos[:, 1] = gps_n
        if "gps_accuracy" in raw_imu.columns:
            gps_accuracy = np.clip(raw_imu["gps_accuracy"].to_numpy(dtype=float), 1.0, 100.0)
        gps_available = (gps_accuracy < config.GNSS_BLACKOUT_THRESHOLD_M) & (np.abs(gps_lat) > 0.1)
    else:
        gps_pos[:, 0] = true_pos_x
        gps_pos[:, 1] = true_pos_y
        gps_accuracy[:] = 3.0

    ref_df = pd.DataFrame({
        "true_vel_x": true_vel_x,
        "true_vel_y": true_vel_y,
        "true_pos_x": true_pos_x,
        "true_pos_y": true_pos_y,
    })

    # INS with re-anchoring (full GPS)
    import hashlib
    session_seed = int(hashlib.md5(session["name"].encode()).hexdigest(), 16) % 1000000
    ins_df = run_ins_mechanization(raw_imu, ref_df=ref_df, gnss_available=gps_available, seed=session_seed)

    ai_corrections = run_ai_correction_batch(ins_df, corrector)

    # Simulated blackout indices
    # Start at 40% along the path
    start_idx = int(0.4 * n)
    # Convert duration to samples
    duration_samples = int(duration_s * config.SAMPLE_RATE_HZ)
    end_idx = min(n - 1, start_idx + duration_samples)

    # Blackout GPS mask
    blackout_gps = gps_available.copy()
    blackout_gps[start_idx:end_idx] = False

    skipped_updates = np.sum(gps_available[start_idx:end_idx])

    # Re-run INS with blackout GPS mask so INS does NOT re-anchor during blackout
    ins_df_blackout = run_ins_mechanization(
        raw_imu, ref_df=ref_df, gnss_available=blackout_gps, seed=session_seed
    )

    # Run EKF blackout
    ekf_result = run_ekf_fusion(
        ins_df_blackout,
        ai_corrections=ai_corrections,
        gnss_available=blackout_gps,
        gnss_pos=gps_pos,
        gnss_accuracy=gps_accuracy,
    )

    drift = compute_drift_pct(ekf_result["pos_x"], ekf_result["pos_y"], true_pos_x, true_pos_y)
    return {
        "skipped": skipped_updates,
        "drift": drift
    }

def main():
    sessions = discover_all_sessions()
    target_names = ["S1", "S2", "Vfa02", "Y1"]
    test_sessions = [s for s in sessions if s["name"] in target_names]

    corrector = load_ai_corrector()
    if corrector is None:
        print("Error: corrector model not found.")
        return

    durations = [20, 60, 120]

    print("======================================================================")
    print("  MULTIPLE BLACKOUT DURATION VALIDATION")
    print("======================================================================")
    print(f"  {'Session':<10} | {'Blackout':<10} | {'Skipped Updates':<15} | {'EKF Drift%':<10}")
    print("  " + "-"*56)

    for session in test_sessions:
        for dur in durations:
            res = run_blackout_test(session, dur, corrector)
            if res:
                print(f"  {session['name']:<10} | {dur:<8}s | {res['skipped']:<15} | {res['drift']:>9.2f}%")
    print("======================================================================")

if __name__ == "__main__":
    main()
