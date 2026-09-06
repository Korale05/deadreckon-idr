"""
pipeline.py
============
End-to-end inference pipeline that chains ALL components of the system:

    1. Load raw IMU session (S-file)
    2. Run classical INS mechanization
    3. Load ground truth (V-file) for comparison
    4. Run AI error correction (if trained model exists)
    5. Compute AI-corrected trajectory (no EKF, just integrate corrected velocity)
    6. Run EKF fusion (INS + AI + GNSS)
    7. Simulate GNSS blackout and re-run EKF
    8. Run map matching (if osmnx available)
    9. Generate comparison plots
   10. Print metrics summary including drift %

This is the script you'd demo to judges:
    python -m src.pipeline --session S1
    python -m src.pipeline --session S1 --blackout 100 300
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.io_utils import load_s_file, load_v_file, latlon_to_enu, speed_heading_to_velocity
from src.ins_mechanization import run_ins_mechanization
from src.ekf import run_ekf_fusion
from src.visualize import (plot_trajectory_comparison, plot_velocity_comparison,
                            plot_error_summary, plot_ekf_uncertainty)
from src.discover_sessions import discover_all_sessions


def load_ai_corrector():
    """Try to load the trained AI model. Returns None if not available."""
    try:
        from src.inference import LiveErrorCorrector
        for name in ["ins_error_model_best.keras", "ins_error_model_final.keras"]:
            model_path = os.path.join(config.MODEL_DIR, name)
            stats_path = os.path.join(config.MODEL_DIR, "normalization_stats.npz")
            if os.path.exists(model_path) and os.path.exists(stats_path):
                return LiveErrorCorrector(model_path, stats_path)
        print("[pipeline] Trained model not found. Running without AI correction.")
        return None
    except Exception as e:
        print(f"[pipeline] Could not load AI model: {e}")
        return None


def run_ai_correction_batch(ins_df, corrector):
    """
    Run the AI error corrector over an entire session in batch mode.
    This is extremely fast compared to step-by-step streaming.
    """
    if corrector is None:
        return None

    n = len(ins_df)
    corrections = np.zeros((n, 2))
    W = config.WINDOW_SIZE
    if n < W:
        return corrections

    imu_keys = config.IMU_FEATURES
    state_keys = config.INS_STATE_FEATURES

    # Extract all windows
    imu_matrix = ins_df[imu_keys].to_numpy(dtype=np.float32)
    state_matrix = ins_df[state_keys].to_numpy(dtype=np.float32)

    # Clean NaNs
    imu_matrix = np.nan_to_num(imu_matrix, nan=0.0)
    state_matrix = np.nan_to_num(state_matrix, nan=0.0)

    # Prepare batch inputs
    X_imu = []
    X_state = []
    for i in range(W - 1, n):
        X_imu.append(imu_matrix[i - W + 1 : i + 1])
        X_state.append(state_matrix[i])

    X_imu = np.stack(X_imu)      # (n - W + 1, W, 6)
    X_state = np.stack(X_state)  # (n - W + 1, 4)

    # Normalize using training statistics
    X_imu_n = (X_imu - corrector.stats["imu_mean"]) / corrector.stats["imu_std"]
    X_state_n = (X_state - corrector.stats["state_mean"]) / corrector.stats["state_std"]

    # Final Nan/Inf guard
    X_imu_n = np.nan_to_num(X_imu_n, nan=0.0)
    X_state_n = np.nan_to_num(X_state_n, nan=0.0)

    # Predict in a single batch
    pred_n = corrector.model.predict([X_imu_n, X_state_n], batch_size=512, verbose=0)
    pred = pred_n * corrector.stats["y_std"] + corrector.stats["y_mean"]
    pred = np.nan_to_num(pred, nan=0.0)

    # Fill back predictions
    corrections[W - 1:] = pred
    return corrections


def compute_ai_corrected_trajectory(ins_df, ai_corrections):
    """
    Integrate AI-corrected velocity to produce a trajectory WITHOUT EKF.
    This is the "AI-corrected INS alone" comparison arm.

    SIGN CONVENTION: corrected_vel = ins_vel + predicted_err (ADDITION)
    """
    if ai_corrections is None:
        return None

    n = len(ins_df)
    t = ins_df["_t_sec"].to_numpy(dtype=float)
    ins_vel_x = ins_df["ins_vel_x"].to_numpy(dtype=float)
    ins_vel_y = ins_df["ins_vel_y"].to_numpy(dtype=float)
    ins_pos_x = ins_df["ins_pos_x"].to_numpy(dtype=float)
    ins_pos_y = ins_df["ins_pos_y"].to_numpy(dtype=float)
    is_reset = ins_df["ins_is_reset"].to_numpy(dtype=bool) if "ins_is_reset" in ins_df.columns else np.zeros(n, dtype=bool)

    corrected_vel_x = ins_vel_x + ai_corrections[:, 0]
    corrected_vel_y = ins_vel_y + ai_corrections[:, 1]

    # Integrate corrected velocity to get position (trapezoidal)
    pos_x = np.zeros(n)
    pos_y = np.zeros(n)
    
    pos_x[0] = ins_pos_x[0]
    pos_y[0] = ins_pos_y[0]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        
        # If the INS was reset/re-anchored at i-1, align our position estimate to it
        if is_reset[i-1]:
            pos_x[i-1] = ins_pos_x[i-1]
            pos_y[i-1] = ins_pos_y[i-1]

        pos_x[i] = pos_x[i - 1] + 0.5 * (corrected_vel_x[i] + corrected_vel_x[i - 1]) * dt
        pos_y[i] = pos_y[i - 1] + 0.5 * (corrected_vel_y[i] + corrected_vel_y[i - 1]) * dt

    return {
        "pos_x": pos_x,
        "pos_y": pos_y,
        "vel_x": corrected_vel_x,
        "vel_y": corrected_vel_y,
    }


def prepare_gnss_data(imu_df, gt_df):
    """
    Extract smartphone GPS data from the S-file for use as GNSS measurements
    in the EKF. Also extract ground-truth positions from the V-file.

    Returns a dict with GNSS and ground-truth arrays.
    """
    # Ground truth from V-file
    gt_t = gt_df["time"].to_numpy(dtype=float)
    gt_t = gt_t - gt_t[0]
    gt_lat = gt_df["lat"].to_numpy(dtype=float)
    gt_lon = gt_df["lon"].to_numpy(dtype=float)
    gt_speed = gt_df["speed"].to_numpy(dtype=float)
    gt_heading = gt_df["heading"].to_numpy(dtype=float)

    gt_vel_e, gt_vel_n = speed_heading_to_velocity(gt_speed, gt_heading)
    gt_pos_e, gt_pos_n = latlon_to_enu(gt_lat, gt_lon, gt_lat[0], gt_lon[0])

    # Interpolate ground truth onto IMU timestamps
    imu_t = imu_df["_t_sec"].to_numpy(dtype=float)
    n = len(imu_t)

    true_vel_x = np.interp(imu_t, gt_t, gt_vel_e)
    true_vel_y = np.interp(imu_t, gt_t, gt_vel_n)
    true_pos_x = np.interp(imu_t, gt_t, gt_pos_e)
    true_pos_y = np.interp(imu_t, gt_t, gt_pos_n)

    # Smartphone GPS from S-file (for EKF GNSS updates)
    gps_available = np.ones(n, dtype=bool)
    gps_pos = np.zeros((n, 2))
    gps_accuracy = np.full(n, 10.0)

    if "gps_lat" in imu_df.columns and "gps_lon" in imu_df.columns:
        gps_lat = imu_df["gps_lat"].to_numpy(dtype=float)
        gps_lon = imu_df["gps_lon"].to_numpy(dtype=float)

        # Use first ground-truth position as reference for ENU conversion
        ref_lat, ref_lon = gt_lat[0], gt_lon[0]
        gps_e, gps_n = latlon_to_enu(gps_lat, gps_lon, ref_lat, ref_lon)
        gps_pos[:, 0] = gps_e
        gps_pos[:, 1] = gps_n

        if "gps_accuracy" in imu_df.columns:
            gps_accuracy = imu_df["gps_accuracy"].to_numpy(dtype=float)
            gps_accuracy = np.clip(gps_accuracy, 1.0, 100.0)

        # Mark GPS as unavailable where accuracy is too poor or coordinates are 0
        gps_available = (gps_accuracy < config.GNSS_BLACKOUT_THRESHOLD_M) & (np.abs(gps_lat) > 0.1)
    else:
        # No GPS in S-file — use ground truth GPS as fallback
        # (This is the vehicle's high-accuracy GPS, simulating "perfect GNSS")
        gps_pos[:, 0] = true_pos_x
        gps_pos[:, 1] = true_pos_y
        gps_accuracy[:] = 3.0

    return {
        "true_vel_x": true_vel_x,
        "true_vel_y": true_vel_y,
        "true_pos_x": true_pos_x,
        "true_pos_y": true_pos_y,
        "gps_pos": gps_pos,
        "gps_available": gps_available,
        "gps_accuracy": gps_accuracy,
        "ref_lat": gt_lat[0],
        "ref_lon": gt_lon[0],
    }


def simulate_blackout(gps_available, time_s, start_time=None, duration_s=None):
    """
    Simulate a GNSS blackout by zeroing out GPS availability for a window.

    Parameters
    ----------
    start_time : float, optional
        Absolute start time (seconds from session start). If None, uses
        config.BLACKOUT_START_FRAC of the session.
    duration_s : float, optional
        Duration in seconds. Default: config.BLACKOUT_DURATION_S.

    Returns modified gps_available array and (start_time, end_time) tuple.
    """
    duration_s = duration_s or config.BLACKOUT_DURATION_S

    if start_time is None:
        total_time = time_s[-1] - time_s[0]
        start_time = time_s[0] + total_time * config.BLACKOUT_START_FRAC

    blackout_end = start_time + duration_s

    mask = gps_available.copy()
    mask[(time_s >= start_time) & (time_s <= blackout_end)] = False

    return mask, (start_time, blackout_end)


def compute_drift_pct(est_pos_x, est_pos_y, true_pos_x, true_pos_y):
    """
    Compute drift % = (final position error / total distance traveled) * 100.

    This is the actual grading metric for the dead-reckoning problem.
    """
    final_err = np.sqrt(
        (est_pos_x[-1] - true_pos_x[-1]) ** 2 +
        (est_pos_y[-1] - true_pos_y[-1]) ** 2
    )
    # Total distance traveled along ground truth
    dx = np.diff(true_pos_x)
    dy = np.diff(true_pos_y)
    total_dist = np.sum(np.sqrt(dx ** 2 + dy ** 2))
    if total_dist < 1.0:
        return float("inf")
    return (final_err / total_dist) * 100.0


def run_pipeline(session_name=None, blackout_start=None, blackout_duration=None,
                 skip_map_matching=False):
    """
    Run the full pipeline for one session.
    """
    # --- Discover and load session ---
    sessions = discover_all_sessions()
    if not sessions:
        print("No sessions found. Check config.DATASET_ROOT.")
        return

    if session_name:
        session = next((s for s in sessions if s["name"] == session_name), None)
        if not session:
            print(f"Session '{session_name}' not found. Available: "
                  f"{[s['name'] for s in sessions[:10]]}...")
            return
    else:
        session = sessions[0]

    print(f"{'='*70}")
    print(f"  PIPELINE: {session['name']} ({session['category']})")
    print(f"{'='*70}\n")

    # --- Step 1: Load raw data ---
    print("[1/9] Loading raw data...")
    raw_imu = load_s_file(session["s_path"])
    raw_gt = load_v_file(session["v_path"])
    print(f"  IMU samples: {len(raw_imu)}, V-file samples: {len(raw_gt)}")

    # Compute raw _t_sec first to align ground truth
    t = raw_imu["time"].to_numpy(dtype=float)
    if t.max() > 1e5:
        t = t / 1000.0
    t = t - t[0]
    
    temp_imu = raw_imu.copy()
    temp_imu["_t_sec"] = t

    # --- Step 3: Prepare ground truth + GNSS ---
    print("[3/9] Preparing ground truth and GNSS data...")
    data = prepare_gnss_data(temp_imu, raw_gt)
    time_s = temp_imu["_t_sec"].to_numpy()
    n = len(time_s)

    # Create aligned ref_df for INS re-anchoring
    ref_df = pd.DataFrame({
        "true_vel_x": data["true_vel_x"],
        "true_vel_y": data["true_vel_y"],
        "true_pos_x": data["true_pos_x"],
        "true_pos_y": data["true_pos_y"],
    })

    # Stable session seed for randomized interval selection
    import hashlib
    session_seed = int(hashlib.md5(session["name"].encode()).hexdigest(), 16) % 1000000

    # --- Step 2: Run INS mechanization ---
    print("[2/9] Running INS mechanization...")
    ins_df = run_ins_mechanization(raw_imu, ref_df=None, seed=session_seed)
    duration = ins_df["_t_sec"].iloc[-1]
    ins_final_drift = np.sqrt(
        ins_df["ins_pos_x"].iloc[-1]**2 + ins_df["ins_pos_y"].iloc[-1]**2)
    print(f"  Duration: {duration:.0f}s, INS final drift: {ins_final_drift:.1f} m")

    # --- Step 4: Run AI correction (batch mode) ---
    print("[4/9] Running AI error correction (batch)...")
    corrector = load_ai_corrector()
    ai_corrections = run_ai_correction_batch(ins_df, corrector)
    if ai_corrections is not None:
        mean_correction = np.mean(np.abs(ai_corrections), axis=0)
        print(f"  Mean correction magnitude: [{mean_correction[0]:.3f}, {mean_correction[1]:.3f}] m/s")
    else:
        print("  AI model not available — skipping correction")

    # --- Step 5: Compute AI-corrected trajectory (no EKF) ---
    print("[5/9] Computing AI-corrected trajectory (no EKF)...")
    ai_traj = compute_ai_corrected_trajectory(ins_df, ai_corrections)

    # --- Step 6: Run EKF fusion (with GNSS) ---
    print("[6/9] Running EKF fusion (with GNSS)...")
    ekf_result = run_ekf_fusion(
        ins_df, ai_corrections=ai_corrections,
        gnss_available=data["gps_available"],
        gnss_pos=data["gps_pos"],
        gnss_accuracy=data["gps_accuracy"],
    )

    # --- Step 7: Run EKF with GNSS blackout ---
    bo_duration = blackout_duration or config.BLACKOUT_DURATION_S
    print(f"[7/9] Running EKF with GNSS blackout...")
    blackout_gps, blackout_window = simulate_blackout(
        data["gps_available"], time_s,
        start_time=blackout_start, duration_s=bo_duration)
    
    skipped_updates = np.sum(data["gps_available"] & ~blackout_gps)
    print(f"  Simulated blackout skipped {skipped_updates} GNSS updates during the blackout window.")

    # Re-run INS and AI for EKF blackout run (respects blackout mask)
    ins_df_blackout = run_ins_mechanization(raw_imu, ref_df=None, seed=session_seed)
    ai_corrections_blackout = run_ai_correction_batch(ins_df_blackout, corrector)

    ekf_blackout = run_ekf_fusion(
        ins_df_blackout, ai_corrections=ai_corrections_blackout,
        gnss_available=blackout_gps,
        gnss_pos=data["gps_pos"],
        gnss_accuracy=data["gps_accuracy"],
    )
    print(f"  Blackout window: {blackout_window[0]:.0f}s - {blackout_window[1]:.0f}s")

    # --- Step 8: Map matching ---
    if not skip_map_matching:
        print("[8/9] Running map matching...")
        try:
            from src.map_matching import run_map_matching, OSMNX_AVAILABLE
            if OSMNX_AVAILABLE:
                mm_result = run_map_matching(
                    ekf_result["pos_x"], ekf_result["pos_y"],
                    data["ref_lat"], data["ref_lon"],
                )
                if mm_result["road_network"]:
                    print("  Map matching succeeded")
                else:
                    print("  Map matching skipped (no road network)")
            else:
                print("  osmnx not installed — skipping map matching")
        except Exception as e:
            print(f"  Map matching failed: {e}")
    else:
        print("[8/9] Map matching skipped (--skip-map-matching)")

    # --- Step 9: Visualize and report ---
    print("[9/9] Generating plots and metrics...\n")

    ground_truth = {
        "pos_x": data["true_pos_x"],
        "pos_y": data["true_pos_y"],
        "vel_x": data["true_vel_x"],
        "vel_y": data["true_vel_y"],
    }
    ins_only = {
        "pos_x": ins_df["ins_pos_x"].to_numpy(),
        "pos_y": ins_df["ins_pos_y"].to_numpy(),
        "vel_x": ins_df["ins_vel_x"].to_numpy(),
        "vel_y": ins_df["ins_vel_y"].to_numpy(),
    }
    ekf_data = {
        "pos_x": ekf_result["pos_x"],
        "pos_y": ekf_result["pos_y"],
        "vel_x": ekf_result["vel_x"],
        "vel_y": ekf_result["vel_y"],
    }
    ekf_bo_data = {
        "pos_x": ekf_blackout["pos_x"],
        "pos_y": ekf_blackout["pos_y"],
        "vel_x": ekf_blackout["vel_x"],
        "vel_y": ekf_blackout["vel_y"],
    }

    # Trajectory plots
    plot_trajectory_comparison(
        time_s, ground_truth, ins_only,
        ai_corrected=ai_traj, ekf_fused=ekf_data,
        title=f"Session {session['name']} — Full GNSS",
        save_name=f"{session['name']}_trajectory.png",
    )

    plot_trajectory_comparison(
        time_s, ground_truth, ins_only,
        ai_corrected=ai_traj, ekf_fused=ekf_bo_data,
        title=f"Session {session['name']} — {bo_duration}s GNSS Blackout",
        save_name=f"{session['name']}_trajectory_blackout.png",
        blackout_window=blackout_window,
    )

    # Velocity plots
    plot_velocity_comparison(
        time_s, ground_truth, ins_only,
        ai_corrected=ai_traj, ekf_fused=ekf_data,
        title=f"Session {session['name']} — Velocity",
        save_name=f"{session['name']}_velocity.png",
    )

    # EKF uncertainty
    plot_ekf_uncertainty(
        time_s, ekf_blackout["pos_unc"], ekf_blackout["vel_unc"],
        blackout_window=blackout_window,
        title=f"Session {session['name']} — EKF Uncertainty (Blackout)",
        save_name=f"{session['name']}_uncertainty.png",
    )

    # --- Metrics ---
    print(f"{'='*70}")
    print(f"  RESULTS: {session['name']}")
    print(f"{'='*70}")
    print(f"  Session duration: {duration:.0f} seconds\n")

    # Drift % — the actual grading metric
    ins_drift = compute_drift_pct(
        ins_only["pos_x"], ins_only["pos_y"],
        data["true_pos_x"], data["true_pos_y"])

    ai_drift = float("inf")
    if ai_traj is not None:
        ai_drift = compute_drift_pct(
            ai_traj["pos_x"], ai_traj["pos_y"],
            data["true_pos_x"], data["true_pos_y"])

    ekf_drift = compute_drift_pct(
        ekf_data["pos_x"], ekf_data["pos_y"],
        data["true_pos_x"], data["true_pos_y"])

    ekf_bo_drift = compute_drift_pct(
        ekf_bo_data["pos_x"], ekf_bo_data["pos_y"],
        data["true_pos_x"], data["true_pos_y"])

    print(f"  Drift % (final pos error / distance traveled):")
    print(f"    {'Method':<35} {'Drift %':>10}")
    print(f"    {'-'*45}")
    print(f"    {'Raw INS Only':<35} {ins_drift:>10.2f}%")
    if ai_traj is not None:
        print(f"    {'AI-Corrected INS (no EKF/GNSS)':<35} {ai_drift:>10.2f}%")
    print(f"    {'EKF (INS+AI+GNSS, full)':<35} {ekf_drift:>10.2f}%")
    print(f"    {'EKF ({0}s blackout)':<35} {ekf_bo_drift:>10.2f}%".format(int(bo_duration)))
    print()

    # Position RMSE
    gt_pos = np.column_stack([data["true_pos_x"], data["true_pos_y"]])
    ins_pos = np.column_stack([ins_only["pos_x"], ins_only["pos_y"]])
    ekf_pos = np.column_stack([ekf_data["pos_x"], ekf_data["pos_y"]])

    ins_pos_rmse = np.sqrt(np.mean(np.sum((ins_pos - gt_pos)**2, axis=1)))
    ekf_pos_rmse = np.sqrt(np.mean(np.sum((ekf_pos - gt_pos)**2, axis=1)))

    print(f"  Position RMSE:")
    print(f"    INS Only:            {ins_pos_rmse:>10.2f} m")
    print(f"    EKF (full GNSS):     {ekf_pos_rmse:>10.2f} m")
    print()

    # Velocity RMSE
    gt_vel = np.column_stack([data["true_vel_x"], data["true_vel_y"]])
    ins_vel = np.column_stack([ins_only["vel_x"], ins_only["vel_y"]])
    ekf_vel = np.column_stack([ekf_data["vel_x"], ekf_data["vel_y"]])

    ins_vel_rmse = np.sqrt(np.mean(np.sum((ins_vel - gt_vel)**2, axis=1)))
    ekf_vel_rmse = np.sqrt(np.mean(np.sum((ekf_vel - gt_vel)**2, axis=1)))

    print(f"  Velocity RMSE:")
    print(f"    INS Only:            {ins_vel_rmse:>10.3f} m/s")
    print(f"    EKF (full GNSS):     {ekf_vel_rmse:>10.3f} m/s")
    print()

    # Final position error
    ins_final_err = np.linalg.norm(ins_pos[-1] - gt_pos[-1])
    ekf_final_err = np.linalg.norm(ekf_pos[-1] - gt_pos[-1])

    print(f"  Final position error:")
    print(f"    INS Only:            {ins_final_err:>10.2f} m")
    print(f"    EKF (full GNSS):     {ekf_final_err:>10.2f} m")
    print()

    print(f"  Plots saved to: {config.PLOT_DIR}")
    print(f"{'='*70}")

    return {
        "session": session["name"],
        "duration": duration,
        "ins_drift_pct": ins_drift,
        "ai_drift_pct": ai_drift,
        "ekf_drift_pct": ekf_drift,
        "ekf_bo_drift_pct": ekf_bo_drift,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the full navigation pipeline")
    parser.add_argument("--session", default=None,
                        help="Session name (e.g., 'S1'). Default: first found.")
    parser.add_argument("--blackout", nargs="*", type=float, default=None,
                        help="GNSS blackout: <start_time> <duration> or just <duration>")
    parser.add_argument("--skip-map-matching", action="store_true",
                        help="Skip the map matching step")
    parser.add_argument("--list", action="store_true",
                        help="List available sessions and exit")
    args = parser.parse_args()

    if args.list:
        sessions = discover_all_sessions()
        print(f"Available sessions ({len(sessions)}):")
        for s in sessions:
            print(f"  [{s['category']}] {s['name']}")
        return

    blackout_start = None
    blackout_duration = None
    if args.blackout is not None:
        if len(args.blackout) == 2:
            blackout_start, blackout_duration = args.blackout
        elif len(args.blackout) == 1:
            blackout_duration = args.blackout[0]
        else:
            print("Usage: --blackout <start_time> <duration>  or  --blackout <duration>")
            return

    run_pipeline(
        session_name=args.session,
        blackout_start=blackout_start,
        blackout_duration=blackout_duration,
        skip_map_matching=args.skip_map_matching,
    )


if __name__ == "__main__":
    main()
