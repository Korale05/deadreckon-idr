"""
evaluate.py
===========
Run the full pipeline across ALL available sessions and report the three-way
comparison that demonstrates the AI is earning its place:

    1. Raw INS alone (no AI, no EKF, no GNSS)
    2. AI-corrected INS alone (no EKF, no GNSS)
    3. Full pipeline (INS + AI + EKF + GNSS)

Reports drift % (final position error / total distance traveled) for each
method across all sessions.  This is the actual evidence table.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.io_utils import load_s_file, load_v_file, latlon_to_enu, speed_heading_to_velocity
from src.ins_mechanization import run_ins_mechanization
from src.ekf import run_ekf_fusion
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
        return None
    except Exception as e:
        print(f"Error loading AI corrector: {e}")
        return None


def run_ai_correction_batch(ins_df, corrector):
    """Run AI correction in batch mode for speed."""
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

    # Normalize
    X_imu_n = (X_imu - corrector.stats["imu_mean"]) / corrector.stats["imu_std"]
    X_state_n = (X_state - corrector.stats["state_mean"]) / corrector.stats["state_std"]

    # Final Nan guard
    X_imu_n = np.nan_to_num(X_imu_n, nan=0.0)
    X_state_n = np.nan_to_num(X_state_n, nan=0.0)

    # Predict in a single batch
    pred_n = corrector.model.predict([X_imu_n, X_state_n], batch_size=512, verbose=0)
    pred = pred_n * corrector.stats["y_std"] + corrector.stats["y_mean"]
    pred = np.nan_to_num(pred, nan=0.0)

    # Fill back
    corrections[W - 1:] = pred
    return corrections


def compute_ai_corrected_trajectory(ins_df, ai_corrections):
    """Integrate AI-corrected velocity (no EKF). corrected = ins + err."""
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
    
    pos_x = np.zeros(n)
    pos_y = np.zeros(n)
    pos_x[0] = ins_pos_x[0]
    pos_y[0] = ins_pos_y[0]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if is_reset[i-1]:
            pos_x[i-1] = ins_pos_x[i-1]
            pos_y[i-1] = ins_pos_y[i-1]

        pos_x[i] = pos_x[i - 1] + 0.5 * (corrected_vel_x[i] + corrected_vel_x[i - 1]) * dt
        pos_y[i] = pos_y[i - 1] + 0.5 * (corrected_vel_y[i] + corrected_vel_y[i - 1]) * dt
    return pos_x, pos_y, corrected_vel_x, corrected_vel_y


def compute_drift_pct(est_pos_x, est_pos_y, true_pos_x, true_pos_y):
    """Drift % = (final position error / total distance traveled) * 100."""
    final_err = np.sqrt(
        (est_pos_x[-1] - true_pos_x[-1])**2 +
        (est_pos_y[-1] - true_pos_y[-1])**2
    )
    dx = np.diff(true_pos_x)
    dy = np.diff(true_pos_y)
    total_dist = np.sum(np.sqrt(dx**2 + dy**2))
    if total_dist < 1.0:
        return float("inf")
    return (final_err / total_dist) * 100.0


def evaluate_session(session, corrector):
    """Evaluate one session with all three methods. Returns metrics dict or None."""
    try:
        raw_imu = load_s_file(session["s_path"])
        raw_gt = load_v_file(session["v_path"])
    except Exception as e:
        print(f"    [ERROR loading] {e}")
        return None

    # Ground truth
    gt_t = raw_gt["time"].to_numpy(dtype=float)
    gt_t = gt_t - gt_t[0]
    gt_lat = raw_gt["lat"].to_numpy(dtype=float)
    gt_lon = raw_gt["lon"].to_numpy(dtype=float)
    gt_speed = raw_gt["speed"].to_numpy(dtype=float)
    gt_heading = raw_gt["heading"].to_numpy(dtype=float)

    gt_vel_e, gt_vel_n = speed_heading_to_velocity(gt_speed, gt_heading)
    gt_pos_e, gt_pos_n = latlon_to_enu(gt_lat, gt_lon, gt_lat[0], gt_lon[0])

    # IMU times
    imu_t = raw_imu["time"].to_numpy(dtype=float)
    if imu_t.max() > 1e5:
        imu_t = imu_t / 1000.0
    imu_t = imu_t - imu_t[0]
    n = len(imu_t)

    true_pos_x = np.interp(imu_t, gt_t, gt_pos_e)
    true_pos_y = np.interp(imu_t, gt_t, gt_pos_n)
    true_vel_x = np.interp(imu_t, gt_t, gt_vel_e)
    true_vel_y = np.interp(imu_t, gt_t, gt_vel_n)

    # Smartphone GPS for EKF
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

    # INS mechanization without ground-truth re-anchoring
    import hashlib
    session_seed = int(hashlib.md5(session["name"].encode()).hexdigest(), 16) % 1000000
    ins_df = run_ins_mechanization(raw_imu, ref_df=None, seed=session_seed)
    
    t = ins_df["_t_sec"].to_numpy(dtype=float)
    duration = t[-1]

    # --- Method 1: Raw INS ---
    ins_pos_x = ins_df["ins_pos_x"].to_numpy()
    ins_pos_y = ins_df["ins_pos_y"].to_numpy()
    ins_drift = compute_drift_pct(ins_pos_x, ins_pos_y, true_pos_x, true_pos_y)

    # --- Method 2: AI-corrected INS (no EKF, no GNSS) ---
    ai_corrections = run_ai_correction_batch(ins_df, corrector)
    ai_drift = float("inf")
    if ai_corrections is not None:
        result = compute_ai_corrected_trajectory(ins_df, ai_corrections)
        if result is not None:
            ai_pos_x, ai_pos_y, _, _ = result
            ai_drift = compute_drift_pct(ai_pos_x, ai_pos_y, true_pos_x, true_pos_y)

    # --- Method 3: Traditional Continuous EKF (INS + AI + GNSS) ---
    ekf_result = run_ekf_fusion(
        ins_df, ai_corrections=ai_corrections,
        gnss_available=gps_available,
        gnss_pos=gps_pos,
        gnss_accuracy=gps_accuracy,
        use_seamless_switching=False,
    )
    ekf_drift = compute_drift_pct(
        ekf_result["pos_x"], ekf_result["pos_y"],
        true_pos_x, true_pos_y)

    # --- Method 4: Seamless Adaptive EKF (GNSS ⇄ AI-IDR State Machine + Battery Gating) ---
    seamless_result = run_ekf_fusion(
        ins_df, ai_corrections=ai_corrections,
        gnss_available=gps_available,
        gnss_pos=gps_pos,
        gnss_accuracy=gps_accuracy,
        use_seamless_switching=True,
    )
    seamless_drift = compute_drift_pct(
        seamless_result["pos_x"], seamless_result["pos_y"],
        true_pos_x, true_pos_y)

    bat_summary = seamless_result.get("battery_summary", {})
    battery_savings_pct = bat_summary.get("battery_savings_pct", 0.0)

    return {
        "session": session["name"],
        "category": session["category"],
        "duration": duration,
        "ins_drift": ins_drift,
        "ai_drift": ai_drift,
        "ekf_drift": ekf_drift,
        "seamless_drift": seamless_drift,
        "battery_savings": battery_savings_pct,
    }


def main():
    sessions = discover_all_sessions()
    if not sessions:
        print("No sessions found. Check config.DATASET_ROOT.")
        return

    print(f"{'='*85}")
    print(f"  EVALUATION WITH SEAMLESS ADAPTIVE GNSS <-> AI SWITCHING: {len(sessions)} sessions")
    print(f"{'='*85}\n")

    corrector = load_ai_corrector()
    if corrector is None:
        print("[WARNING] AI model not loaded. AI-corrected column will show inf.\n")

    results = []
    for i, session in enumerate(sessions):
        print(f"[{i+1}/{len(sessions)}] {session['name']} ({session['category']})...", end=" ")
        try:
            r = evaluate_session(session, corrector)
            if r is not None:
                results.append(r)
                print(f"INS={r['ins_drift']:.1f}%  AI={r['ai_drift']:.1f}%  "
                      f"EKF={r['ekf_drift']:.1f}%  Seamless={r['seamless_drift']:.1f}%  "
                      f"BatSaved={r['battery_savings']:.1f}%", flush=True)
            else:
                print("skipped", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    if not results:
        print("\nNo sessions evaluated successfully.")
        return

    # --- Summary table ---
    print(f"\n{'='*85}")
    print(f"  FOUR-WAY NAVIGATION & BATTERY COMPARISON ({len(results)} sessions)")
    print(f"{'='*85}")
    print(f"  {'Session':<12} {'Duration':>8} {'INS Drift%':>11} {'AI Drift%':>11} {'EKF Drift%':>11} {'Seamless%':>11} {'BatSaved%':>10}")
    print(f"  {'-'*79}")

    for r in results:
        ai_str = f"{r['ai_drift']:>11.2f}" if r['ai_drift'] < 1e6 else f"{'N/A':>11}"
        print(f"  {r['session']:<12} {r['duration']:>7.0f}s {r['ins_drift']:>11.2f} "
              f"{ai_str} {r['ekf_drift']:>11.2f} {r['seamless_drift']:>11.2f} {r['battery_savings']:>9.1f}%")
    print(f"{'='*85}")

    ins_drifts = [r["ins_drift"] for r in results if r["ins_drift"] < 1e6 and not np.isnan(r["ins_drift"])]
    ai_drifts = [r["ai_drift"] for r in results if r["ai_drift"] < 1e6 and not np.isnan(r["ai_drift"])]
    ekf_drifts = [r["ekf_drift"] for r in results if r["ekf_drift"] < 1e6 and not np.isnan(r["ekf_drift"])]
    seamless_drifts = [r["seamless_drift"] for r in results if r["seamless_drift"] < 1e6 and not np.isnan(r["seamless_drift"])]
    battery_savings = [r["battery_savings"] for r in results]

    print(f"  {'MEAN':<12} {'':>8} ", end="")
    print(f"{np.mean(ins_drifts):>11.2f} " if ins_drifts else f"{'N/A':>11} ", end="")
    print(f"{np.mean(ai_drifts):>11.2f} " if ai_drifts else f"{'N/A':>11} ", end="")
    print(f"{np.mean(ekf_drifts):>11.2f} " if ekf_drifts else f"{'N/A':>11} ", end="")
    print(f"{np.mean(seamless_drifts):>11.2f} " if seamless_drifts else f"{'N/A':>11} ", end="")
    print(f"{np.mean(battery_savings):>9.1f}%" if battery_savings else f"{'N/A':>9}")

    print(f"  {'MEDIAN':<12} {'':>8} ", end="")
    print(f"{np.median(ins_drifts):>11.2f} " if ins_drifts else f"{'N/A':>11} ", end="")
    print(f"{np.median(ai_drifts):>11.2f} " if ai_drifts else f"{'N/A':>11} ", end="")
    print(f"{np.median(ekf_drifts):>11.2f} " if ekf_drifts else f"{'N/A':>11} ", end="")
    print(f"{np.median(seamless_drifts):>11.2f} " if seamless_drifts else f"{'N/A':>11} ", end="")
    print(f"{np.median(battery_savings):>9.1f}%" if battery_savings else f"{'N/A':>9}")

    print(f"\n{'='*85}")

    # Show improvement
    if seamless_drifts and ins_drifts:
        seamless_imp = (1 - np.mean(seamless_drifts) / np.mean(ins_drifts)) * 100
        print(f"  Seamless Adaptive EKF reduces drift by: {seamless_imp:.1f}% vs raw INS (Mean)")
    if battery_savings:
        print(f"  Average AI Battery Energy Saved:         {np.mean(battery_savings):.1f}% (AI Duty Cycle Off)")

    
    # Explicitly report failed/excluded sessions count
    failed_sessions = [r["session"] for r in results if np.isnan(r["ekf_drift"]) or np.isinf(r["ekf_drift"]) or r["ekf_drift"] > 1e6]
    excluded_count = len(sessions) - len(results)
    
    print(f"\n  Evaluation Status Summary:")
    print(f"    Total sessions discovered: {len(sessions)}")
    print(f"    Successfully evaluated:   {len(results)}")
    print(f"    Excluded from loading:    {excluded_count}")
    print(f"    Failed during processing:  {len(failed_sessions)} {failed_sessions if failed_sessions else ''}")
    print(f"{'='*70}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    out_path = os.path.join(config.PLOT_DIR, "evaluation_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
