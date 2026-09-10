"""
tune_ekf.py
===========
Automated grid search over EKF process noise (Q_vel) and AI measurement noise (R_ai_base).
Caches AI inference predictions ONCE to evaluate grid combinations in milliseconds.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.io_utils import load_s_file, load_v_file
from src.ins_mechanization import run_ins_mechanization
from src.pipeline import load_ai_corrector, run_ai_correction_batch
from src.generate_labels import derive_ground_truth, align_ground_truth
from src.ekf import run_ekf_fusion


def grid_search_ekf(session_name: str = "S1"):
    print(f"\n==================================================")
    print(f"  EKF COVARIANCE GRID SEARCH (Session: {session_name})")
    print(f"==================================================\n")

    s_file = os.path.join(config.DATASET_ROOT, "S (Driver A)", session_name, f"S-{session_name}.csv")
    v_file = os.path.join(config.DATASET_ROOT, "S (Driver A)", session_name, f"V-{session_name}.csv")

    if not os.path.exists(s_file) or not os.path.exists(v_file):
        # Fallback search in data/processed or discover_sessions
        from src.discover_sessions import discover_all_sessions
        sessions = discover_all_sessions()
        target = [s for s in sessions if s["name"] == session_name]
        if not target:
            raise FileNotFoundError(f"Session {session_name} not found.")
        s_file = target[0]["s_path"]
        v_file = target[0]["v_path"]

    raw_imu = load_s_file(s_file)
    raw_gt = load_v_file(v_file)

    print("[1/3] Running INS mechanization...")
    ins_df = run_ins_mechanization(raw_imu)
    gt_derived = derive_ground_truth(raw_gt)
    gt_aligned = align_ground_truth(ins_df, gt_derived)

    true_pos_x = gt_aligned["true_pos_x"].to_numpy()
    true_pos_y = gt_aligned["true_pos_y"].to_numpy()
    total_distance = np.sum(np.sqrt(np.diff(true_pos_x)**2 + np.diff(true_pos_y)**2))

    print("[2/3] Running AI inference ONCE and caching predictions...")
    corrector = load_ai_corrector()
    ai_corrections, ai_stds = run_ai_correction_batch(ins_df, corrector)

    # Convert GT position to array for EKF evaluation
    gps_available = np.ones(len(ins_df), dtype=bool)
    gps_pos = np.column_stack([true_pos_x, true_pos_y])
    gps_accuracy = np.full(len(ins_df), 5.0)

    # Grid parameters
    q_vel_values = [0.1, 0.25, 0.5, 1.0, 2.0]
    r_ai_values = [0.1, 0.25, 0.5, 1.0, 2.0]

    results = []

    print("[3/3] Executing fast EKF grid search...\n")
    orig_q_vel = config.EKF_Q_VEL_STD
    orig_r_ai = config.EKF_R_AI_VEL_STD

    best_rmse = float("inf")
    best_config = None

    for q_vel in q_vel_values:
        for r_ai in r_ai_values:
            config.EKF_Q_VEL_STD = q_vel
            config.EKF_R_AI_VEL_STD = r_ai

            ekf_res = run_ekf_fusion(
                ins_df, ai_corrections=ai_corrections, ai_stds=ai_stds,
                gnss_available=gps_available, gnss_pos=gps_pos, gnss_accuracy=gps_accuracy
            )

            pos_x = ekf_res["pos_x"]
            pos_y = ekf_res["pos_y"]

            # Compute Position RMSE & Final Error
            err_x = pos_x - true_pos_x
            err_y = pos_y - true_pos_y
            pos_rmse = np.sqrt(np.mean(err_x**2 + err_y**2))
            final_err = np.sqrt((pos_x[-1] - true_pos_x[-1])**2 + (pos_y[-1] - true_pos_y[-1])**2)
            drift_pct = (final_err / total_distance) * 100.0 if total_distance > 0 else 0.0

            results.append({
                "Q_vel (m/s)": q_vel,
                "R_ai (m/s)": r_ai,
                "Position RMSE (m)": pos_rmse,
                "Final Error (m)": final_err,
                "Drift %": drift_pct,
            })

            if pos_rmse < best_rmse:
                best_rmse = pos_rmse
                best_config = (q_vel, r_ai, pos_rmse, final_err, drift_pct)

    # Restore originals
    config.EKF_Q_VEL_STD = orig_q_vel
    config.EKF_R_AI_VEL_STD = orig_r_ai

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"OPTIMAL EKF COVARIANCE CONFIGURATION:")
    print(f"  Best Q_vel_std: {best_config[0]:.2f} m/s")
    print(f"  Best R_ai_std:  {best_config[1]:.2f} m/s")
    print(f"  Position RMSE:  {best_config[2]:.3f} m")
    print(f"  Final Error:    {best_config[3]:.3f} m ({best_config[4]:.4f}% drift)")
    print(f"{'='*60}\n")

    return res_df, best_config


if __name__ == "__main__":
    grid_search_ekf("S1")
