"""
create_baseline.py
==================
Generates the classical baseline metrics and trajectory plot for the pure INS.
Saves outputs to results/baseline_metrics.csv and results/baseline_trajectory.png.
"""

import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.io_utils import load_s_file, load_v_file, latlon_to_enu, speed_heading_to_velocity
from src.ins_mechanization import run_ins_mechanization
from src.discover_sessions import discover_all_sessions


def main():
    session_name = "S1"
    sessions = discover_all_sessions()
    session = next((s for s in sessions if s["name"] == session_name), None)
    if not session:
        print(f"Session '{session_name}' not found.")
        return

    print(f"Generating baseline for session: {session_name}")

    # Load S and V files
    s_df = load_s_file(session["s_path"])
    v_df = load_v_file(session["v_path"])

    # Run INS Mechanization
    ins_df = run_ins_mechanization(s_df)
    t_s = ins_df["_t_sec"].to_numpy()

    # Ground Truth from V-file
    t_v = v_df["time"].to_numpy()
    t_v = t_v - t_v[0]
    lat = v_df["lat"].to_numpy()
    lon = v_df["lon"].to_numpy()
    speed = v_df["speed"].to_numpy()
    heading = v_df["heading"].to_numpy()

    gt_pos_x, gt_pos_y = latlon_to_enu(lat, lon, lat[0], lon[0])
    gt_vel_x, gt_vel_y = speed_heading_to_velocity(speed, heading)

    # Interpolate ground truth to S-file time
    true_pos_x = np.interp(t_s, t_v, gt_pos_x)
    true_pos_y = np.interp(t_s, t_v, gt_pos_y)
    true_vel_x = np.interp(t_s, t_v, gt_vel_x)
    true_vel_y = np.interp(t_s, t_v, gt_vel_y)

    # INS pos/vel
    ins_pos_x = ins_df["ins_pos_x"].to_numpy()
    ins_pos_y = ins_df["ins_pos_y"].to_numpy()
    ins_vel_x = ins_df["ins_vel_x"].to_numpy()
    ins_vel_y = ins_df["ins_vel_y"].to_numpy()

    # Calculate metrics
    pos_err = np.sqrt((ins_pos_x - true_pos_x)**2 + (ins_pos_y - true_pos_y)**2)
    vel_err = np.sqrt((ins_vel_x - true_vel_x)**2 + (ins_vel_y - true_vel_y)**2)

    pos_rmse = np.sqrt(np.mean(pos_err**2))
    pos_mae = np.mean(pos_err)
    final_pos_err = pos_err[-1]

    vel_rmse = np.sqrt(np.mean(vel_err**2))
    vel_mae = np.mean(vel_err)

    print("\nClassical Baseline Metrics:")
    print(f"  Position RMSE:      {pos_rmse:.2f} m")
    print(f"  Position MAE:       {pos_mae:.2f} m")
    print(f"  Final Position Err: {final_pos_err:.2f} m")
    print(f"  Velocity RMSE:      {vel_rmse:.3f} m/s")
    print(f"  Velocity MAE:       {vel_mae:.3f} m/s")

    # Save to results/baseline_metrics.csv
    os.makedirs("results", exist_ok=True)
    metrics_df = pd.DataFrame({
        "Metric": ["Position RMSE (m)", "Position MAE (m)", "Final Position Error (m)", "Velocity RMSE (m/s)", "Velocity MAE (m/s)"],
        "Value": [pos_rmse, pos_mae, final_pos_err, vel_rmse, vel_mae]
    })
    metrics_path = os.path.join("results", "baseline_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    # Plot results/baseline_trajectory.png
    plt.figure(figsize=(10, 8))
    plt.plot(true_pos_x, true_pos_y, 'k-', linewidth=2, label="GNSS Reference")
    plt.plot(ins_pos_x, ins_pos_y, 'r--', linewidth=1.5, label="Pure INS (Drifting)")
    plt.plot(true_pos_x[0], true_pos_y[0], 'go', markersize=8, label="Start")
    plt.plot(true_pos_x[-1], true_pos_y[-1], 'ro', markersize=8, label="GNSS End")
    plt.plot(ins_pos_x[-1], ins_pos_y[-1], 'rx', markersize=8, label="INS End")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title(f"Baseline Trajectory Comparison - Session {session_name}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    
    trajectory_path = os.path.join("results", "baseline_trajectory.png")
    plt.savefig(trajectory_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {trajectory_path}")
    plt.close()


if __name__ == "__main__":
    main()
