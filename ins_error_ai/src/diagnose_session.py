"""
diagnose_session.py
===================
Diagnostic script to inspect the raw data and INS outputs for physical plausibility.
Prints 21 detailed metrics and generates 8 diagnostic plots.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.discover_sessions import discover_all_sessions
from src.io_utils import load_s_file, load_v_file, latlon_to_enu, speed_heading_to_velocity
from src.ins_mechanization import run_ins_mechanization


def analyze_session(session_name):
    # 1. Discover sessions
    sessions = discover_all_sessions()
    session = next((s for s in sessions if s["name"] == session_name), None)
    if not session:
        print(f"Session '{session_name}' not found. Available sessions:")
        for s in sessions[:15]:
            print(f"  {s['name']}")
        return

    print(f"Analyzing session: {session_name}")
    print(f"S-file: {session['s_path']}")
    print(f"V-file: {session['v_path']}")

    # Load S and V files
    s_df = load_s_file(session["s_path"])
    v_df = load_v_file(session["v_path"])

    n_s = len(s_df)
    n_v = len(v_df)

    # 3. Timestamps & Duration
    # S-file uses TIME SINCE START (ms)
    t_s_raw = s_df["time"].to_numpy()
    t_s = t_s_raw / 1000.0 if t_s_raw.max() > 1e5 else t_s_raw
    duration_s = t_s[-1] - t_s[0]

    # Timestep statistics for S-file
    dt_s = np.diff(t_s)
    median_dt = np.median(dt_s)
    min_dt = np.min(dt_s)
    max_dt = np.max(dt_s)
    est_freq = 1.0 / median_dt if median_dt > 0 else 0.0

    # Missing & duplicate timestamps in S-file
    duplicates = len(t_s) - len(np.unique(t_s))
    
    # Check for gaps (e.g. dt > 2 * median_dt)
    large_gaps = np.sum(dt_s > 2.0 * median_dt)

    # 8. NaN Counts
    nan_counts = {}
    for col in s_df.columns:
        nan_counts[f"S_{col}"] = s_df[col].isna().sum()
    for col in v_df.columns:
        nan_counts[f"V_{col}"] = v_df[col].isna().sum()

    # 9-11. Sensor statistics
    def get_stats(df, col_x, col_y, col_z):
        if all(c in df.columns for c in [col_x, col_y, col_z]):
            mag = np.linalg.norm(df[[col_x, col_y, col_z]].to_numpy(), axis=1)
            return np.min(mag), np.max(mag), np.mean(mag), np.std(mag)
        return 0.0, 0.0, 0.0, 0.0

    acc_stats = get_stats(s_df, "acc_x", "acc_y", "acc_z")
    gyro_stats = get_stats(s_df, "gyro_yaw", "gyro_pitch", "gyro_roll")
    grav_stats = get_stats(s_df, "grav_x", "grav_y", "grav_z")

    # 12-15. V-file ranges
    lat = v_df["lat"].to_numpy()
    lon = v_df["lon"].to_numpy()
    speed_kmh = v_df["speed"].to_numpy()
    heading = v_df["heading"].to_numpy()

    lat_range = (np.min(lat), np.max(lat))
    lon_range = (np.min(lon), np.max(lon))
    speed_range = (np.min(speed_kmh), np.max(speed_kmh))
    heading_range = (np.min(heading), np.max(heading))

    # 16-17. Derived velocity & position (Ground Truth)
    vel_east, vel_north = speed_heading_to_velocity(speed_kmh, heading)
    pos_east, pos_north = latlon_to_enu(lat, lon, lat[0], lon[0])
    
    derived_vel_range_x = (np.min(vel_east), np.max(vel_east))
    derived_vel_range_y = (np.min(vel_north), np.max(vel_north))
    derived_pos_range_x = (np.min(pos_east), np.max(pos_east))
    derived_pos_range_y = (np.min(pos_north), np.max(pos_north))

    # 18-19. Run INS Mechanization
    ins_df = run_ins_mechanization(s_df)
    ins_vel_x = ins_df["ins_vel_x"].to_numpy()
    ins_vel_y = ins_df["ins_vel_y"].to_numpy()
    ins_pos_x = ins_df["ins_pos_x"].to_numpy()
    ins_pos_y = ins_df["ins_pos_y"].to_numpy()

    ins_vel_range_x = (np.min(ins_vel_x), np.max(ins_vel_x))
    ins_vel_range_y = (np.min(ins_vel_y), np.max(ins_vel_y))
    ins_pos_range_x = (np.min(ins_pos_x), np.max(ins_pos_x))
    ins_pos_range_y = (np.min(ins_pos_y), np.max(ins_pos_y))

    # 20-21. Displacements
    final_ins_disp = np.sqrt(ins_pos_x[-1]**2 + ins_pos_y[-1]**2)
    final_gt_disp = np.sqrt(pos_east[-1]**2 + pos_north[-1]**2)

    # ------------------ PRINT METRICS ------------------
    print("\n" + "="*50)
    print("DIAGNOSTIC METRICS")
    print("="*50)
    print(f"1. Number of S-file samples: {n_s}")
    print(f"2. Number of V-file samples: {n_v}")
    print(f"3. Duration: {duration_s:.3f} s")
    print(f"4. Actual median/min/max timestep (S-file): {median_dt:.3f} / {min_dt:.3f} / {max_dt:.3f} s")
    print(f"5. Estimated sampling frequency: {est_freq:.2f} Hz")
    print(f"6. Number of missing timestamps (large gaps > 2*dt): {large_gaps}")
    print(f"7. Number of duplicate timestamps: {duplicates}")
    print("8. NaN counts for important columns:")
    for col, count in nan_counts.items():
        if count > 0 or "acc" in col or "gyro" in col or "lat" in col or "lon" in col or "speed" in col:
            print(f"   {col}: {count}")
    print(f"9. Accelerometer magnitude min/max/mean/std: {acc_stats[0]:.3f} / {acc_stats[1]:.3f} / {acc_stats[2]:.3f} / {acc_stats[3]:.3f} m/s²")
    print(f"10. Gyroscope magnitude min/max/mean/std: {gyro_stats[0]:.3f} / {gyro_stats[1]:.3f} / {gyro_stats[2]:.3f} / {gyro_stats[3]:.3f} rad/s")
    print(f"11. Gravity magnitude min/max/mean/std: {grav_stats[0]:.3f} / {grav_stats[1]:.3f} / {grav_stats[2]:.3f} / {grav_stats[3]:.3f} m/s²")
    print(f"12. Latitude range: {lat_range[0]:.6f} to {lat_range[1]:.6f} (span: {lat_range[1]-lat_range[0]:.6f} deg)")
    print(f"13. Longitude range: {lon_range[0]:.6f} to {lon_range[1]:.6f} (span: {lon_range[1]-lon_range[0]:.6f} deg)")
    print(f"14. Speed range: {speed_range[0]:.3f} to {speed_range[1]:.3f} km/h")
    print(f"15. Heading range: {heading_range[0]:.3f} to {heading_range[1]:.3f} deg")
    print(f"16. Derived velocity range: X [{derived_vel_range_x[0]:.3f}, {derived_vel_range_x[1]:.3f}] | Y [{derived_vel_range_y[0]:.3f}, {derived_vel_range_y[1]:.3f}] m/s")
    print(f"17. Derived position range: X [{derived_pos_range_x[0]:.1f}, {derived_pos_range_x[1]:.1f}] | Y [{derived_pos_range_y[0]:.1f}, {derived_pos_range_y[1]:.1f}] m")
    print(f"18. INS velocity range: X [{ins_vel_range_x[0]:.3f}, {ins_vel_range_x[1]:.3f}] | Y [{ins_vel_range_y[0]:.3f}, {ins_vel_range_y[1]:.3f}] m/s")
    print(f"19. INS position range: X [{ins_pos_range_x[0]:.1f}, {ins_pos_range_x[1]:.1f}] | Y [{ins_pos_range_y[0]:.1f}, {ins_pos_range_y[1]:.1f}] m")
    print(f"20. Final INS displacement: {final_ins_disp:.2f} m")
    print(f"21. GNSS/reference displacement: {final_gt_disp:.2f} m")
    print("="*50 + "\n")

    # ------------------ PLOTS ------------------
    plot_dir = os.path.join("results", "diagnostics", session_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to {plot_dir}...")

    # 01_gnss_trajectory.png
    plt.figure()
    plt.plot(pos_east, pos_north, 'k-', label='GNSS Ground Truth')
    plt.plot(pos_east[0], pos_north[0], 'go', label='Start')
    plt.plot(pos_east[-1], pos_north[-1], 'ro', label='End')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title(f'01 GNSS Trajectory - Session {session_name}')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "01_gnss_trajectory.png"))
    plt.close()

    # 02_ins_trajectory.png
    plt.figure()
    plt.plot(ins_pos_x, ins_pos_y, 'r-', label='Pure INS Trajectory')
    plt.plot(ins_pos_x[0], ins_pos_y[0], 'go', label='Start')
    plt.plot(ins_pos_x[-1], ins_pos_y[-1], 'ro', label='End')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title(f'02 INS Trajectory - Session {session_name}')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "02_ins_trajectory.png"))
    plt.close()

    # 03_gnss_vs_ins.png
    plt.figure()
    plt.plot(pos_east, pos_north, 'k-', label='GNSS Ground Truth')
    plt.plot(ins_pos_x, ins_pos_y, 'r--', label='Pure INS')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title(f'03 GNSS vs INS Trajectory - Session {session_name}')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "03_gnss_vs_ins.png"))
    plt.close()

    # 04_acceleration.png
    t_axis = ins_df["_t_sec"].to_numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, s_df["acc_x"], label='Acc X')
    plt.plot(t_axis, s_df["acc_y"], label='Acc Y')
    plt.plot(t_axis, s_df["acc_z"], label='Acc Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f'04 Acceleration - Session {session_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "04_acceleration.png"))
    plt.close()

    # 05_gyro.png
    plt.figure(figsize=(10, 6))
    plt.plot(t_axis, s_df["gyro_yaw"], label='Gyro Yaw (Z)')
    plt.plot(t_axis, s_df["gyro_pitch"], label='Gyro Pitch (Y)')
    plt.plot(t_axis, s_df["gyro_roll"], label='Gyro Roll (X)')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title(f'05 Gyroscope - Session {session_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "05_gyro.png"))
    plt.close()

    # 06_gravity.png
    plt.figure(figsize=(10, 6))
    if "grav_x" in s_df.columns:
        plt.plot(t_axis, s_df["grav_x"], label='Gravity X')
        plt.plot(t_axis, s_df["grav_y"], label='Gravity Y')
        plt.plot(t_axis, s_df["grav_z"], label='Gravity Z')
    else:
        plt.text(0.5, 0.5, 'Gravity columns missing', ha='center', va='center')
    plt.xlabel('Time (s)')
    plt.ylabel('Gravity (m/s²)')
    plt.title(f'06 Gravity - Session {session_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "06_gravity.png"))
    plt.close()

    # 07_speed.png
    # Get ground truth speed aligned with time
    gt_t_raw = v_df["time"].to_numpy()
    gt_t = gt_t_raw - gt_t_raw[0]
    plt.figure(figsize=(10, 6))
    plt.plot(gt_t, speed_kmh, 'k-', label='Ground Truth Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (km/h)')
    plt.title(f'07 Speed Profile - Session {session_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "07_speed.png"))
    plt.close()

    # 08_heading.png
    plt.figure(figsize=(10, 6))
    plt.plot(gt_t, heading, 'k-', label='Ground Truth Heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading (degrees)')
    plt.title(f'08 Heading - Session {session_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "08_heading.png"))
    plt.close()

    print("Diagnostics completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="S1", help="Session to diagnose")
    args = parser.parse_args()
    analyze_session(args.session)
