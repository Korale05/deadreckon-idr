"""
ins_mechanization.py
=====================
STEP 1 of the plan: the classical physics INS.

This is plain dead-reckoning -- no AI here at all. It:
  1. Tracks orientation over time using the Euler angles (yaw/pitch/roll)
     provided by the smartphone's sensor fusion, OR by integrating the
     gyroscope if Euler angles are unavailable.
  2. Uses that orientation to rotate raw accelerometer readings from the
     "body frame" (strapped to the phone) into the "navigation frame"
     (a fixed east/north/up frame).
  3. Subtracts gravity — using the phone's own gravity sensor if available,
     or a constant [0, 0, g] vector otherwise.
  4. Integrates acceleration -> velocity -> position (trapezoidal rule).

This module produces the INS's own (flawed, drifting) estimate at every
timestep. That estimate is later compared against ground truth to generate
the error labels that the AI model learns to predict.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------
def euler_to_rotmat(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from Euler angles (in degrees).
    Convention: ZYX intrinsic (yaw around Z, pitch around Y, roll around X).
    This rotates a vector from body frame to navigation frame.
    """
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)

    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)

    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr  ],
    ])
    return R


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else q


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_from_gyro_delta(gyro_xyz, dt):
    """Small-angle quaternion representing rotation over dt, from gyro rad/s."""
    angle = np.linalg.norm(gyro_xyz) * dt
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = gyro_xyz / (np.linalg.norm(gyro_xyz) + 1e-12)
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


# ---------------------------------------------------------------------------
# Main mechanization routine
# ---------------------------------------------------------------------------
def run_ins_mechanization(imu_df: pd.DataFrame, ref_df: pd.DataFrame = None,
                          gnss_available: np.ndarray = None, seed: int = None) -> pd.DataFrame:
    """
    Run classical strapdown INS mechanization over one session.
    Uses a physically correct 2D vehicle INS model:
      1. Calibrates accelerometer bias in the body frame when stationary.
      2. Removes gravity from the vertical Z-axis of the phone.
      3. Rotates horizontal accelerations to the navigation frame using the
         phone's orientation Yaw (azimuth) converted to standard angle.
      4. Integrates twice: acceleration -> velocity -> position.
      
    If ref_df is provided, the INS state is periodically reset/re-anchored
    back to the reference values at randomized intervals (10-120 seconds)
    when GNSS is available.
    """
    n = len(imu_df)

    # --- Time ---
    t = imu_df["time"].to_numpy(dtype=float)
    if t.max() > 1e5:
        t = t / 1000.0
    t = t - t[0]

    # --- Raw accelerometer ---
    acc = np.column_stack([
        imu_df["acc_x"].to_numpy(dtype=float),
        imu_df["acc_y"].to_numpy(dtype=float),
        imu_df["acc_z"].to_numpy(dtype=float),
    ])

    # --- Calibrate Accelerometer Bias in Body Frame (stationary at start, first 100 samples)
    n_stationary = min(100, n)
    bias_x = np.mean(acc[:n_stationary, 0])
    bias_y = np.mean(acc[:n_stationary, 1])

    # Subtract bias and gravity in the body frame
    lin_acc_body_x = acc[:, 0] - bias_x
    lin_acc_body_y = acc[:, 1] - bias_y
    lin_acc_body_z = acc[:, 2] - config.GRAVITY

    # --- Orientation Yaw (Azimuth) ---
    have_euler = "orient_yaw" in imu_df.columns
    orient_yaw = imu_df["orient_yaw"].to_numpy(dtype=float) if have_euler else np.zeros(n)

    # --- Rotate to navigation frame (East, North, Up) ---
    nav_acc = np.zeros((n, 3))
    for i in range(n):
        if have_euler:
            # Conversion from Android azimuth (0=N, 90=E clockwise) to standard math angle
            # East = acc_y * sin(azimuth) + acc_x * cos(azimuth)
            # North = acc_y * cos(azimuth) - acc_x * sin(azimuth)
            yaw_rad = np.radians(orient_yaw[i])
            nav_acc[i, 0] = lin_acc_body_y[i] * np.sin(yaw_rad) + lin_acc_body_x[i] * np.cos(yaw_rad)
            nav_acc[i, 1] = lin_acc_body_y[i] * np.cos(yaw_rad) - lin_acc_body_x[i] * np.sin(yaw_rad)
            nav_acc[i, 2] = lin_acc_body_z[i]
        else:
            # Fallback (no orientation)
            nav_acc[i, 0] = lin_acc_body_x[i]
            nav_acc[i, 1] = lin_acc_body_y[i]
            nav_acc[i, 2] = lin_acc_body_z[i]

    # --- Trapezoidal integration: acceleration -> velocity -> position ---
    vel = np.zeros((n, 3))
    pos = np.zeros((n, 3))
    is_reset_arr = np.zeros(n, dtype=bool)

    if ref_df is not None:
        # Initialize INS starting state to the ground truth reference values
        vel[0, 0] = ref_df["true_vel_x"].iloc[0]
        vel[0, 1] = ref_df["true_vel_y"].iloc[0]
        pos[0, 0] = ref_df["true_pos_x"].iloc[0]
        pos[0, 1] = ref_df["true_pos_y"].iloc[0]

    # Deterministic randomness per session if seed is provided
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng(42)
    last_reset_time = t[0]
    # Randomized interval between 10 and 120 seconds matching blackout durations
    current_interval = rng.uniform(10.0, 120.0)

    for i in range(1, n):
        dt = max(t[i] - t[i-1], 1e-6)

        # Check for periodic re-anchoring back to ground truth / EKF state
        if ref_df is not None and (t[i] - last_reset_time) > current_interval:
            gps_ok = True
            if gnss_available is not None:
                gps_ok = bool(gnss_available[i-1])

            if gps_ok:
                # Reset INS to the true reference velocity & position
                vel[i-1, 0] = ref_df["true_vel_x"].iloc[i-1]
                vel[i-1, 1] = ref_df["true_vel_y"].iloc[i-1]
                pos[i-1, 0] = ref_df["true_pos_x"].iloc[i-1]
                pos[i-1, 1] = ref_df["true_pos_y"].iloc[i-1]
                is_reset_arr[i-1] = True
                
                last_reset_time = t[i-1]
                current_interval = rng.uniform(10.0, 120.0)

        vel[i] = vel[i-1] + 0.5 * (nav_acc[i] + nav_acc[i-1]) * dt

        # --- Zero-Velocity Update (ZUPT) & Physical Motion Constraints ---
        # Detect stationary state: low angular velocity (<0.08 rad/s) and gravity-only accel (|acc| ~ 9.81 m/s²)
        if "gyro_yaw" in imu_df.columns:
            gyro_mag = np.sqrt(
                imu_df["gyro_yaw"].iloc[i]**2 +
                imu_df["gyro_pitch"].iloc[i]**2 +
                imu_df["gyro_roll"].iloc[i]**2
            )
            acc_mag = np.sqrt(
                acc[i, 0]**2 + acc[i, 1]**2 + acc[i, 2]**2
            )
            if gyro_mag < 0.08 and abs(acc_mag - config.GRAVITY) < 0.35:
                vel[i, 0] *= 0.85
                vel[i, 1] *= 0.85
                vel[i, 2] *= 0.85

        # Physical speed limit bounding for land vehicles (+-45 m/s = 162 km/h max)
        vel[i, 0] = np.clip(vel[i, 0], -45.0, 45.0)
        vel[i, 1] = np.clip(vel[i, 1], -45.0, 45.0)

        pos[i] = pos[i-1] + 0.5 * (vel[i] + vel[i-1]) * dt

    # --- Build output DataFrame ---
    out = imu_df.copy()
    out["ins_vel_x"] = vel[:, 0]
    out["ins_vel_y"] = vel[:, 1]
    out["ins_vel_z"] = vel[:, 2]
    out["ins_pos_x"] = pos[:, 0]
    out["ins_pos_y"] = pos[:, 1]
    out["ins_pos_z"] = pos[:, 2]
    out["ins_is_reset"] = is_reset_arr
    out["_t_sec"] = t
    return out


# ---------------------------------------------------------------------------
# Streaming (stateful) INS tracker — for real-time, sample-by-sample use
# ---------------------------------------------------------------------------
class INSTracker:
    """
    Stateful INS tracker that processes one IMU sample at a time.

    Shares the same rotation and trapezoidal integration math as
    run_ins_mechanization() but can be called incrementally, carrying
    internal state between calls.  This is what a real-time bridge needs.

    Usage:
        tracker = INSTracker()
        for each sample:
            vel_x, vel_y, pos_x, pos_y = tracker.step(acc_x, acc_y, acc_z, yaw_deg, dt)
    """

    def __init__(self):
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0
        # Previous-step nav-frame accelerations (for trapezoidal rule)
        self._prev_nav_acc_x = 0.0
        self._prev_nav_acc_y = 0.0
        # Bias calibration accumulator
        self._bias_x = 0.0
        self._bias_y = 0.0
        self._calibrated = False
        self._cal_samples = []
        self._N_CAL = 100  # number of stationary samples for bias estimation

    def _rotate_body_to_nav(self, acc_x_body, acc_y_body, yaw_deg):
        """
        Rotate horizontal body-frame accelerations to ENU navigation frame.

        Uses the exact formula from spec §1.4:
            East (x)  = acc_y * sin(yaw) + acc_x * cos(yaw)
            North (y) = acc_y * cos(yaw) - acc_x * sin(yaw)
        """
        yaw_rad = np.radians(yaw_deg)
        nav_x = acc_y_body * np.sin(yaw_rad) + acc_x_body * np.cos(yaw_rad)
        nav_y = acc_y_body * np.cos(yaw_rad) - acc_x_body * np.sin(yaw_rad)
        return nav_x, nav_y

    def step(self, acc_x, acc_y, acc_z, yaw_deg, dt):
        """
        Process one IMU sample and update internal state.

        Parameters
        ----------
        acc_x, acc_y, acc_z : float
            Raw accelerometer readings in body frame (m/s²).
        yaw_deg : float
            Orientation yaw/azimuth in degrees (0=N, 90=E, clockwise).
        dt : float
            Time step in seconds since last sample.

        Returns
        -------
        (vel_x, vel_y, pos_x, pos_y) : tuple of float
            Current INS state estimate in ENU frame (m/s, meters).
        """
        # --- Bias calibration phase (first N_CAL samples assumed stationary) ---
        if not self._calibrated:
            self._cal_samples.append((acc_x, acc_y))
            if len(self._cal_samples) >= self._N_CAL:
                self._bias_x = np.mean([s[0] for s in self._cal_samples])
                self._bias_y = np.mean([s[1] for s in self._cal_samples])
                self._calibrated = True
                self._cal_samples = []
            else:
                return 0.0, 0.0, 0.0, 0.0

        # --- Remove bias (horizontal) and gravity (vertical) ---
        lin_x = acc_x - self._bias_x
        lin_y = acc_y - self._bias_y
        # acc_z - gravity: not used for 2D, but kept for completeness

        # --- Rotate to navigation frame ---
        nav_acc_x, nav_acc_y = self._rotate_body_to_nav(lin_x, lin_y, yaw_deg)

        # --- Trapezoidal integration ---
        if dt > 0:
            # velocity += 0.5 * (current_acc + previous_acc) * dt
            self.vel_x += 0.5 * (nav_acc_x + self._prev_nav_acc_x) * dt
            self.vel_y += 0.5 * (nav_acc_y + self._prev_nav_acc_y) * dt
            # position += vel * dt  (using average of old and new velocity
            # is embedded in the sequential update)
            self.pos_x += self.vel_x * dt
            self.pos_y += self.vel_y * dt

        self._prev_nav_acc_x = nav_acc_x
        self._prev_nav_acc_y = nav_acc_y

        return self.vel_x, self.vel_y, self.pos_x, self.pos_y

    def reset_state(self, vel_x, vel_y, pos_x, pos_y):
        """Reset internal state to the provided values (closed-loop EKF feedback)."""
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.pos_x = pos_x
        self.pos_y = pos_y



if __name__ == "__main__":
    # Quick standalone sanity check
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from io_utils import load_s_file
    from discover_sessions import discover_all_sessions

    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None,
                        help="Session name (e.g. 'S1'). Default: first found.")
    args = parser.parse_args()

    sessions = discover_all_sessions()
    if not sessions:
        print("No sessions found. Check config.DATASET_ROOT.")
        sys.exit(1)

    if args.session:
        session = next((s for s in sessions if s["name"] == args.session), None)
        if not session:
            print(f"Session '{args.session}' not found.")
            sys.exit(1)
    else:
        session = sessions[0]

    print(f"Running INS mechanization on: {session['name']}")
    raw = load_s_file(session["s_path"])
    result = run_ins_mechanization(raw)

    final_drift = np.linalg.norm(result[["ins_pos_x", "ins_pos_y"]].iloc[-1].to_numpy())
    duration = result["_t_sec"].iloc[-1]
    print(f"Session duration: {duration:.1f} s")
    print(f"Raw INS-only final position estimate: "
          f"x={result['ins_pos_x'].iloc[-1]:.2f} m, y={result['ins_pos_y'].iloc[-1]:.2f} m")
    print(f"(Distance from origin: {final_drift:.2f} m -- this is expected to be "
          f"WRONG / drifted. That's exactly the error the AI model will learn to predict.)")
    print(f"Final velocity: vx={result['ins_vel_x'].iloc[-1]:.2f} m/s, "
          f"vy={result['ins_vel_y'].iloc[-1]:.2f} m/s")
