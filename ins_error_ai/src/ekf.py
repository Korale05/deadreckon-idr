"""
ekf.py
======
STEP 4 of the plan: Extended Kalman Filter that fuses three information sources.

The EKF ties together everything in the system:
    1. INS prediction (from ins_mechanization.py) — propagates state forward
       using the physics-based dead-reckoning estimate.
    2. AI correction (from inference.py) — a learned velocity correction that
       compensates for the INS's systematic drift patterns.
    3. GNSS measurement (smartphone GPS) — absolute position fix, when available.

State vector: x = [pos_east, pos_north, vel_east, vel_north]  (4-state)

The EKF naturally handles GNSS blackouts: when GPS is unavailable, only the
predict step (INS) and the AI velocity update run.  No special-case code is
needed — the filter simply has fewer measurements to incorporate, and the
covariance grows (uncertainty increases), which is exactly the right behavior.

This matches diagrams 1-4 from the architecture:
    Diagram 1/3:  IMU -> INS state ---+---> EKF ---> Final state
                  IMU -> AI -> correction --+          ^
                                                       |
    Diagram 2:    GNSS (absolute position) ------------+
    Diagram 4:    GNSS X (crossed out) — EKF runs on INS+AI only
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class NavigationEKF:
    """
    4-state Extended Kalman Filter for INS/AI/GNSS fusion.

    State: [pos_x, pos_y, vel_x, vel_y]  (ENU meters and m/s)

    Prediction model (constant-velocity):
        pos_new = pos_old + vel_old * dt
        vel_new = vel_old  (+ process noise)

    Measurement models:
        1. GNSS: measures [pos_x, pos_y] directly
        2. AI:   measures [vel_x, vel_y] (corrected velocity)
    """

    def __init__(self):
        # State vector: [pos_x, pos_y, vel_x, vel_y]
        self.x = np.zeros(4)

        # State covariance matrix
        self.P = np.eye(4) * 100.0  # start with high uncertainty

        # Process noise (tuned via config)
        q_pos = config.EKF_Q_POS_STD ** 2
        q_vel = config.EKF_Q_VEL_STD ** 2
        self.Q_base = np.diag([q_pos, q_pos, q_vel, q_vel])

        # GNSS measurement noise
        r_gps = config.EKF_R_GNSS_POS_STD ** 2
        self.R_gnss = np.diag([r_gps, r_gps])

        # AI velocity correction measurement noise
        r_ai = config.EKF_R_AI_VEL_STD ** 2
        self.R_ai = np.diag([r_ai, r_ai])

        # GNSS measurement matrix: observes [pos_x, pos_y]
        self.H_gnss = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # AI measurement matrix: observes [vel_x, vel_y]
        self.H_ai = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

    def reset(self, pos_x=0.0, pos_y=0.0, vel_x=0.0, vel_y=0.0):
        """Reset filter state and covariance."""
        self.x = np.array([pos_x, pos_y, vel_x, vel_y])
        self.P = np.eye(4) * 100.0

    def predict(self, dt: float, ins_vel: np.ndarray = None):
        """
        Prediction step: propagate state forward using the constant-velocity
        model, optionally informed by the INS velocity estimate.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        ins_vel : np.ndarray, optional
            [vel_x, vel_y] from the INS mechanization. If provided, the
            velocity state is updated to the INS estimate before propagation
            (this is the "INS prediction" arrow in the diagram).
        """
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # If INS velocity is provided, inject it as our velocity estimate
        if ins_vel is not None:
            self.x[2] = ins_vel[0]
            self.x[3] = ins_vel[1]

        # Propagate state
        self.x = F @ self.x

        # Propagate covariance: P = F P F^T + Q
        Q = self.Q_base * dt  # scale process noise by timestep
        self.P = F @ self.P @ F.T + Q

    def update_gnss(self, pos_measurement: np.ndarray, accuracy_m: float = None):
        """
        GNSS measurement update: correct state using GPS position fix.

        Parameters
        ----------
        pos_measurement : np.ndarray
            [pos_x, pos_y] in ENU meters from session start.
        accuracy_m : float, optional
            GPS accuracy in meters. If provided, overrides default R_gnss.
        """
        H = self.H_gnss
        z = pos_measurement

        if accuracy_m is not None and accuracy_m > 0:
            R = np.diag([accuracy_m**2, accuracy_m**2])
        else:
            R = self.R_gnss

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_ai_velocity(self, corrected_vel: np.ndarray):
        """
        AI velocity correction update: the AI predicts the INS's velocity
        error, we subtract it from the INS velocity to get a "corrected"
        velocity, then feed that corrected velocity in as a measurement.

        Parameters
        ----------
        corrected_vel : np.ndarray
            [vel_x, vel_y] — the INS velocity AFTER subtracting the AI's
            predicted error.
        """
        H = self.H_ai
        z = corrected_vel
        R = self.R_ai

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    @property
    def position(self) -> np.ndarray:
        """Current estimated position [pos_x, pos_y]."""
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current estimated velocity [vel_x, vel_y]."""
        return self.x[2:4].copy()

    @property
    def position_uncertainty(self) -> float:
        """1-sigma position uncertainty (meters)."""
        return np.sqrt(self.P[0, 0] + self.P[1, 1])

    @property
    def velocity_uncertainty(self) -> float:
        """1-sigma velocity uncertainty (m/s)."""
        return np.sqrt(self.P[2, 2] + self.P[3, 3])


def run_ekf_fusion(ins_df, ai_corrections=None, gnss_available=None,
                   gnss_pos=None, gnss_accuracy=None):
    """
    Run the full EKF fusion over a session.

    Parameters
    ----------
    ins_df : pd.DataFrame
        Output of run_ins_mechanization(), with ins_vel_x/y, ins_pos_x/y, _t_sec.
    ai_corrections : np.ndarray, optional
        (N, 2) array of predicted velocity errors [err_vel_x, err_vel_y].
        If None, no AI correction is applied (pure INS + GNSS).
    gnss_available : np.ndarray, optional
        Boolean array of length N. True where GPS is available.
        If None, GPS is always available (when gnss_pos is provided).
    gnss_pos : np.ndarray, optional
        (N, 2) array of GPS positions [pos_x, pos_y] in ENU meters.
    gnss_accuracy : np.ndarray, optional
        (N,) array of GPS accuracy values in meters.

    Returns
    -------
    dict with keys: 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'pos_unc', 'vel_unc'
        Each is a numpy array of length N.
    """
    n = len(ins_df)
    t = ins_df["_t_sec"].to_numpy(dtype=float)
    ins_vel_x = ins_df["ins_vel_x"].to_numpy(dtype=float)
    ins_vel_y = ins_df["ins_vel_y"].to_numpy(dtype=float)

    ekf = NavigationEKF()

    # Initialize with first GNSS position if available
    if gnss_pos is not None and len(gnss_pos) > 0:
        ekf.reset(pos_x=gnss_pos[0, 0], pos_y=gnss_pos[0, 1])

    # Output arrays
    out_pos_x = np.zeros(n)
    out_pos_y = np.zeros(n)
    out_vel_x = np.zeros(n)
    out_vel_y = np.zeros(n)
    out_pos_unc = np.zeros(n)
    out_vel_unc = np.zeros(n)

    for i in range(n):
        dt = t[i] - t[i-1] if i > 0 else 0.1

        # 1. PREDICT: propagate forward using INS velocity
        if i > 0:
            ins_vel = np.array([ins_vel_x[i], ins_vel_y[i]])
            ekf.predict(dt, ins_vel=ins_vel)

        # 2. UPDATE with AI velocity correction (if available)
        # SIGN CONVENTION: err = true - ins → corrected = ins + err (ADDITION)
        if ai_corrections is not None and i < len(ai_corrections):
            corrected_vel = np.array([
                ins_vel_x[i] + ai_corrections[i, 0],  # ADD predicted error
                ins_vel_y[i] + ai_corrections[i, 1],
            ])
            ekf.update_ai_velocity(corrected_vel)

        # 3. UPDATE with GNSS (if available, not blacked out, and accuracy OK)
        gps_ok = True
        if gnss_available is not None:
            gps_ok = gnss_available[i]
        # Reject GPS fixes with accuracy worse than threshold
        if gps_ok and gnss_accuracy is not None:
            gps_ok = gps_ok and (gnss_accuracy[i] < config.GNSS_BLACKOUT_THRESHOLD_M)
        if gps_ok and gnss_pos is not None and i < len(gnss_pos):
            acc = gnss_accuracy[i] if gnss_accuracy is not None else None
            ekf.update_gnss(gnss_pos[i], accuracy_m=acc)

        # Record output
        out_pos_x[i] = ekf.position[0]
        out_pos_y[i] = ekf.position[1]
        out_vel_x[i] = ekf.velocity[0]
        out_vel_y[i] = ekf.velocity[1]
        out_pos_unc[i] = ekf.position_uncertainty
        out_vel_unc[i] = ekf.velocity_uncertainty

    return {
        "pos_x": out_pos_x,
        "pos_y": out_pos_y,
        "vel_x": out_vel_x,
        "vel_y": out_vel_y,
        "pos_unc": out_pos_unc,
        "vel_unc": out_vel_unc,
    }


if __name__ == "__main__":
    # Quick demo with synthetic data
    print("EKF module loaded successfully.")
    ekf = NavigationEKF()
    print(f"Initial state: {ekf.x}")
    print(f"Initial uncertainty: pos={ekf.position_uncertainty:.1f} m, "
          f"vel={ekf.velocity_uncertainty:.1f} m/s")

    # Simulate a few steps
    for i in range(10):
        ekf.predict(0.1, ins_vel=np.array([5.0, 0.1]))
        if i % 3 == 0:
            ekf.update_gnss(np.array([i * 0.5, 0.01]))
        ekf.update_ai_velocity(np.array([5.0, 0.1]))

    print(f"After 10 steps: pos={ekf.position}, vel={ekf.velocity}")
    print(f"Uncertainty: pos={ekf.position_uncertainty:.2f} m, "
          f"vel={ekf.velocity_uncertainty:.2f} m/s")
