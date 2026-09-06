"""
inference.py
============
Real-time-usable wrapper for the trained AI error-correction model.

Given a rolling buffer of the last WINDOW_SIZE IMU samples plus the INS's
current running state, it returns the predicted velocity error.

SIGN CONVENTION (must match train.py and config.py):
    err_vel = true_vel - ins_vel
    corrected_vel = ins_vel + predicted_err   (ADDITION, not subtraction)
"""

import os
import sys
import time
from collections import deque

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LiveErrorCorrector:
    def __init__(self, model_path=None, stats_path=None):
        model_path = model_path or os.path.join(config.MODEL_DIR, "ins_error_model_final.keras")
        stats_path = stats_path or os.path.join(config.MODEL_DIR, "normalization_stats.npz")

        self.model = tf.keras.models.load_model(model_path)
        self.stats = np.load(stats_path)
        self.buffer = deque(maxlen=config.WINDOW_SIZE)

        # JIT-compile the prediction call to optimize CPU latency
        @tf.function(jit_compile=True)
        def predict_jit(x_imu, x_state):
            return self.model([x_imu, x_state], training=False)
        self.predict_jit = predict_jit

    def push_sample(self, imu_sample: np.ndarray):
        """
        Append one IMU sample to the rolling buffer.

        imu_sample: array of shape (6,) = [acc_x, acc_y, acc_z,
                                            gyro_yaw, gyro_pitch, gyro_roll]

        NaN/Inf values are replaced with 0.0 to prevent silent corruption
        of the live stream (a zeroed-out sample is safer than NaN propagation).
        """
        # TODO: iOS axis mapping differs (CoreMotion) — not implemented
        cleaned = np.nan_to_num(imu_sample, nan=0.0, posinf=0.0, neginf=0.0)
        self.buffer.append(cleaned)

    def ready(self) -> bool:
        return len(self.buffer) == config.WINDOW_SIZE

    def correct(self, ins_state: np.ndarray) -> np.ndarray:
        """
        Predict the INS velocity error given the current IMU buffer + INS state.

        Parameters
        ----------
        ins_state : np.ndarray, shape (2,)
            [ins_vel_x, ins_vel_y]

        Returns
        -------
        np.ndarray, shape (2,)
            [err_vel_x, err_vel_y] — the predicted velocity error.

            To get corrected velocity:
                corrected_vel = ins_state + predicted_err   (ADDITION)
        """
        if not self.ready():
            raise RuntimeError(
                f"Need {config.WINDOW_SIZE} IMU samples before correcting, "
                f"have {len(self.buffer)}."
            )

        X_imu = np.stack(list(self.buffer))[None, ...].astype(np.float32)   # (1, W, 6)
        X_state = np.nan_to_num(ins_state, nan=0.0, posinf=0.0, neginf=0.0)
        X_state = X_state[None, ...].astype(np.float32)                      # (1, 2)

        # Normalize using training statistics
        X_imu_n = (X_imu - self.stats["imu_mean"]) / self.stats["imu_std"]
        X_state_n = (X_state - self.stats["state_mean"]) / self.stats["state_std"]

        # Replace any residual NaN/Inf after normalization
        X_imu_n = np.nan_to_num(X_imu_n, nan=0.0, posinf=0.0, neginf=0.0)
        X_state_n = np.nan_to_num(X_state_n, nan=0.0, posinf=0.0, neginf=0.0)

        pred_n = self.predict_jit(X_imu_n, X_state_n).numpy()[0]
        pred = pred_n * self.stats["y_std"] + self.stats["y_mean"]

        # Final NaN/Inf guard on output
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        return pred  # [err_vel_x, err_vel_y]


if __name__ == "__main__":
    # Smoke test with random data — also measures inference latency.
    corrector = LiveErrorCorrector()
    for _ in range(config.WINDOW_SIZE):
        corrector.push_sample(np.random.randn(6) * 0.1)

    dummy_ins_state = np.array([5.0, 0.2])  # vel_x, vel_y

    # Warm-up call (first call is slow due to TF graph tracing)
    _ = corrector.correct(dummy_ins_state)

    # Timed inference
    n_calls = 100
    t0 = time.perf_counter()
    for _ in range(n_calls):
        correction = corrector.correct(dummy_ins_state)
    elapsed = (time.perf_counter() - t0) / n_calls * 1000  # ms per call

    print(f"Predicted velocity error: {correction}")
    # SIGN CONVENTION: corrected = ins + err (addition)
    corrected_velocity = dummy_ins_state[:2] + correction
    print(f"INS raw velocity:      {dummy_ins_state[:2]}")
    print(f"AI-corrected velocity: {corrected_velocity}")
    print(f"\nInference latency: {elapsed:.1f} ms per call "
          f"({'OK' if elapsed < 100 else 'TOO SLOW'} — budget is 100 ms at 10 Hz)")
