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
from src.model import gaussian_nll_loss, mean_error_mae, combined_huber_nll_loss


class LiveErrorCorrector:
    def __init__(self, model_path=None, stats_path=None):
        model_path = model_path or os.path.join(config.MODEL_DIR, "ins_error_model_final.keras")
        stats_path = stats_path or os.path.join(config.MODEL_DIR, "normalization_stats.npz")

        custom_objs = {
            "gaussian_nll_loss": gaussian_nll_loss,
            "mean_error_mae": mean_error_mae,
            "combined_huber_nll_loss": combined_huber_nll_loss
        }
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
        self.stats = np.load(stats_path)
        self.buffer = deque(maxlen=config.WINDOW_SIZE)

        # Standard graph function call (jit_compile=False to avoid Windows XLA CPU compilation hangs)
        @tf.function(jit_compile=False)
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

    def correct(self, ins_state: np.ndarray, return_std: bool = False):
        """
        Predict the INS velocity error given the current IMU buffer + INS state.

        Parameters
        ----------
        ins_state : np.ndarray, shape (2,)
            [ins_vel_x, ins_vel_y]
        return_std : bool, optional
            If True, also returns the predicted standard deviation [std_x, std_y] (in m/s).

        Returns
        -------
        err_vel : np.ndarray, shape (2,)
            [err_vel_x, err_vel_y] — predicted velocity error.
        std_vel : np.ndarray, shape (2,), optional
            [std_x, std_y] — predicted standard deviation in m/s (if model has uncertainty head).
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

        if len(pred_n) >= 4:
            mean_n = pred_n[:2]
            log_var_n = pred_n[2:]
            log_var_clamped = np.clip(log_var_n, getattr(config, "LOG_VAR_MIN", -7.0), getattr(config, "LOG_VAR_MAX", 7.0))
            
            pred_err = mean_n * self.stats["y_std"] + self.stats["y_mean"]
            pred_std = np.sqrt(np.exp(log_var_clamped)) * self.stats["y_std"]

            pred_err = np.nan_to_num(pred_err, nan=0.0, posinf=0.0, neginf=0.0)
            pred_std = np.nan_to_num(pred_std, nan=config.EKF_R_AI_VEL_STD, posinf=config.EKF_R_AI_VEL_STD, neginf=config.EKF_R_AI_VEL_STD)
            
            if return_std:
                return pred_err, pred_std
            return pred_err
        else:
            pred_err = pred_n[:2] * self.stats["y_std"] + self.stats["y_mean"]
            pred_err = np.nan_to_num(pred_err, nan=0.0, posinf=0.0, neginf=0.0)
            if return_std:
                std_fallback = np.full(2, config.EKF_R_AI_VEL_STD, dtype=np.float32)
                return pred_err, std_fallback
            return pred_err


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
