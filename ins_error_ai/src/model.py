"""
model.py
========
STEP 3 of the plan: the actual AI model. This is the ONE trained network in
the whole system -- everything else (INS mechanization, EKF, map matching)
is classical, non-learned engineering.

Architecture, in plain words:
    - The raw IMU window (a short time-series) goes through 1D convolutions
      first, to pick up local shapes in the signal (a bump from a bad road,
      a steady turn, vibration patterns) -- this is much more efficient than
      a plain LSTM eating raw samples one at a time.
    - Those extracted features feed into an LSTM, which learns how the drift
      pattern evolves over the window (drift is not instantaneous -- it
      accumulates).
    - The INS's own current state (velocity/position estimate) is fed in
      through a separate small branch and concatenated in -- this gives the
      network a hint: "the INS already thinks it's going this fast / has
      drifted this much," which helps it predict how wrong that is.
    - A small dense head produces the final 2-value output: the predicted
      velocity error (x, y) to subtract from the INS's estimate.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_error_correction_model(
    window_size: int = config.WINDOW_SIZE,
    n_imu_features: int = len(config.IMU_FEATURES),
    n_state_features: int = len(config.INS_STATE_FEATURES),
    n_targets: int = len(config.TARGET_FEATURES),
    unroll_lstms: bool = False,
    enable_uncertainty: bool = None,
) -> tf.keras.Model:
    if enable_uncertainty is None:
        enable_uncertainty = config.ENABLE_UNCERTAINTY_HEAD

    # --- Branch 1: raw IMU window -> Conv1D -> LSTM ---
    imu_input = layers.Input(shape=(window_size, n_imu_features), name="imu_window")

    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(imu_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.LSTM(64, return_sequences=True, unroll=unroll_lstms)(x)
    x = layers.LSTM(32, unroll=unroll_lstms)(x)
    x = layers.Dropout(0.2)(x)

    # --- Branch 2: INS's own current state (velocity/position estimate) ---
    state_input = layers.Input(shape=(n_state_features,), name="ins_state")
    s = layers.Dense(16, activation="relu")(state_input)

    # --- Merge both branches ---
    merged = layers.Concatenate()([x, s])
    m = layers.Dense(64, activation="relu")(merged)
    m = layers.Dropout(0.2)(m)
    m = layers.Dense(32, activation="relu")(m)

    # --- Output head ---
    if enable_uncertainty:
        # Output 4 values: [pred_err_x, pred_err_y, log_var_x, log_var_y]
        output = layers.Dense(n_targets * 2, activation="linear", name="predicted_error_and_variance")(m)
    else:
        # Output 2 values: [pred_err_x, pred_err_y]
        output = layers.Dense(n_targets, activation="linear", name="predicted_error")(m)

    model = models.Model(inputs=[imu_input, state_input], outputs=output,
                          name="ins_error_correction_model")
    return model


@tf.keras.utils.register_keras_serializable(package="Custom", name="gaussian_nll_loss")
def gaussian_nll_loss(y_true, y_pred):
    """
    Heteroscedastic Gaussian Negative Log-Likelihood Loss.
    y_true: shape (batch_size, 2)  -- true velocity error (normalized)
    y_pred: shape (batch_size, 4)  -- [pred_mean_x, pred_mean_y, pred_log_var_x, pred_log_var_y]
    """
    mean = y_pred[:, :2]
    log_var = y_pred[:, 2:]

    # CRITICAL: Clamp log-variance to [-7.0, 7.0] to prevent numerical explosion & NaN loss
    log_var_clamped = tf.clip_by_value(log_var, config.LOG_VAR_MIN, config.LOG_VAR_MAX)
    inv_var = tf.exp(-log_var_clamped)

    sq_err = tf.square(y_true - mean)
    # Gaussian NLL formula: 0.5 * (exp(-s) * (y - mu)^2 + s)
    loss = 0.5 * (inv_var * sq_err + log_var_clamped)
    return tf.reduce_mean(loss)


@tf.keras.utils.register_keras_serializable(package="Custom", name="mean_error_mae")
def mean_error_mae(y_true, y_pred):
    """MAE metric on predicted mean velocity error."""
    return tf.reduce_mean(tf.abs(y_true - y_pred[:, :2]))


@tf.keras.utils.register_keras_serializable(package="Custom", name="combined_huber_nll_loss")
def combined_huber_nll_loss(y_true, y_pred):
    """
    Dual-weighted loss combining Huber Loss for mean velocity precision
    and Heteroscedastic NLL Loss for predicted uncertainty calibration.
    """
    huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred[:, :2])
    nll = gaussian_nll_loss(y_true, y_pred)
    return huber + 0.2 * nll


if __name__ == "__main__":
    m = build_error_correction_model()
    m.summary()

