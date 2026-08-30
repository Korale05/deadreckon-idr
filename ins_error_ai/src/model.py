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
) -> tf.keras.Model:

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

    # --- Output: predicted velocity error (m/s), one value per target axis ---
    output = layers.Dense(n_targets, activation="linear", name="predicted_error")(m)

    model = models.Model(inputs=[imu_input, state_input], outputs=output,
                          name="ins_error_correction_model")
    return model


if __name__ == "__main__":
    m = build_error_correction_model()
    m.summary()
