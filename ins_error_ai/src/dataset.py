"""
dataset.py
==========
Turns the labeled per-session CSVs (from generate_labels.py) into fixed-size
sliding windows that the neural network can actually consume.

Each training example =
    X_imu   : (WINDOW_SIZE, 6)  -- a short chunk of raw accel+gyro history
    X_state : (4,)              -- the INS's own vel/pos estimate at the END
                                     of that window (extra context, per the
                                     diagram: "INS state -> AI" hint)
    y       : (2,)              -- true_vel - ins_vel at the END of the window
                                     (the correction the AI must predict)
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def make_windows_from_session(df: pd.DataFrame):
    """
    Slice one labeled session DataFrame into sliding-window training examples.
    The labeled CSV has standardized column keys (acc_x, acc_y, ...,
    ins_vel_x, ins_vel_y, ins_pos_x, ins_pos_y, err_vel_x, err_vel_y).

    Windows where the velocity error exceeds config.ERR_VEL_CLIP are discarded
    (the INS has drifted too far for meaningful regression).
    """
    # IMU features: raw sensor channels (6 channels)
    imu_col_keys = config.IMU_FEATURES  # ["acc_x", "acc_y", "acc_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]

    # Check that all required columns exist
    missing = [k for k in imu_col_keys if k not in df.columns]
    if missing:
        print(f"  [warn] Missing IMU columns: {missing}")
        return None

    missing_state = [k for k in config.INS_STATE_FEATURES if k not in df.columns]
    if missing_state:
        print(f"  [warn] Missing INS state columns: {missing_state}")
        return None

    missing_target = [k for k in config.TARGET_FEATURES if k not in df.columns]
    if missing_target:
        print(f"  [warn] Missing target columns: {missing_target}")
        return None

    imu_matrix = df[imu_col_keys].to_numpy(dtype=np.float32)
    state_matrix = df[config.INS_STATE_FEATURES].to_numpy(dtype=np.float32)
    target_matrix = df[config.TARGET_FEATURES].to_numpy(dtype=np.float32)

    # Replace NaN/inf with 0 (defensive)
    imu_matrix = np.nan_to_num(imu_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    state_matrix = np.nan_to_num(state_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    target_matrix = np.nan_to_num(target_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    W = config.WINDOW_SIZE
    S = config.STRIDE
    clip = config.ERR_VEL_CLIP
    n = len(df)

    X_imu, X_state, y = [], [], []
    n_clipped = 0
    for end in range(W, n, S):
        start = end - W
        target = target_matrix[end - 1]

        # --- Label clipping: discard windows with absurd velocity errors ---
        if np.any(np.abs(target) > clip):
            n_clipped += 1
            continue

        X_imu.append(imu_matrix[start:end])          # shape (W, 6)
        X_state.append(state_matrix[end - 1])         # INS state at window end
        y.append(target)                               # error to correct at window end

    if n_clipped > 0:
        total_possible = max(1, (n - W) // S)
        print(f"    clipped {n_clipped}/{total_possible + n_clipped} windows "
              f"(|err_vel| > {clip} m/s)")

    if not X_imu:
        return None
    return np.stack(X_imu), np.stack(X_state), np.stack(y)


def build_full_dataset(save_path: str = None):
    processed_files = sorted(glob.glob(os.path.join(config.PROCESSED_DIR, "*_labeled.csv")))
    if not processed_files:
        raise FileNotFoundError(
            f"No labeled sessions found in {config.PROCESSED_DIR}. "
            f"Run generate_labels.py first."
        )

    all_imu, all_state, all_y, all_session_id = [], [], [], []
    session_names = []
    for i, f in enumerate(processed_files):
        name = os.path.basename(f).replace("_labeled.csv", "")
        session_names.append(name)
        df = pd.read_csv(f)
        result = make_windows_from_session(df)
        if result is None:
            print(f"[skip] {os.path.basename(f)} too short or missing columns")
            continue
        X_imu, X_state, y = result
        all_imu.append(X_imu)
        all_state.append(X_state)
        all_y.append(y)
        all_session_id.append(np.full(len(X_imu), i))
        print(f"[ok] {name} -> {len(X_imu)} windows")

    if not all_imu:
        raise ValueError("No valid windows were generated from any session.")

    X_imu = np.concatenate(all_imu, axis=0)
    X_state = np.concatenate(all_state, axis=0)
    y = np.concatenate(all_y, axis=0)
    session_id = np.concatenate(all_session_id, axis=0)

    print(f"\nTotal windows: {X_imu.shape[0]}  |  IMU window shape: {X_imu.shape[1:]}  "
          f"|  state dim: {X_state.shape[1]}  |  target dim: {y.shape[1]}")
    print(f"Target (err_vel) stats after clipping:")
    print(f"  err_vel_x: mean={y[:, 0].mean():.3f}, std={y[:, 0].std():.3f}, "
          f"min={y[:, 0].min():.3f}, max={y[:, 0].max():.3f}")
    print(f"  err_vel_y: mean={y[:, 1].mean():.3f}, std={y[:, 1].std():.3f}, "
          f"min={y[:, 1].min():.3f}, max={y[:, 1].max():.3f}")

    if save_path is None:
        save_path = os.path.join(config.PROCESSED_DIR, "windowed_dataset.npz")
    np.savez_compressed(save_path, X_imu=X_imu, X_state=X_state, y=y, session_id=session_id)
    print(f"Saved -> {save_path}")

    # Save session name manifest so train/val splits are inspectable
    manifest_path = save_path.replace(".npz", "_manifest.json")
    manifest = {
        "session_names": session_names,
        "session_id_to_name": {str(i): name for i, name in enumerate(session_names)},
        "total_windows": int(X_imu.shape[0]),
        "clip_threshold": config.ERR_VEL_CLIP,
        "window_size": config.WINDOW_SIZE,
        "stride": config.STRIDE,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest -> {manifest_path}")

    return X_imu, X_state, y, session_id


if __name__ == "__main__":
    build_full_dataset()
