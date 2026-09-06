"""
train.py
========
Loads the windowed dataset (built by dataset.py), splits it into
train/val/test BY SESSION (never mixing windows from the same drive across
splits -- otherwise you leak information and get falsely great results),
normalizes inputs, trains the model, and saves the best checkpoint.

Uses Huber loss for robustness to outlier windows (per spec).
Reports per-epoch MAE in real m/s units (de-normalized).
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import build_error_correction_model


def session_wise_split(session_id, val_split, test_split, seed):
    rng = np.random.default_rng(seed)
    unique_sessions = np.unique(session_id)
    rng.shuffle(unique_sessions)

    n = len(unique_sessions)
    n_test = max(1, int(n * test_split)) if n > 2 else 0
    n_val = max(1, int(n * val_split)) if n > 2 else 0

    test_sessions = set(unique_sessions[:n_test])
    val_sessions = set(unique_sessions[n_test:n_test + n_val])
    train_sessions = set(unique_sessions[n_test + n_val:])

    if not train_sessions:  # tiny dataset (e.g. only 1-2 sessions) fallback
        train_sessions, val_sessions, test_sessions = set(unique_sessions), set(), set()

    train_mask = np.isin(session_id, list(train_sessions))
    val_mask = np.isin(session_id, list(val_sessions))
    test_mask = np.isin(session_id, list(test_sessions))
    return train_mask, val_mask, test_mask, train_sessions, val_sessions, test_sessions


def compute_normalization(X_imu_train, X_state_train, y_train):
    """Return mean/std stats so inference.py can apply identical scaling."""
    stats = {
        "imu_mean": X_imu_train.reshape(-1, X_imu_train.shape[-1]).mean(axis=0),
        "imu_std": X_imu_train.reshape(-1, X_imu_train.shape[-1]).std(axis=0) + 1e-8,
        "state_mean": X_state_train.mean(axis=0),
        "state_std": X_state_train.std(axis=0) + 1e-8,
        "y_mean": y_train.mean(axis=0),
        "y_std": y_train.std(axis=0) + 1e-8,
    }
    return stats


def apply_normalization(X_imu, X_state, y, stats):
    X_imu_n = (X_imu - stats["imu_mean"]) / stats["imu_std"]
    X_state_n = (X_state - stats["state_mean"]) / stats["state_std"]
    y_n = (y - stats["y_mean"]) / stats["y_std"] if y is not None else None
    return X_imu_n, X_state_n, y_n


class RealUnitMAECallback(tf.keras.callbacks.Callback):
    """Print validation MAE in real m/s units (de-normalized) at end of each epoch."""

    def __init__(self, y_mean, y_std):
        super().__init__()
        self.y_mean = y_mean
        self.y_std = y_std

    def on_epoch_end(self, epoch, logs=None):
        # The logged 'mae' is in normalized units. De-normalize:
        # normalized_mae * y_std ≈ real_mae  (approximate, but informative)
        if logs:
            train_mae_n = logs.get("mae", 0)
            val_mae_n = logs.get("val_mae", None)
            real_train_mae = train_mae_n * np.mean(np.abs(self.y_std))
            msg = f"  -> Real-unit MAE: train={real_train_mae:.3f} m/s"
            if val_mae_n is not None:
                real_val_mae = val_mae_n * np.mean(np.abs(self.y_std))
                msg += f", val={real_val_mae:.3f} m/s"
            print(msg)


def main():
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    data_path = os.path.join(config.PROCESSED_DIR, "windowed_dataset.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run: python -m src.generate_labels  then  "
            f"python -m src.dataset"
        )

    data = np.load(data_path)
    X_imu, X_state, y, session_id = data["X_imu"], data["X_state"], data["y"], data["session_id"]

    train_mask, val_mask, test_mask, train_sess, val_sess, test_sess = session_wise_split(
        session_id, config.VAL_SPLIT, config.TEST_SPLIT, config.RANDOM_SEED
    )

    X_imu_tr, X_state_tr, y_tr = X_imu[train_mask], X_state[train_mask], y[train_mask]
    X_imu_val, X_state_val, y_val = X_imu[val_mask], X_state[val_mask], y[val_mask]
    X_imu_te, X_state_te, y_te = X_imu[test_mask], X_state[test_mask], y[test_mask]

    print(f"Train windows: {len(y_tr)} | Val windows: {len(y_val)} | Test windows: {len(y_te)}")

    # Log the session split for reproducibility/inspection
    manifest_path = data_path.replace(".npz", "_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        id_to_name = manifest.get("session_id_to_name", {})
        print(f"\nTrain sessions ({len(train_sess)}): "
              f"{[id_to_name.get(str(int(s)), str(s)) for s in sorted(train_sess)][:10]}...")
        print(f"Val sessions ({len(val_sess)}): "
              f"{[id_to_name.get(str(int(s)), str(s)) for s in sorted(val_sess)]}")
        print(f"Test sessions ({len(test_sess)}): "
              f"{[id_to_name.get(str(int(s)), str(s)) for s in sorted(test_sess)]}")
    print()

    stats = compute_normalization(X_imu_tr, X_state_tr, y_tr)
    np.savez(os.path.join(config.MODEL_DIR, "normalization_stats.npz"), **stats)
    print(f"Normalization stats: y_mean={stats['y_mean']}, y_std={stats['y_std']}")

    X_imu_tr, X_state_tr, y_tr = apply_normalization(X_imu_tr, X_state_tr, y_tr, stats)
    if len(y_val):
        X_imu_val, X_state_val, y_val = apply_normalization(X_imu_val, X_state_val, y_val, stats)
    if len(y_te):
        X_imu_te, X_state_te, y_te = apply_normalization(X_imu_te, X_state_te, y_te, stats)

    model = build_error_correction_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.Huber(delta=1.0),  # Robust to outlier windows
        metrics=["mae"],
    )
    model.summary()

    checkpoint_path = os.path.join(config.MODEL_DIR, "ins_error_model_best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss" if len(y_val) else "loss",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if len(y_val) else "loss",
            patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if len(y_val) else "loss",
            factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(config.LOG_DIR, "training_log.csv")),
        RealUnitMAECallback(stats["y_mean"], stats["y_std"]),
    ]

    val_data = ([X_imu_val, X_state_val], y_val) if len(y_val) else None

    model.fit(
        [X_imu_tr, X_state_tr], y_tr,
        validation_data=val_data,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(config.MODEL_DIR, "ins_error_model_final.keras")
    model.save(final_path)
    print(f"\nSaved final model -> {final_path}")

    # --- Final evaluation in real units ---
    if len(y_val):
        y_val_pred_n = model.predict([X_imu_val, X_state_val], verbose=0)
        y_val_pred = y_val_pred_n * stats["y_std"] + stats["y_mean"]
        y_val_real = apply_normalization(
            X_imu[val_mask], X_state[val_mask], None, stats)[1]  # not needed
        # De-normalize the actual val targets
        data_reload = np.load(data_path)
        y_val_orig = data_reload["y"][val_mask]
        val_mae_real = np.mean(np.abs(y_val_pred - y_val_orig))
        val_mae_x = np.mean(np.abs(y_val_pred[:, 0] - y_val_orig[:, 0]))
        val_mae_y = np.mean(np.abs(y_val_pred[:, 1] - y_val_orig[:, 1]))
        print(f"\n{'='*60}")
        print(f"FINAL VALIDATION MAE (real units):")
        print(f"  err_vel_x MAE: {val_mae_x:.3f} m/s")
        print(f"  err_vel_y MAE: {val_mae_y:.3f} m/s")
        print(f"  Overall MAE:   {val_mae_real:.3f} m/s")
        print(f"{'='*60}")

    if len(y_te):
        y_te_pred_n = model.predict([X_imu_te, X_state_te], verbose=0)
        y_te_pred = y_te_pred_n * stats["y_std"] + stats["y_mean"]
        data_reload = np.load(data_path)
        y_te_orig = data_reload["y"][test_mask]
        test_mae_real = np.mean(np.abs(y_te_pred - y_te_orig))
        print(f"Held-out TEST MAE: {test_mae_real:.3f} m/s")


if __name__ == "__main__":
    main()
