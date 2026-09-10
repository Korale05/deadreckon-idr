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
from src.model import build_error_correction_model, gaussian_nll_loss, mean_error_mae, combined_huber_nll_loss
from src.dataset import apply_imu_augmentation


def session_wise_split(session_id, val_split, test_split, seed, id_to_name=None):
    rng = np.random.default_rng(seed)
    unique_sessions = np.unique(session_id)
    rng.shuffle(unique_sessions)

    # Force explicit test sessions (e.g. S1) into test_sessions set if requested
    explicit_test = getattr(config, "EXPLICIT_TEST_SESSIONS", [])
    test_sessions = set()
    candidate_sessions = []

    if id_to_name:
        for s in unique_sessions:
            s_name = id_to_name.get(str(int(s)), str(s))
            if s_name in explicit_test:
                test_sessions.add(s)
            else:
                candidate_sessions.append(s)
    else:
        candidate_sessions = list(unique_sessions)

    n_cand = len(candidate_sessions)
    n_test_needed = max(0, int(len(unique_sessions) * test_split) - len(test_sessions))
    n_val_needed = max(1, int(len(unique_sessions) * val_split)) if len(unique_sessions) > 2 else 0

    test_sessions.update(candidate_sessions[:n_test_needed])
    val_sessions = set(candidate_sessions[n_test_needed:n_test_needed + n_val_needed])
    train_sessions = set(candidate_sessions[n_test_needed + n_val_needed:])

    if not train_sessions:  # fallback
        train_sessions = set(candidate_sessions)

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

    def __init__(self, y_mean, y_std, enable_uncertainty=False):
        super().__init__()
        self.y_mean = y_mean
        self.y_std = y_std
        self.enable_uncertainty = enable_uncertainty

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            train_mae_n = logs.get("mae", logs.get("mean_absolute_error", 0))
            val_mae_n = logs.get("val_mae", logs.get("val_mean_absolute_error", None))
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

    manifest_path = data_path.replace(".npz", "_manifest.json")
    id_to_name = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        id_to_name = manifest.get("session_id_to_name", {})

    train_mask, val_mask, test_mask, train_sess, val_sess, test_sess = session_wise_split(
        session_id, config.VAL_SPLIT, config.TEST_SPLIT, config.RANDOM_SEED, id_to_name
    )

    X_imu_tr, X_state_tr, y_tr = X_imu[train_mask], X_state[train_mask], y[train_mask]
    X_imu_val, X_state_val, y_val = X_imu[val_mask], X_state[val_mask], y[val_mask]
    X_imu_te, X_state_te, y_te = X_imu[test_mask], X_state[test_mask], y[test_mask]

    # --- Apply IMU Data Augmentation on Training Split (if enabled) ---
    if getattr(config, "AUGMENT_IMU", False):
        X_aug, idx_aug = apply_imu_augmentation(X_imu_tr, ratio=config.AUGMENT_RATIO)
        if X_aug is not None:
            print(f"[Augmentation] Appending {len(X_aug)} augmented IMU windows to training set.")
            X_imu_tr = np.concatenate([X_imu_tr, X_aug], axis=0)
            X_state_tr = np.concatenate([X_state_tr, X_state_tr[idx_aug]], axis=0)
            y_tr = np.concatenate([y_tr, y_tr[idx_aug]], axis=0)

    print(f"Train windows: {len(y_tr)} | Val windows: {len(y_val)} | Test windows: {len(y_te)}")

    if id_to_name:
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

    enable_unc = getattr(config, "ENABLE_UNCERTAINTY_HEAD", False)
    model = build_error_correction_model(enable_uncertainty=enable_unc)

    if enable_unc:
        # Dual-Weighted Loss for [err_x, err_y, log_var_x, log_var_y]
        loss_fn = combined_huber_nll_loss
        metrics_list = [mean_error_mae]
        print("[Model] Initialized with 4-output Heteroscedastic Uncertainty Head + Combined Huber-NLL Loss")
    else:
        loss_fn = tf.keras.losses.Huber(delta=1.0)
        metrics_list = ["mae"]
        print("[Model] Initialized with Standard 2-output Error Prediction Head + Huber Loss")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn,
        metrics=metrics_list,
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
        mean_pred_n = y_val_pred_n[:, :2] if y_val_pred_n.shape[1] >= 2 else y_val_pred_n
        y_val_pred = mean_pred_n * stats["y_std"] + stats["y_mean"]
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
        mean_te_n = y_te_pred_n[:, :2] if y_te_pred_n.shape[1] >= 2 else y_te_pred_n
        y_te_pred = mean_te_n * stats["y_std"] + stats["y_mean"]
        data_reload = np.load(data_path)
        y_te_orig = data_reload["y"][test_mask]
        test_mae_real = np.mean(np.abs(y_te_pred - y_te_orig))
        print(f"Held-out TEST MAE: {test_mae_real:.3f} m/s")


if __name__ == "__main__":
    main()
