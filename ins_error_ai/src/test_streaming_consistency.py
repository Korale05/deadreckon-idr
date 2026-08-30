"""
test_streaming_consistency.py
==============================
Verify that the LiveErrorCorrector streaming interface produces the same
results as the batch pipeline for the same data.

Feeds one real session's data into push_sample() one row at a time
(simulating real-time arrival) and compares the output of correct() with
batch-computed predictions for the same timesteps.

This catches any batch-vs-streaming inconsistency before it becomes a
real-time bug that's much harder to debug live.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def test_streaming_consistency():
    print("=" * 60)
    print("TEST: Streaming vs Batch Consistency")
    print("=" * 60)

    # Load a labeled session
    import glob
    labeled_files = sorted(glob.glob(os.path.join(config.PROCESSED_DIR, "*_labeled.csv")))
    if not labeled_files:
        print("[SKIP] No labeled files found. Run generate_labels.py first.")
        return False

    # Use a small session for speed
    df = None
    for f in labeled_files:
        temp = pd.read_csv(f, nrows=5)
        n_rows = sum(1 for _ in open(f)) - 1
        if 500 < n_rows < 5000:
            df = pd.read_csv(f)
            print(f"Using session: {os.path.basename(f)} ({len(df)} samples)")
            break
    if df is None:
        df = pd.read_csv(labeled_files[0], nrows=2000)
        print(f"Using session: {os.path.basename(labeled_files[0])} (first 2000 samples)")

    # Check model exists
    model_path = os.path.join(config.MODEL_DIR, "ins_error_model_final.keras")
    stats_path = os.path.join(config.MODEL_DIR, "normalization_stats.npz")
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print("[SKIP] Trained model not found. Run train.py first.")
        return False

    from src.inference import LiveErrorCorrector
    from src.dataset import make_windows_from_session

    # Temporarily disable clipping to align batch and streaming indices exactly
    orig_clip = config.ERR_VEL_CLIP
    config.ERR_VEL_CLIP = 1e9

    # --- Method A: Batch prediction (same as evaluate.py) ---
    print("\n[A] Running batch prediction...")
    result = make_windows_from_session(df)
    
    # Restore original clip threshold
    config.ERR_VEL_CLIP = orig_clip

    if result is None:
        print("[SKIP] Could not build windows from this session.")
        return False

    X_imu_batch, X_state_batch, _ = result
    stats = np.load(stats_path)

    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    X_imu_n = (X_imu_batch - stats["imu_mean"]) / stats["imu_std"]
    X_state_n = (X_state_batch - stats["state_mean"]) / stats["state_std"]

    batch_pred_n = model.predict([X_imu_n, X_state_n], verbose=0)
    batch_pred = batch_pred_n * stats["y_std"] + stats["y_mean"]
    print(f"    Batch predictions shape: {batch_pred.shape}")

    # --- Method B: Streaming prediction ---
    print("[B] Running streaming prediction...")
    corrector = LiveErrorCorrector(model_path, stats_path)

    imu_keys = config.IMU_FEATURES
    state_keys = config.INS_STATE_FEATURES
    W = config.WINDOW_SIZE
    S = config.STRIDE

    streaming_preds = []
    window_count = 0
    n = len(df)

    t_start = time.perf_counter()
    for i in range(n):
        sample = np.array([df[k].iloc[i] for k in imu_keys], dtype=np.float32)
        corrector.push_sample(sample)

        # Only compare at window end points that match the batch windowing
        if corrector.ready() and i >= (W - 1):
            # Check if this is a window boundary matching batch stride
            window_end = i
            window_idx = (window_end - W + 1)
            if window_idx >= 0 and window_idx % S == 0:
                ins_state = np.array([df[k].iloc[i] for k in state_keys], dtype=np.float32)
                ins_state = np.nan_to_num(ins_state, nan=0.0)
                pred = corrector.correct(ins_state)
                streaming_preds.append((window_count, pred))
                window_count += 1

    t_elapsed = time.perf_counter() - t_start
    print(f"    Streaming predictions: {len(streaming_preds)} windows")
    print(f"    Total streaming time: {t_elapsed:.2f}s")

    if not streaming_preds:
        print("[SKIP] No streaming predictions generated.")
        return False

    # --- Compare ---
    print("\n[C] Comparing results...")
    n_compare = min(len(streaming_preds), len(batch_pred))
    max_diff = 0.0
    mean_diff = 0.0

    for idx, (batch_idx, stream_pred) in enumerate(streaming_preds[:n_compare]):
        if batch_idx < len(batch_pred):
            diff = np.max(np.abs(stream_pred - batch_pred[batch_idx]))
            max_diff = max(max_diff, diff)
            mean_diff += diff

    mean_diff /= max(n_compare, 1)

    print(f"    Compared {n_compare} windows")
    print(f"    Max absolute difference: {max_diff:.6f} m/s")
    print(f"    Mean absolute difference: {mean_diff:.6f} m/s")

    # Tolerance: numerical differences from float32 ops should be very small
    tolerance = 0.01  # 1 cm/s tolerance
    passed = max_diff < tolerance

    print(f"\n    Tolerance: {tolerance} m/s")
    print(f"    Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    # --- Timing check ---
    if streaming_preds:
        per_call_ms = (t_elapsed / len(streaming_preds)) * 1000
        print(f"\n[D] Inference timing:")
        print(f"    {per_call_ms:.1f} ms per correct() call")
        print(f"    Budget: 100 ms at 10 Hz")
        print(f"    Result: {'PASS ✓' if per_call_ms < 100 else 'FAIL ✗ (too slow)'}")

    print(f"\n{'='*60}")
    print(f"OVERALL: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"{'='*60}")
    return passed


if __name__ == "__main__":
    test_streaming_consistency()
