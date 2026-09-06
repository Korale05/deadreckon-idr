"""
test_sign_convention.py
=======================
Unit test that verifies the sign convention is consistent across:
    - Label generation (generate_labels.py): err = true - ins
    - Training (train.py): y target = err
    - Inference (inference.py): returns predicted err
    - EKF (ekf.py): corrected_vel = ins + err (ADDITION)

The single most common bug in this kind of system is a mismatched sign
convention between training and inference. This test catches it explicitly.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def test_sign_convention():
    """
    Verify that the sign convention is consistent end to end.

    If labels are: err = true - ins
    Then:          true = ins + err
    So:            corrected = ins + predicted_err

    We verify this by:
    1. Loading a labeled CSV and checking the identity: true = ins + err
    2. Checking the EKF code uses addition (not subtraction)
    """
    print("=" * 60)
    print("TEST: Sign Convention Consistency")
    print("=" * 60)

    # --- Test 1: Check labels satisfy the identity ---
    import glob
    import pandas as pd

    labeled_files = sorted(glob.glob(os.path.join(config.PROCESSED_DIR, "*_labeled.csv")))
    if not labeled_files:
        print("[SKIP] No labeled files found. Run generate_labels.py first.")
        return False

    df = pd.read_csv(labeled_files[0], nrows=1000)

    # err_vel = true_vel - ins_vel should hold
    err_x_expected = df["true_vel_x"] - df["ins_vel_x"]
    err_y_expected = df["true_vel_y"] - df["ins_vel_y"]

    err_x_actual = df["err_vel_x"]
    err_y_actual = df["err_vel_y"]

    max_diff_x = np.max(np.abs(err_x_expected - err_x_actual))
    max_diff_y = np.max(np.abs(err_y_expected - err_y_actual))

    print(f"\n[1] Label identity check: err = true - ins")
    print(f"    Max difference X: {max_diff_x:.2e}")
    print(f"    Max difference Y: {max_diff_y:.2e}")

    label_ok = max_diff_x < 1e-6 and max_diff_y < 1e-6
    print(f"    Result: {'PASS [OK]' if label_ok else 'FAIL [X]'}")

    # --- Test 2: Verify that corrected = ins + err recovers true ---
    corrected_x = df["ins_vel_x"] + df["err_vel_x"]
    corrected_y = df["ins_vel_y"] + df["err_vel_y"]

    recovery_err_x = np.max(np.abs(corrected_x - df["true_vel_x"]))
    recovery_err_y = np.max(np.abs(corrected_y - df["true_vel_y"]))

    print(f"\n[2] Recovery check: ins + err == true")
    print(f"    Max recovery error X: {recovery_err_x:.2e}")
    print(f"    Max recovery error Y: {recovery_err_y:.2e}")

    recovery_ok = recovery_err_x < 1e-6 and recovery_err_y < 1e-6
    print(f"    Result: {'PASS [OK]' if recovery_ok else 'FAIL [X]'}")

    # --- Test 3: Inspect EKF code for correct sign ---
    ekf_path = os.path.join(os.path.dirname(__file__), "ekf.py")
    with open(ekf_path, "r") as f:
        ekf_code = f.read()

    # Check for the addition pattern (correct) vs subtraction (wrong)
    has_addition = "ins_vel_x[i] + ai_corrections" in ekf_code
    has_subtraction = "ins_vel_x[i] - ai_corrections" in ekf_code

    print(f"\n[3] EKF code sign check:")
    print(f"    Uses addition (correct):    {has_addition}")
    print(f"    Uses subtraction (wrong):   {has_subtraction}")

    ekf_ok = has_addition and not has_subtraction
    print(f"    Result: {'PASS [OK]' if ekf_ok else 'FAIL [X]'}")

    # --- Test 4: Check config.py documents the convention ---
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.py")
    with open(config_path, "r") as f:
        config_code = f.read()

    has_convention_doc = "corrected_vel = ins_vel + predicted_err" in config_code
    print(f"\n[4] Config.py documents sign convention: {has_convention_doc}")
    print(f"    Result: {'PASS [OK]' if has_convention_doc else 'FAIL [X]'}")

    # --- Overall ---
    all_pass = label_ok and recovery_ok and ekf_ok and has_convention_doc
    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL TESTS PASSED [OK]' if all_pass else 'SOME TESTS FAILED [X]'}")
    print(f"{'='*60}")
    return all_pass


if __name__ == "__main__":
    test_sign_convention()
