"""
config.py
=========
Single source of truth for column names, file paths, and hyperparameters.

Configured for the IO-VNBD (Inertial Odometry Vehicle Navigation Benchmark
Dataset) synchronized dataset.  The S-files contain smartphone IMU data and
the V-files contain vehicle CAN-bus ground truth.

IMPORTANT: The raw CSVs have leading spaces in column names and use latin-1
encoding for special characters (°, ², µ).  All loading must use
encoding='latin-1' and strip column names.
"""

import os

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
PLOT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

for d in [DATA_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# Path to the real IO-VNBD synchronised dataset root.
DATASET_ROOT = os.path.join(
    os.path.dirname(PROJECT_ROOT),      # DeadReckon/
    "Data", "IO-VNBD",
    "Synchronised V abd S datasets",
    "Categorised IOVNB Dataset",
)

# CSV encoding used by IO-VNBD dataset files
CSV_ENCODING = "latin-1"

# ---------------------------------------------------------------------------
# 2. RAW COLUMN NAMES  — matched to the actual IO-VNBD CSV headers
#    (after stripping leading/trailing whitespace from column names)
# ---------------------------------------------------------------------------

# S-file (smartphone sensors) columns.  The ² character in "m/s²" may render
# differently depending on encoding, so we match using a partial-match fallback
# in the loader utility (src/io_utils.py).
IMU_COLUMNS = {
    "time":         "TIME SINCE START (ms)",
    "acc_x":        "ACCELEROMETER X",         # partial match — full has "(m/s²) "
    "acc_y":        "ACCELEROMETER Y",
    "acc_z":        "ACCELEROMETER Z",
    "grav_x":       "GRAVITY X",
    "grav_y":       "GRAVITY Y",
    "grav_z":       "GRAVITY Z",
    "gyro_yaw":     "GYROSCOPE Yaw (rad/s)",
    "gyro_pitch":   "GYROSCOPE Pitch (rad/s)",
    "gyro_roll":    "GYROSCOPE Roll (rad/s)",
    "orient_yaw":   "ORIENTATION (Yaw)",
    "orient_pitch": "ORIENTATION (Pitch)",
    "orient_roll":  "ORIENTATION (Roll",       # note: real column has trailing " )"
    "gps_lat":      "GPS LATITUDE (degrees)",
    "gps_lon":      "GPS LONGITUDE (degrees)",
    "gps_speed":    "GPS SPEED (Kmh)",
    "gps_accuracy": "GPS ACCURACY (m)",
    "gps_orient":   "GPS ORIENTATION",
    "gps_sats":     "GPS SATELLITES IN RANGE",
}

# V-file (vehicle CAN-bus ground truth) columns
GT_COLUMNS = {
    "time":         "Time Since Start of Day (seconds)",
    "lat":          "Latitude (degrees)",
    "lon":          "Longitude (degrees)",
    "speed":        "Velocity (km/hr)",
    "heading":      "Heading (degrees)",
    "height":       "Height (km)",
    "vert_vel":     "Vertical velocity (km/hr)",
    "sample_period":"Sample period (seconds)",
    "yaw_rate":     "Yaw Rate (deg/sec)",
    "veh_speed":    "Indicated Vehicle Speed (km/hr)",
    "long_acc":     "Indicated Longitudinal Acceleration (g)",
    "lat_acc":      "Indicated Lateral Acceleration (g)",
    "ws_fl":        "Wheel Speed Front Left (rad/sec)",
    "ws_fr":        "Wheel Speed Front Right (rad/sec)",
    "ws_rl":        "Wheel Speed Rear Left (rad/sec)",
    "ws_rr":        "Wheel Speed Rear Right (rad/sec)",
    "gps_sats":     "No of GPS Satellites Available",
}

# ---------------------------------------------------------------------------
# 3. PHYSICS CONSTANTS
# ---------------------------------------------------------------------------
GRAVITY = 9.80665  # m/s^2

# Approximate wheel radius for converting wheel speed (rad/s) to m/s
WHEEL_RADIUS_M = 0.31

# Earth's radius for lat/lon to meters conversion (equirectangular approx.)
EARTH_RADIUS_M = 6_371_000.0

# ---------------------------------------------------------------------------
# 4. WINDOWING / MODEL HYPERPARAMETERS
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 10           # IO-VNBD smartphone samples at ~10 Hz
WINDOW_SECONDS = 2.0          # how much IMU history the AI sees at once
WINDOW_SIZE = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)   # 20 samples
STRIDE = 5                    # slide by 0.5s (5 samples at 10 Hz)

# IMU channels fed into the network per timestep (6 channels)
IMU_FEATURES = ["acc_x", "acc_y", "acc_z", "gyro_yaw", "gyro_pitch", "gyro_roll"]

# INS auxiliary state fed in alongside the IMU window (velocity features only, avoiding non-stationary position features)
INS_STATE_FEATURES = ["ins_vel_x", "ins_vel_y"]

# What the network predicts: the INS's error in velocity (m/s)
TARGET_FEATURES = ["err_vel_x", "err_vel_y"]

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# SIGN CONVENTION (critical — must be consistent in train, inference, EKF):
#
#   Label definition:   err_vel = true_vel - ins_vel
#   At inference:        corrected_vel = ins_vel + predicted_err
#
# This is ADDITION, not subtraction. If the AI predicts the error is +5 m/s,
# the INS is 5 m/s too slow, so we add 5 to correct it.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Clip velocity error labels at ±ERR_VEL_CLIP m/s during dataset building.
# Windows where the INS has already drifted beyond this are not representative
# of real-time usage (where corrections are applied every 0.1s) and make
# regression intractable.
ERR_VEL_CLIP = 50.0  # m/s

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 5. EKF TUNING PARAMETERS
# ---------------------------------------------------------------------------
# State vector: [pos_x, pos_y, vel_x, vel_y]  (local ENU meters & m/s)

# Process noise standard deviations per timestep
EKF_Q_POS_STD = 0.5       # m — position process noise std
EKF_Q_VEL_STD = 1.0       # m/s — velocity process noise std

# Measurement noise: GNSS position
EKF_R_GNSS_POS_STD = 5.0  # m — smartphone GPS accuracy (~5 m)

# Measurement noise: AI velocity correction
EKF_R_AI_VEL_STD = 0.5    # m/s — trust AI corrections moderately

# GNSS accuracy threshold — ignore GPS fixes worse than this in the EKF
GNSS_BLACKOUT_THRESHOLD_M = 50.0

# ---------------------------------------------------------------------------
# 6. SEAMLESS ADAPTIVE GNSS ⇄ AI-IDR SWITCHING & BATTERY OPTIMIZATION
# ---------------------------------------------------------------------------
# Signal Quality Thresholds
GNSS_THRESHOLD_GOOD_M = 10.0      # GPS accuracy <= 10m is GOOD
GNSS_THRESHOLD_DEGRADED_M = 25.0  # GPS accuracy <= 25m is DEGRADED, >25m or blackout is LOST
GNSS_MIN_SATS_GOOD = 6            # Minimum satellite count for GOOD state

# Hysteresis Debounce Counters (at ~10 Hz sample rate)
N_RECOVERY_CYCLES = 20            # 2.0 seconds stable GOOD signal to transition from DEGRADED/LOST -> GOOD
N_DEGRADE_CYCLES = 5              # 0.5 seconds unstable signal to transition -> DEGRADED
N_LOST_CYCLES = 10                # 1.0 seconds poor/missing signal to transition -> LOST

# Innovation Gating (Mahalanobis distance chi-square threshold for 2D position, 99% confidence)
EKF_INNOVATION_GATE_CHI2 = 9.21

# Battery Optimization Flag (AI Sleeping in GOOD state)
AI_SLEEP_ON_GOOD_GNSS = True

# Non-Holonomic Constraint (NHC) noise std
NHC_LATERAL_VEL_STD = 0.2         # m/s — zero lateral velocity constraint

# ---------------------------------------------------------------------------
# 7. GNSS BLACKOUT SIMULATION
# ---------------------------------------------------------------------------
BLACKOUT_DURATION_S = 60     # seconds of simulated GPS outage
BLACKOUT_START_FRAC = 0.4    # start blackout at 40% through the session

