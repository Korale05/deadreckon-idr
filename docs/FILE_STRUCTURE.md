# DeadReckon — Repository File Structure & Architecture

This document provides a comprehensive overview of the directory hierarchy, file organization, module responsibilities, and data flow within the **DeadReckon** project repository.

---

## 1. Directory Tree Visual

```
DeadReckon/
├── docs/                                 # Project documentation
│   └── FILE_STRUCTURE.md                 # Repository structure reference (this file)
├── Data/                                 # Raw benchmark datasets
│   └── IO-VNBD/                          # Inertial Odometry Vehicle Navigation Benchmark Dataset
├── ins_error_ai/                         # Core AI-assisted INS Python package
│   ├── config.py                         # Single source of truth for configuration & parameters
│   ├── README.md                         # Package-level architectural documentation
│   ├── requirements.txt                  # Python dependencies for ins_error_ai
│   ├── cache/                            # Preprocessed feature cache & numpy arrays
│   ├── data/                             # Data staging directory
│   │   ├── raw/                          # Raw session data references
│   │   └── processed/                    # Generated labels & windowed arrays
│   ├── outputs/                          # Generated model artifacts & logs
│   │   ├── logs/                         # Training & evaluation run logs
│   │   ├── models/                       # Keras models (.h5/.keras) and TFLite exports (.tflite)
│   │   └── plots/                        # Output trajectory plots & diagnostic figures
│   ├── results/                          # Benchmark evaluation metrics & CSV logs
│   │   ├── baseline_metrics.csv          # Pure INS baseline results
│   │   └── diagnostics/                  # Detailed error breakdown by session
│   └── src/                              # Source code modules
│       ├── __init__.py                   # Package initializer
│       ├── create_baseline.py            # Non-AI classical INS evaluation
│       ├── dataset.py                    # Windowing, scaling & TF/PyTorch Dataset loader
│       ├── diagnose_session.py           # Per-session error & drift diagnostic analyzer
│       ├── discover_sessions.py          # Session finder & IO-VNBD dataset indexer
│       ├── ekf.py                        # 4D Extended Kalman Filter implementation
│       ├── evaluate.py                   # Full system benchmarking & comparison engine
│       ├── export_tflite.py              # Model conversion script to TFLite format
│       ├── generate_labels.py            # Reference velocity error label generator
│       ├── inference.py                  # Real-time / streaming inference engine
│       ├── ins_mechanization.py          # Strapdown INS kinematics & double integration
│       ├── io_utils.py                   # File parsing, CSV standardizer & frame math
│       ├── map_matching.py               # Road network snap-to-node map matcher
│       ├── model.py                      # Conv1D-LSTM hybrid neural network architecture
│       ├── pipeline.py                   # End-to-end execution pipeline coordinator
│       ├── seamless_controller.py        # GPS blackout state machine & transition manager
│       ├── test_blackouts.py             # Simulated GNSS blackout resilience tester
│       ├── test_sign_convention.py       # Rotation angle & sign convention unit validator
│       ├── test_streaming_consistency.py # Batch vs. real-time stream consistency verifier
│       ├── train.py                      # Model training execution script
│       └── visualize.py                  # Trajectory & error plotting utilities
├── results/                              # Top-level evaluation output store
├── requirements.txt                      # Project root Python dependencies
└── README.md                             # High-level project summary
```

---

## 2. Top-Level Directory Breakdown

### `Data/`
Contains raw benchmark datasets used for training and evaluating dead-reckoning models.
* **`IO-VNBD/`**: Inertial Odometry Vehicle Navigation Benchmark Dataset containing synchronized smartphone sensor recordings (`S-files`) and vehicle CAN-bus ground truth (`V-files`).

### `results/`
Stores top-level aggregate experiment runs, generated trajectory comparison graphs, and diagnostic summaries across multiple benchmark passes.

### `ins_error_ai/`
The main working directory and package root for the AI-assisted INS correction pipeline. Contains environment setup files, configurations, datasets, outputs, and the source code modules in `src/`.

---

## 3. Core Source Modules (`ins_error_ai/src/`)

The core algorithms in [`ins_error_ai/src/`](file:///d:/Project/DeadReckon/ins_error_ai/src) are structured into functional modules:

### 3.1 INS Kinematics & Mechanics
* **[`ins_mechanization.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/ins_mechanization.py)**: Performs classical strapdown INS operations:
  * Phone body frame $\rightarrow$ Navigation ENU (East-North-Up) coordinate rotation using smartphone orientation azimuth/pitch/roll.
  * Gravity vector ($g = 9.80665\text{ m/s}^2$) removal.
  * Double trapezoidal integration of accelerations to compute instantaneous velocities ($\mathbf{v}_{\text{INS}}$) and positions ($\mathbf{p}_{\text{INS}}$).
* **[`generate_labels.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/generate_labels.py)**: Generates ground-truth velocity error target labels $\mathbf{y}_{\text{label}} = \mathbf{v}_{\text{GT}} - \mathbf{v}_{\text{INS}}$ with periodic INS re-anchoring to reflect realistic operational drift distributions during EKF tracking.

### 3.2 Deep Learning Model & Training
* **[`model.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/model.py)**: Implements the hybrid Conv1D-LSTM neural network with two input paths:
  1. *IMU Window Branch*: 1D Convolutional layers + BatchNorm + Pooling + Stacked LSTM layers to extract temporal features from raw IMU windows ($20 \text{ samples} \times 6 \text{ features}$).
  2. *INS State Branch*: Dense layers processing instantaneous INS kinematics (velocities and accelerations).
* **[`dataset.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/dataset.py)**: Handles sliding window generation, normalization, caching, and data loaders for training/validation sets.
* **[`train.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/train.py)**: Executes model training using Keras/TensorFlow, managing hyperparameter setup, loss monitoring, early stopping, and saving model checkpoints to `outputs/models/`.
* **[`export_tflite.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/export_tflite.py)**: Converts trained Keras `.h5`/`.keras` models into lightweight FlatBuffer TFLite (`.tflite`) format for embedded or mobile deployment.

### 3.3 Sensor Fusion & Filtering
* **[`ekf.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/ekf.py)**: Implements a 4D Extended Kalman Filter ($\mathbf{x} = [p_x, p_y, v_x, v_y]^T$):
  * *Time Propagation*: Integrates INS kinematic updates into the state estimate.
  * *AI Measurement Update*: Fuses model-predicted velocity errors $\mathbf{z}_{\text{AI}} = \mathbf{v}_{\text{INS}} + \hat{\mathbf{y}}_{\text{AI}}$.
  * *GNSS Measurement Update*: Fuses absolute GNSS position fixes when GNSS signal accuracy is within acceptable thresholds.
* **[`map_matching.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/map_matching.py)**: Post-processes filtered position trajectories by snapping coordinate points to nearest road network nodes.
* **[`seamless_controller.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/seamless_controller.py)**: Manages state transitions during GNSS availability, signal degradation, and total GNSS blackout conditions.

### 3.4 Execution Pipelines & Inference
* **[`pipeline.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/pipeline.py)**: Orchestrates the complete end-to-end flow: loading data $\rightarrow$ INS mechanization $\rightarrow$ sliding window inference $\rightarrow$ EKF state estimation $\rightarrow$ metric evaluation $\rightarrow$ visualization.
* **[`inference.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/inference.py)**: Streaming inference engine for real-time operation on windowed sensor feeds.
* **[`create_baseline.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/create_baseline.py)**: Evaluates uncorrected pure INS dead reckoning to produce baseline performance metrics.

### 3.5 Evaluation & Verification Suites
* **[`evaluate.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/evaluate.py)**: Compares performance metrics (Absolute Trajectory Error - ATE, RMSE, drift percentage over distance) across Pure INS, GNSS-only, and AI-Corrected INS + EKF trajectories.
* **[`diagnose_session.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/diagnose_session.py)**: Deep-dive diagnostic tool to inspect single-session error accumulation, velocity residual spikes, and state covariances.
* **[`test_blackouts.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/test_blackouts.py)**: Tests system resilience by simulating synthetic GNSS outage windows (e.g., 30s, 60s, 120s blackouts).
* **[`test_sign_convention.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/test_sign_convention.py)**: Unit test verifying frame transformation matrix directions and sign conventions.
* **[`test_streaming_consistency.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/test_streaming_consistency.py)**: Verifies that real-time chunked streaming predictions match offline batch processing outputs.

### 3.6 Data Utilities & Visualization
* **[`io_utils.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/io_utils.py)**: Handles encoding, header parsing, column normalization for IO-VNBD CSV files, and coordinate frame conversions.
* **[`discover_sessions.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/discover_sessions.py)**: Scans data directories to index smartphone (S-files) and vehicle (V-files) session pairs.
* **[`visualize.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/visualize.py)**: Generates 2D trajectory overlays, velocity profile comparisons, error histograms, and cumulative drift plots.

---

## 4. Pipeline Data Flow

```
[ Smartphone Sensors ]  ──►  [ ins_mechanization.py ]  ──►  v_INS, p_INS
 (Acc, Gyro 10Hz)                    │                           │
                                     ▼                           ▼
                             [ dataset.py ]            [ ekf.py State Prop ]
                                 (20-sample win)                 │
                                     │                           │
                                     ▼                           │
                             [ model.py (AI) ]                   │
                                     │                           │
                              Predicts err_v                     │
                                     │                           │
                                     ▼                           ▼
                             [ ekf.py Measurement Update ] ◄──────
                                     │
                                     ▼
                             [ map_matching.py ]
                                     │
                                     ▼
                             [ Fused Trajectory ]
```
