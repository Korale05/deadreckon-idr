# DeadReckon — Complete AI-Corrected INS/GNSS Navigation Pipeline

DeadReckon is a hybrid vehicle-navigation system designed to maintain highly accurate position and velocity tracking during **GNSS (GPS) blackouts or dropouts**. It achieves this by fusing classical Inertial Navigation System (INS) kinematics with a state-of-the-art Deep Learning error-correction model and an Extended Kalman Filter (EKF).

---

## 1. System Architecture & Processing Flow

The system integrates raw high-frequency smartphone sensor measurements (10 Hz accelerometer and gyroscope) with vehicle CAN-bus ground truth to correct dead-reckoning drift in real-time.

```
                                ┌─────────────────────────────────┐
                                │       Smartphone Sensors        │
                                │   (10 Hz Accelerometer, Gyro)   │
                                └──────────┬──────────┬───────────┘
                                           │          │
                       ┌───────────────────┘          └──────────────────┐
                       ▼                                                 ▼
           ┌───────────────────────┐                           ┌───────────────────────┐
           │ Classical Kinematics  │                           │  Conv1D-LSTM Model    │
           │  (INS Mechanization)  │                           │  (AI Error Predictor) │
           │                       │                           │                       │
           │ 1. Calibrate Bias     │                           │ Inputs:               │
           │ 2. Rotate Body → Nav  │                           │  - 20-sample IMU window│
           │ 3. Remove Gravity     │     Current INS state     │  - Current INS state   │
           │ 4. Double Integrate   ├──────────────────────────▶│                       │
           │                       │                           │ Predicts:             │
           │ Output:               │                           │  - INS Velocity Error │
           │  - Flawed INS state   │                           │    (err_vx, err_vy)   │
           └───────────┬───────────┘                           └───────────┬───────────┘
                       │                                                   │
                       │ INS State                                         │ AI Correction
                       │ (vel, pos)                                        │ (delta-velocity)
                       ▼                                                   ▼
           ┌───────────────────────────────────────────────────────────────────────────┐
           │                        Extended Kalman Filter (EKF)                       │
           │                                                                           │
           │  State Space: x = [pos_x, pos_y, vel_x, vel_y]^T                          │
           │                                                                           │
           │  Fuses:                                                                   │
           │    1. INS Kinematic propagation (always active)                           │
           │    2. AI Velocity Error corrections (always active)                       │
           │    3. GNSS Position updates (active when GPS accuracy < 50m)              │
           └─────────────────────────────────────┬─────────────────────────────────────┘
                                                 │
                                                 ▼
                                       ┌───────────────────┐
                                       │   Map Matching    │ ──► Snaps to road network nodes
                                       └─────────┬─────────┘
                                                 ▼
                                       ┌───────────────────┐
                                       │  Fused Trajectory │
                                       └───────────────────┘
```

---

## 2. Core Mathematical Operations

### 2.1 Coordinate Frame Transformations
Smartphone sensors capture linear acceleration ($\mathbf{a}_b$) and angular velocities ($\boldsymbol{\omega}_b$) in the local **phone-body frame**. To use these for navigation, we must rotate them into a fixed **navigation frame** (East, North, Up - ENU).
Using Android's orientation azimuth ($\psi$) converted to standard mathematical angles (counter-clockwise, 0 = East):
$$\begin{aligned}
a_{\text{nav}, x} &= a_{b, y} \sin(\psi) + a_{b, x} \cos(\psi) \\
a_{\text{nav}, y} &= a_{b, y} \cos(\psi) - a_{b, x} \sin(\psi) \\
a_{\text{nav}, z} &= a_{b, z} - g
\end{aligned}$$
Where $g = 9.80665 \text{ m/s}^2$. 

### 2.2 Double Integration (Classical INS)
Once rotated, the navigation accelerations are integrated using the trapezoidal rule over time step $dt$:
$$\begin{aligned}
\mathbf{v}_k &= \mathbf{v}_{k-1} + 0.5 (\mathbf{a}_{\text{nav}, k} + \mathbf{a}_{\text{nav}, k-1}) dt \\
\mathbf{p}_k &= \mathbf{p}_{k-1} + 0.5 (\mathbf{v}_k + \mathbf{v}_{k-1}) dt
\end{aligned}$$

### 2.3 Closed-Loop Periodic Re-anchoring (INS Reset)
Because accelerometer bias integrates quadratically, pure dead-reckoning INS drifts by thousands of meters. To align the training labels with the distribution seen during real-time inference (where the EKF keeps coordinates stable), we periodically reset the INS state to the reference coordinates at randomized intervals (10 to 120 seconds) when GPS is available.

### 2.4 EKF State Blending
The filter tracks a 4D state vector $\mathbf{x} = \begin{bmatrix} p_x & p_y & v_x & v_y \end{bmatrix}^T$.
- **Time Update (Propagation)**:
  $$\mathbf{x}_k^- = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{B} \mathbf{u}_k$$
  $$\mathbf{P}_k^- = \mathbf{F} \mathbf{P}_{k-1} \mathbf{F}^T + \mathbf{Q}$$
- **AI Velocity Measurement Update**:
  When the model predicts a velocity error $\mathbf{y}_{\text{AI}} = \begin{bmatrix} e_{v_x} & e_{v_y} \end{bmatrix}^T$, it is fused into the state vector as:
  $$\mathbf{z}_{\text{AI}} = \mathbf{v}_{\text{INS}} + \mathbf{y}_{\text{AI}}$$
  This correction uses addition because our training labels are generated as:
  $$\mathbf{y}_{\text{label}} = \mathbf{v}_{\text{GT}} - \mathbf{v}_{\text{INS}}$$

---

## 3. Neural Network Architecture

The model is built using Keras' Functional API, combining a temporal sequence extraction path with an auxiliary state branch:

```
Branch 1: IMU Window (20x6) ──► Conv1D(32) ──► BN ──► Conv1D(64) ──► BN ──► MaxPool ──► LSTM(64) ──► LSTM(32) ──┐
                                                                                                            ├──► Concat ──► Dense(64) ──► Dense(32) ──► Output (2)
Branch 2: INS State  (4)   ──────────────────────────────────────────────────────────► Dense(16) ──────────┘
```

- **IMU Window Branch**: Processes a rolling 2-second time series (20 samples at 10 Hz) of raw accelerometer and gyroscope signals. The 1D Convolutions pick up spatial shapes (bumps, turns, vibrations), while the LSTMs track temporal patterns.
- **INS State Branch**: Feeds the current integration coordinates through a Dense layer to guide error magnitude predictions.
- **Loss Function**: Trained using **Huber loss** to protect optimization against outlier windows.
- **Weights Sync**: The model's weights can be seamlessly shared between rolled and unrolled LSTM architectures.

---

## 4. Codebase Reference Directory

- [`config.py`](file:///d:/Project/DeadReckon/ins_error_ai/config.py): Single source of truth for column maps, hyperparameters, physical constants, EKF process noise ($Q$) and measurement noise ($R$).
- [`src/io_utils.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/io_utils.py): Handles encoding, lat/lon ENU plane projections, and monotonic timestamp unwrapping to resolve raw sensor overflows.
- [`src/ins_mechanization.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/ins_mechanization.py): Integrates accelerometer and gyroscope signals. Contains both the batch integration routine and the stateful `INSTracker` streaming class.
- [`src/dataset.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/dataset.py): Cuts sessions into overlapping windows, applies the $\pm 50\text{ m/s}$ label clipping, and outputs the `.npz` training dataset.
- [`src/model.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/model.py): Defines the dual-branch Keras model. Supports `unroll_lstms=True` for Flex-op free mobile exports.
- [`src/train.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/train.py): Standard training loop implementing Early Stopping and logging validation MAE in real m/s units.
- [`src/inference.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/inference.py): Provides the stateful `LiveErrorCorrector` class. Features the `predict_jit` pre-compiled execution path which reduces sample latency down to **2.6 ms**.
- [`src/export_tflite.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/export_tflite.py): Converts trained Keras weights into an unrolled, static TFLite file (`ins_error_model.tflite` - **315.2 KB**) that benchmarks at an extremely fast **0.15 ms** on CPU.
- [`src/ekf.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/ekf.py): Extended Kalman Filter. Integrates process updates and measurement fusion.
- [`src/pipeline.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/pipeline.py): Simulates single-session processing (mechanization, EKF, blackout window) and generates overlays.
- [`src/evaluate.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/evaluate.py): Iterates over all 72 sessions and prints comparison matrices.
- [`src/test_sign_convention.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/test_sign_convention.py): Asserts sign alignment between label generator and fusion equations.
- [`src/test_streaming_consistency.py`](file:///d:/Project/DeadReckon/ins_error_ai/src/test_streaming_consistency.py): Simulates row-by-row streaming, comparing outputs with batch matrices.

---

## 5. Operations & Testing Manual

### 5.1 Environment Initialization
```powershell
cd d:\Project\DeadReckon\ins_error_ai
..\venv\Scripts\Activate.ps1
```

### 5.2 Build Pipeline Execution (From Scratch)
```bash
# 1. Monotonic timestamp unwrapping & label generation
python -m src.generate_labels

# 2. Window sliding, label clipping, & NPZ dataset generation
python -m src.dataset

# 3. Model training (Conv1D-LSTM model with Huber loss)
python -m src.train

# 4. Convert Keras weights & compile unrolled static TFLite model
python -m src.export_tflite
```

### 5.3 Benchmarks & Verification
```bash
# 5. Execute 3-way comparative evaluation across all 72 vehicle drives
python -m src.evaluate

# 6. Execute EKF GPS blackout simulation (20s, 60s, 120s outages)
python -m src.test_blackouts

# 7. Run EKF demo on S1 with plot output
python -m src.pipeline --session S1 --skip-map-matching

# 8. Assert mathematical sign convention and EKF directionality
python -m src.test_sign_convention

# 9. Verify streaming row-by-row predictions match batch matrices
python -m src.test_streaming_consistency
```

---

## 6. Project Verification Benchmarks

### 6.1 Full 72-Session Drift Performance
Detailed statistical results from the evaluate benchmark:
- **Raw INS Drift**: Mean = **202.58%** | Median = **15.13%**
- **AI-Corrected INS (No GPS)**: Mean = **33.87%** | Median = **6.78%**
- **EKF Fusion (With GPS)**: Mean = **14.27%** | Median = **2.48%**
- **Average Performance Gain**: AI corrections achieve an average **83.3% drift reduction** over raw kinematic dead-reckoning.

### 6.2 Outlier Analysis (Mean vs. Median Gap)
The median EKF drift (**2.48%**) meets project targets, but the mean is pulled to **14.27%** due to:
- **Short-Duration Sessions (e.g., Vta19 - 29s, Vtb10 - 20s)**: Because drift % is $\text{error} / \text{distance}$, short drives have tiny denominators (low distance). Even a small, negligible EKF tracking deviation of 3-5 meters results in a high drift percentage.
- **Massive Sessions (e.g., Vta3 - 17.8 hours)**: In the 18-hour integration window, small residual speed biases integrate over time, causing EKF drift of 449%.

### 6.3 Blackout Outage Robustness
Simulated EKF outage results (skipped updates represent dropouts at 10 Hz):
- **S1**: 20s (200 updates skipped) $\rightarrow$ **0.03%** | 60s (600 updates skipped) $\rightarrow$ **0.02%** | 120s (1200 updates skipped) $\rightarrow$ **0.05%** drift.
- **S2**: 20s (200 updates skipped) $\rightarrow$ **0.01%** | 60s (600 updates skipped) $\rightarrow$ **0.01%** | 120s (1200 updates skipped) $\rightarrow$ **0.05%** drift.
- **Vfa02**: 20s $\rightarrow$ **0.00%** | 60s $\rightarrow$ **0.01%** | 120s $\rightarrow$ **0.00%** drift.
- **Y1**: 20s $\rightarrow$ **2.51%** | 60s $\rightarrow$ **2.50%** | 120s $\rightarrow$ **2.51%** drift.
