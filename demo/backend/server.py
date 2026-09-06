"""
server.py
=========
FastAPI backend for the DeadReckon end-to-end demo.

Wraps the existing ins_error_ai pipeline and exposes it via REST endpoints.
Also serves the frontend static files directly — no separate web server needed.

Start with:
    cd demo/backend
    python -m uvicorn server:app --reload --port 8000
"""

import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — make ins_error_ai importable
# ---------------------------------------------------------------------------
DEMO_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.dirname(DEMO_BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(DEMO_DIR)
INS_ERROR_AI_DIR = os.path.join(PROJECT_ROOT, "ins_error_ai")

# Add ins_error_ai to path so its internal imports work
sys.path.insert(0, INS_ERROR_AI_DIR)
sys.path.insert(0, os.path.join(INS_ERROR_AI_DIR, "src"))

import config
from src.discover_sessions import discover_all_sessions
from src.io_utils import load_s_file, load_v_file, latlon_to_enu, speed_heading_to_velocity
from src.ins_mechanization import run_ins_mechanization, INSTracker
from src.pipeline import (
    load_ai_corrector, run_ai_correction_batch,
    compute_ai_corrected_trajectory, prepare_gnss_data,
    simulate_blackout, compute_drift_pct,
)
from src.ekf import run_ekf_fusion

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DeadReckon Navigation Demo",
    description="End-to-end INS/AI/EKF navigation demo with live visualization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache the AI corrector (expensive to load each time)
_ai_corrector = None

def get_ai_corrector():
    global _ai_corrector
    if _ai_corrector is None:
        _ai_corrector = load_ai_corrector()
    return _ai_corrector


def ndarray_to_list(obj):
    """Recursively convert numpy arrays/scalars to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ndarray_to_list(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def list_sessions():
    """List all available IO-VNBD sessions."""
    try:
        sessions = discover_all_sessions()
        return {
            "sessions": [
                {
                    "name": s["name"],
                    "category": s["category"],
                    "s_path": os.path.basename(s["s_path"]),
                    "v_path": os.path.basename(s["v_path"]),
                }
                for s in sessions
            ],
            "count": len(sessions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PipelineRequest(BaseModel):
    blackout_duration: Optional[float] = None
    blackout_start: Optional[float] = None
    downsample: Optional[int] = 10  # Downsample factor for trajectory data sent to frontend


@app.post("/api/run/{session_name}")
async def run_session_pipeline(session_name: str, req: PipelineRequest = None):
    """
    Run the full INS → AI → EKF pipeline on a session.
    Returns trajectories, metrics, and plot paths.
    """
    if req is None:
        req = PipelineRequest()

    try:
        sessions = discover_all_sessions()
        session = next((s for s in sessions if s["name"] == session_name), None)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_name}' not found")

        t_start = time.time()

        # Step 1: Load raw data
        raw_imu = load_s_file(session["s_path"])
        raw_gt = load_v_file(session["v_path"])

        # Compute _t_sec
        t = raw_imu["time"].to_numpy(dtype=float)
        if t.max() > 1e5:
            t = t / 1000.0
        t = t - t[0]

        temp_imu = raw_imu.copy()
        temp_imu["_t_sec"] = t

        # Step 2: Prepare ground truth + GNSS
        data = prepare_gnss_data(temp_imu, raw_gt)
        time_s = temp_imu["_t_sec"].to_numpy()
        n = len(time_s)

        # Session seed for deterministic INS
        import hashlib
        session_seed = int(hashlib.md5(session["name"].encode()).hexdigest(), 16) % 1000000

        # Step 3: Run INS mechanization (no ground-truth reset)
        ins_df = run_ins_mechanization(raw_imu, ref_df=None, seed=session_seed)
        duration = ins_df["_t_sec"].iloc[-1]

        # Step 4: AI correction
        corrector = get_ai_corrector()
        ai_corrections = run_ai_correction_batch(ins_df, corrector)

        # Step 5: AI-corrected trajectory (no EKF)
        ai_traj = compute_ai_corrected_trajectory(ins_df, ai_corrections)

        # Step 6: EKF fusion (with GNSS)
        ekf_result = run_ekf_fusion(
            ins_df, ai_corrections=ai_corrections,
            gnss_available=data["gps_available"],
            gnss_pos=data["gps_pos"],
            gnss_accuracy=data["gps_accuracy"],
        )

        # Step 7: EKF with blackout
        bo_duration = req.blackout_duration or config.BLACKOUT_DURATION_S
        blackout_gps, blackout_window = simulate_blackout(
            data["gps_available"], time_s,
            start_time=req.blackout_start, duration_s=bo_duration,
        )

        ins_df_bo = run_ins_mechanization(raw_imu, ref_df=None, seed=session_seed)
        ai_corrections_bo = run_ai_correction_batch(ins_df_bo, corrector)

        ekf_blackout = run_ekf_fusion(
            ins_df_bo, ai_corrections=ai_corrections_bo,
            gnss_available=blackout_gps,
            gnss_pos=data["gps_pos"],
            gnss_accuracy=data["gps_accuracy"],
        )

        elapsed = time.time() - t_start

        # --- Compute metrics ---
        gt_pos_x = data["true_pos_x"]
        gt_pos_y = data["true_pos_y"]
        ins_pos_x = ins_df["ins_pos_x"].to_numpy()
        ins_pos_y = ins_df["ins_pos_y"].to_numpy()

        ins_drift = compute_drift_pct(ins_pos_x, ins_pos_y, gt_pos_x, gt_pos_y)

        ai_drift = float("inf")
        if ai_traj is not None:
            ai_drift = compute_drift_pct(
                ai_traj["pos_x"], ai_traj["pos_y"], gt_pos_x, gt_pos_y)

        ekf_drift = compute_drift_pct(
            ekf_result["pos_x"], ekf_result["pos_y"], gt_pos_x, gt_pos_y)

        ekf_bo_drift = compute_drift_pct(
            ekf_blackout["pos_x"], ekf_blackout["pos_y"], gt_pos_x, gt_pos_y)

        # RMSE
        gt_pos = np.column_stack([gt_pos_x, gt_pos_y])
        ins_pos = np.column_stack([ins_pos_x, ins_pos_y])
        ekf_pos = np.column_stack([ekf_result["pos_x"], ekf_result["pos_y"]])

        ins_pos_rmse = float(np.sqrt(np.mean(np.sum((ins_pos - gt_pos)**2, axis=1))))
        ekf_pos_rmse = float(np.sqrt(np.mean(np.sum((ekf_pos - gt_pos)**2, axis=1))))

        gt_vel = np.column_stack([data["true_vel_x"], data["true_vel_y"]])
        ins_vel = np.column_stack([ins_df["ins_vel_x"].to_numpy(), ins_df["ins_vel_y"].to_numpy()])
        ekf_vel = np.column_stack([ekf_result["vel_x"], ekf_result["vel_y"]])

        ins_vel_rmse = float(np.sqrt(np.mean(np.sum((ins_vel - gt_vel)**2, axis=1))))
        ekf_vel_rmse = float(np.sqrt(np.mean(np.sum((ekf_vel - gt_vel)**2, axis=1))))

        ins_final_err = float(np.linalg.norm(ins_pos[-1] - gt_pos[-1]))
        ekf_final_err = float(np.linalg.norm(ekf_pos[-1] - gt_pos[-1]))

        # Total distance
        dx = np.diff(gt_pos_x)
        dy = np.diff(gt_pos_y)
        total_dist = float(np.sum(np.sqrt(dx**2 + dy**2)))

        # --- Downsample trajectories for frontend ---
        ds = max(1, req.downsample or 10)
        idx = np.arange(0, n, ds)
        # Always include the last point
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)

        trajectories = {
            "time_s": time_s[idx].tolist(),
            "ground_truth": {
                "pos_x": gt_pos_x[idx].tolist(),
                "pos_y": gt_pos_y[idx].tolist(),
            },
            "ins_only": {
                "pos_x": ins_pos_x[idx].tolist(),
                "pos_y": ins_pos_y[idx].tolist(),
            },
            "ekf_full": {
                "pos_x": ekf_result["pos_x"][idx].tolist(),
                "pos_y": ekf_result["pos_y"][idx].tolist(),
            },
            "ekf_blackout": {
                "pos_x": ekf_blackout["pos_x"][idx].tolist(),
                "pos_y": ekf_blackout["pos_y"][idx].tolist(),
            },
        }

        if ai_traj is not None:
            trajectories["ai_corrected"] = {
                "pos_x": ai_traj["pos_x"][idx].tolist(),
                "pos_y": ai_traj["pos_y"][idx].tolist(),
            }

        # Uncertainty for blackout visualization
        trajectories["uncertainty"] = {
            "pos_unc": ekf_blackout["pos_unc"][idx].tolist(),
            "vel_unc": ekf_blackout["vel_unc"][idx].tolist(),
        }

        metrics = {
            "session": session_name,
            "duration_s": float(duration),
            "duration_str": f"~{duration/3600:.1f} Hours / {total_dist/1000:.2f} km driven",
            "total_distance_m": total_dist,
            "n_samples": int(n),
            "computation_time_s": round(elapsed, 2),
            "drift_pct": {
                "ins": round(ins_drift, 4),
                "ai": round(ai_drift, 4) if ai_traj is not None else None,
                "ekf_full": round(ekf_drift, 4),
                "ekf_blackout": round(ekf_bo_drift, 4),
            },
            "pos_rmse_m": {
                "ins": round(ins_pos_rmse, 2),
                "ekf_full": round(ekf_pos_rmse, 2),
            },
            "vel_rmse_ms": {
                "ins": round(ins_vel_rmse, 3),
                "ekf_full": round(ekf_vel_rmse, 3),
            },
            "final_error_m": {
                "ins": round(ins_final_err, 2),
                "ekf_full": round(ekf_final_err, 2),
            },
            "blackout": {
                "duration_s": float(bo_duration),
                "window_start": float(blackout_window[0]),
                "window_end": float(blackout_window[1]),
            },
            "battery_summary": ndarray_to_list(ekf_result.get("battery_summary", {})),
        }

        return {"trajectories": trajectories, "metrics": metrics}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/plots/{filename}")
async def get_plot(filename: str):
    """Serve generated plot images."""
    path = os.path.join(config.PLOT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Plot '{filename}' not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/system-info")
async def system_info():
    """Return model architecture info and config parameters."""
    model_info = {
        "architecture": "CNN-LSTM (Conv1D → LSTM → Dense)",
        "input_imu_window": config.WINDOW_SIZE,
        "input_imu_features": config.IMU_FEATURES,
        "input_state_features": config.INS_STATE_FEATURES,
        "output": config.TARGET_FEATURES,
        "sample_rate_hz": config.SAMPLE_RATE_HZ,
        "window_seconds": config.WINDOW_SECONDS,
    }

    ekf_info = {
        "state_vector": ["pos_x", "pos_y", "vel_x", "vel_y"],
        "q_pos_std": config.EKF_Q_POS_STD,
        "q_vel_std": config.EKF_Q_VEL_STD,
        "r_gnss_pos_std": config.EKF_R_GNSS_POS_STD,
        "r_ai_vel_std": config.EKF_R_AI_VEL_STD,
        "gnss_blackout_threshold_m": config.GNSS_BLACKOUT_THRESHOLD_M,
        "innovation_gate_chi2": config.EKF_INNOVATION_GATE_CHI2,
    }

    model_exists = os.path.exists(os.path.join(config.MODEL_DIR, "ins_error_model_best.keras"))
    tflite_exists = os.path.exists(os.path.join(config.MODEL_DIR, "ins_error_model.tflite"))

    return {
        "model": model_info,
        "ekf": ekf_info,
        "sign_convention": "err_vel = true_vel - ins_vel; corrected = ins + predicted (ADDITION)",
        "model_loaded": model_exists,
        "tflite_available": tflite_exists,
        "dataset_root": config.DATASET_ROOT,
        "dataset_exists": os.path.isdir(config.DATASET_ROOT),
    }


class SimulateRequest(BaseModel):
    session_name: Optional[str] = None
    n_samples: Optional[int] = 500  # Number of samples to simulate
    downsample: Optional[int] = 5


@app.post("/api/simulate-phone")
async def simulate_phone(req: SimulateRequest = None):
    """
    Simulate real-time phone IMU streaming.
    Feeds S-file data through INSTracker + LiveErrorCorrector sample-by-sample
    to demonstrate real-time capability.
    """
    if req is None:
        req = SimulateRequest()

    try:
        sessions = discover_all_sessions()
        if not sessions:
            raise HTTPException(status_code=404, detail="No sessions found")

        if req.session_name:
            session = next((s for s in sessions if s["name"] == req.session_name), None)
            if not session:
                raise HTTPException(status_code=404, detail=f"Session '{req.session_name}' not found")
        else:
            session = sessions[0]

        raw_imu = load_s_file(session["s_path"])
        raw_gt = load_v_file(session["v_path"])

        # Time
        t = raw_imu["time"].to_numpy(dtype=float)
        if t.max() > 1e5:
            t = t / 1000.0
        t = t - t[0]

        temp_imu = raw_imu.copy()
        temp_imu["_t_sec"] = t

        # Ground truth for comparison
        data = prepare_gnss_data(temp_imu, raw_gt)

        n_sim = min(req.n_samples or 500, len(raw_imu))

        # Create INS tracker
        tracker = INSTracker()

        # Try to get AI corrector
        corrector = get_ai_corrector()

        # Simulate sample-by-sample
        ins_trajectory = []
        ai_trajectory = []
        timestamps = []

        for i in range(n_sim):
            dt = float(t[i] - t[i-1]) if i > 0 else 0.1

            acc_x = float(raw_imu["acc_x"].iloc[i])
            acc_y = float(raw_imu["acc_y"].iloc[i])
            acc_z = float(raw_imu["acc_z"].iloc[i])
            yaw = float(raw_imu["orient_yaw"].iloc[i]) if "orient_yaw" in raw_imu.columns else 0.0

            vx, vy, px, py = tracker.step(acc_x, acc_y, acc_z, yaw, dt)

            ins_trajectory.append({"px": px, "py": py, "vx": vx, "vy": vy})
            timestamps.append(float(t[i]))

            # AI correction (if corrector available and buffer full)
            if corrector is not None:
                imu_sample = np.array([
                    acc_x, acc_y, acc_z,
                    float(raw_imu["gyro_yaw"].iloc[i]) if "gyro_yaw" in raw_imu.columns else 0.0,
                    float(raw_imu["gyro_pitch"].iloc[i]) if "gyro_pitch" in raw_imu.columns else 0.0,
                    float(raw_imu["gyro_roll"].iloc[i]) if "gyro_roll" in raw_imu.columns else 0.0,
                ], dtype=np.float32)
                corrector.push_sample(imu_sample)

                if corrector.ready():
                    correction = corrector.correct(np.array([vx, vy], dtype=np.float32))
                    ai_vx = vx + float(correction[0])
                    ai_vy = vy + float(correction[1])
                    ai_trajectory.append({"vx": ai_vx, "vy": ai_vy})
                else:
                    ai_trajectory.append({"vx": vx, "vy": vy})
            else:
                ai_trajectory.append({"vx": vx, "vy": vy})

        # Downsample for transfer
        ds = max(1, req.downsample or 5)
        idx = list(range(0, n_sim, ds))
        if idx[-1] != n_sim - 1:
            idx.append(n_sim - 1)

        return {
            "session": session["name"],
            "n_samples": n_sim,
            "timestamps": [timestamps[i] for i in idx],
            "ins_trajectory": [ins_trajectory[i] for i in idx],
            "ai_trajectory": [ai_trajectory[i] for i in idx],
            "ground_truth": {
                "pos_x": data["true_pos_x"][idx].tolist(),
                "pos_y": data["true_pos_y"][idx].tolist(),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(DEMO_DIR, "frontend")


@app.get("/")
async def serve_index():
    """Serve the main dashboard page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse(
        {"error": "Frontend not found. Place index.html in demo/frontend/"},
        status_code=404,
    )


# Mount static files for CSS/JS
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    print(f"Starting DeadReckon Demo Backend...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"ML models: {config.MODEL_DIR}")
    print(f"Frontend: {FRONTEND_DIR}")
    print(f"Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
