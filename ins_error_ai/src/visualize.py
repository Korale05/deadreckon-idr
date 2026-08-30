"""
visualize.py
============
Plotting utilities for visualizing and comparing trajectories, errors, and
the effect of AI correction and EKF fusion.

Produces publication-quality plots saved to outputs/plots/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def set_style():
    """Set a clean plot style."""
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.facecolor": "#f8f9fa",
        "figure.facecolor": "white",
    })


def plot_trajectory_comparison(time_s, ground_truth, ins_only, ai_corrected=None,
                                ekf_fused=None, title="Trajectory Comparison",
                                save_name=None, blackout_window=None):
    """
    Plot 2D trajectories from different methods overlaid.

    Parameters
    ----------
    time_s : np.ndarray
    ground_truth : dict with 'pos_x', 'pos_y'
    ins_only : dict with 'pos_x', 'pos_y'
    ai_corrected : dict with 'pos_x', 'pos_y', optional
    ekf_fused : dict with 'pos_x', 'pos_y', optional
    blackout_window : tuple (start_s, end_s), optional
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- 2D trajectory (bird's eye) ---
    ax = axes[0]
    ax.plot(ground_truth["pos_x"], ground_truth["pos_y"],
            "k-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax.plot(ins_only["pos_x"], ins_only["pos_y"],
            "r--", linewidth=1.5, label="INS Only (drifting)", alpha=0.7)
    if ai_corrected is not None:
        ax.plot(ai_corrected["pos_x"], ai_corrected["pos_y"],
                "b-.", linewidth=1.5, label="INS + AI Correction", alpha=0.7)
    if ekf_fused is not None:
        ax.plot(ekf_fused["pos_x"], ekf_fused["pos_y"],
                "g-", linewidth=1.5, label="EKF Fused", alpha=0.9)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("2D Trajectory")
    ax.legend()
    ax.set_aspect("equal")

    # --- Position error over time ---
    ax = axes[1]
    gt_pos = np.column_stack([ground_truth["pos_x"], ground_truth["pos_y"]])

    ins_err = np.linalg.norm(
        np.column_stack([ins_only["pos_x"], ins_only["pos_y"]]) - gt_pos, axis=1)
    ax.plot(time_s, ins_err, "r-", label="INS Only", alpha=0.7)

    if ai_corrected is not None:
        ai_err = np.linalg.norm(
            np.column_stack([ai_corrected["pos_x"], ai_corrected["pos_y"]]) - gt_pos, axis=1)
        ax.plot(time_s, ai_err, "b-", label="INS + AI", alpha=0.7)

    if ekf_fused is not None:
        ekf_err = np.linalg.norm(
            np.column_stack([ekf_fused["pos_x"], ekf_fused["pos_y"]]) - gt_pos, axis=1)
        ax.plot(time_s, ekf_err, "g-", label="EKF Fused", alpha=0.9)

    if blackout_window is not None:
        ax.axvspan(blackout_window[0], blackout_window[1],
                   alpha=0.2, color="orange", label="GNSS Blackout")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Error Over Time")
    ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        path = os.path.join(config.PLOT_DIR, save_name)
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved plot -> {path}")
    plt.close(fig)
    return fig


def plot_velocity_comparison(time_s, ground_truth, ins_only, ai_corrected=None,
                              ekf_fused=None, title="Velocity Comparison",
                              save_name=None):
    """
    Plot velocity components and speed over time.
    """
    set_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # --- Velocity X (East) ---
    ax = axes[0]
    ax.plot(time_s, ground_truth["vel_x"], "k-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax.plot(time_s, ins_only["vel_x"], "r--", linewidth=1, label="INS Only", alpha=0.6)
    if ai_corrected is not None:
        ax.plot(time_s, ai_corrected["vel_x"], "b-.", linewidth=1, label="INS + AI", alpha=0.7)
    if ekf_fused is not None:
        ax.plot(time_s, ekf_fused["vel_x"], "g-", linewidth=1, label="EKF Fused", alpha=0.8)
    ax.set_ylabel("Velocity East (m/s)")
    ax.set_title("Velocity X (East)")
    ax.legend()

    # --- Velocity Y (North) ---
    ax = axes[1]
    ax.plot(time_s, ground_truth["vel_y"], "k-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax.plot(time_s, ins_only["vel_y"], "r--", linewidth=1, label="INS Only", alpha=0.6)
    if ai_corrected is not None:
        ax.plot(time_s, ai_corrected["vel_y"], "b-.", linewidth=1, label="INS + AI", alpha=0.7)
    if ekf_fused is not None:
        ax.plot(time_s, ekf_fused["vel_y"], "g-", linewidth=1, label="EKF Fused", alpha=0.8)
    ax.set_ylabel("Velocity North (m/s)")
    ax.set_title("Velocity Y (North)")
    ax.legend()

    # --- Speed magnitude ---
    ax = axes[2]
    gt_speed = np.sqrt(ground_truth["vel_x"]**2 + ground_truth["vel_y"]**2) * 3.6
    ins_speed = np.sqrt(ins_only["vel_x"]**2 + ins_only["vel_y"]**2) * 3.6
    ax.plot(time_s, gt_speed, "k-", linewidth=2, label="Ground Truth", alpha=0.8)
    ax.plot(time_s, ins_speed, "r--", linewidth=1, label="INS Only", alpha=0.6)
    if ai_corrected is not None:
        ai_speed = np.sqrt(ai_corrected["vel_x"]**2 + ai_corrected["vel_y"]**2) * 3.6
        ax.plot(time_s, ai_speed, "b-.", linewidth=1, label="INS + AI", alpha=0.7)
    if ekf_fused is not None:
        ekf_speed = np.sqrt(ekf_fused["vel_x"]**2 + ekf_fused["vel_y"]**2) * 3.6
        ax.plot(time_s, ekf_speed, "g-", linewidth=1, label="EKF Fused", alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed Magnitude")
    ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        path = os.path.join(config.PLOT_DIR, save_name)
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved plot -> {path}")
    plt.close(fig)
    return fig


def plot_error_summary(time_s, ins_vel_err, ai_vel_err=None,
                       title="Velocity Error Summary", save_name=None):
    """
    Plot velocity error RMSE comparison as a bar chart + time series.
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Time series of velocity error magnitude ---
    ax = axes[0]
    ins_err_mag = np.sqrt(ins_vel_err[:, 0]**2 + ins_vel_err[:, 1]**2)
    ax.plot(time_s, ins_err_mag, "r-", label="INS Error", alpha=0.7)
    if ai_vel_err is not None:
        ai_err_mag = np.sqrt(ai_vel_err[:, 0]**2 + ai_vel_err[:, 1]**2)
        ax.plot(time_s, ai_err_mag, "b-", label="Residual after AI", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity Error Magnitude (m/s)")
    ax.set_title("Error Over Time")
    ax.legend()

    # --- Bar chart of RMSE ---
    ax = axes[1]
    ins_rmse = np.sqrt(np.mean(ins_err_mag**2))
    labels = ["INS Only"]
    values = [ins_rmse]
    colors = ["#e74c3c"]

    if ai_vel_err is not None:
        ai_rmse = np.sqrt(np.mean(ai_err_mag**2))
        improvement = (1 - ai_rmse / (ins_rmse + 1e-9)) * 100
        labels.append(f"After AI\n({improvement:.1f}% better)")
        values.append(ai_rmse)
        colors.append("#3498db")

    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Velocity RMSE (m/s)")
    ax.set_title("RMSE Comparison")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        path = os.path.join(config.PLOT_DIR, save_name)
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved plot -> {path}")
    plt.close(fig)
    return fig


def plot_ekf_uncertainty(time_s, pos_unc, vel_unc, blackout_window=None,
                          title="EKF Uncertainty", save_name=None):
    """Plot how the EKF's uncertainty evolves, especially during GNSS blackout."""
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.plot(time_s, pos_unc, "g-", linewidth=1.5)
    ax.set_ylabel("Position 1σ (m)")
    ax.set_title("Position Uncertainty")
    if blackout_window:
        ax.axvspan(blackout_window[0], blackout_window[1],
                   alpha=0.2, color="orange", label="GNSS Blackout")
        ax.legend()

    ax = axes[1]
    ax.plot(time_s, vel_unc, "b-", linewidth=1.5)
    ax.set_ylabel("Velocity 1σ (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Velocity Uncertainty")
    if blackout_window:
        ax.axvspan(blackout_window[0], blackout_window[1],
                   alpha=0.2, color="orange", label="GNSS Blackout")
        ax.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_name:
        path = os.path.join(config.PLOT_DIR, save_name)
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved plot -> {path}")
    plt.close(fig)
    return fig
