"""
seamless_controller.py
======================
Implements GNSS Quality State Machine with Hysteresis/Debouncing, Battery-Optimized
AI Gating, and Innovation Validation for Seamless GNSS ⇄ AI-IDR Navigation.
"""

from enum import Enum
import numpy as np
import config


class NavigationMode(Enum):
    GOOD = "GOOD"            # High GNSS quality; AI Model SLEEP (0% CPU/NPU load)
    DEGRADED = "DEGRADED"    # Reduced GNSS trust; AI Model WAKING / ACTIVE
    LOST = "LOST"            # Total GNSS outage / blackout; Pure AI-IDR + NHC
    RECOVERING = "RECOVERING"# Re-acquired GNSS; Smooth R deflation & innovation check


class GNSSQualityStateMachine:
    """
    4-State Machine with Debounce Counters and Dynamic Trust Scaling.
    """
    def __init__(self, 
                 n_recovery: int = config.N_RECOVERY_CYCLES,
                 n_degrade: int = config.N_DEGRADE_CYCLES,
                 n_lost: int = config.N_LOST_CYCLES,
                 good_thresh: float = config.GNSS_THRESHOLD_GOOD_M,
                 degraded_thresh: float = config.GNSS_THRESHOLD_DEGRADED_M,
                 ai_sleep_on_good: bool = config.AI_SLEEP_ON_GOOD_GNSS):
        
        self.n_recovery = n_recovery
        self.n_degrade = n_degrade
        self.n_lost = n_lost
        self.good_thresh = good_thresh
        self.degraded_thresh = degraded_thresh
        self.ai_sleep_on_good = ai_sleep_on_good

        # State tracking
        self.mode = NavigationMode.GOOD
        self.degraded_counter = 0
        self.lost_counter = 0
        self.recovery_counter = 0

        # Metrics for battery tracking
        self.total_cycles = 0
        self.ai_active_cycles = 0

    def evaluate_sample(self, gps_ok: bool, accuracy_m: float = None, sats_count: int = None) -> dict:
        """
        Evaluate raw GNSS metrics and update the state machine with debounce logic.

        Returns dict containing:
        - mode: NavigationMode
        - quality_score: float in [0.0, 1.0]
        - trust_factor: float in [0.0, 1.0] (for EKF R scaling: R = R_base / trust_factor)
        - ai_active: bool (True if AI inference should be run, False if AI can sleep)
        - skip_gnss: bool (True if EKF should skip GNSS correction)
        """
        self.total_cycles += 1

        # 1. Compute instant continuous quality score S_Q in [0.0, 1.0]
        if not gps_ok or accuracy_m is None or accuracy_m >= config.GNSS_BLACKOUT_THRESHOLD_M:
            instant_score = 0.0
        else:
            if accuracy_m <= self.good_thresh:
                # High quality: accuracy 0 to 10m -> score 1.0 down to 0.8
                instant_score = 1.0 - 0.2 * (accuracy_m / max(1.0, self.good_thresh))
            elif accuracy_m <= self.degraded_thresh:
                # Degraded quality: accuracy 10m to 25m -> score 0.8 down to 0.2
                frac = (accuracy_m - self.good_thresh) / (self.degraded_thresh - self.good_thresh)
                instant_score = 0.8 - 0.6 * frac
            else:
                # Poor quality: score < 0.2 down to 0.0
                instant_score = max(0.0, 0.2 * (1.0 - (accuracy_m - self.degraded_thresh) / 25.0))

        # Satellite check penalty if sats < MIN
        if sats_count is not None and sats_count < config.GNSS_MIN_SATS_GOOD:
            instant_score *= 0.7

        # 2. State Machine Transitions with Debounce Counters
        prev_mode = self.mode

        if instant_score >= 0.7:
            # Good signal detected
            self.degraded_counter = 0
            self.lost_counter = 0
            if prev_mode in [NavigationMode.DEGRADED, NavigationMode.LOST, NavigationMode.RECOVERING]:
                self.recovery_counter += 1
                if self.recovery_counter >= self.n_recovery:
                    self.mode = NavigationMode.GOOD
                    self.recovery_counter = 0
                else:
                    self.mode = NavigationMode.RECOVERING
            else:
                self.mode = NavigationMode.GOOD
                self.recovery_counter = 0

        elif instant_score >= 0.2:
            # Degraded signal detected
            self.recovery_counter = 0
            self.lost_counter = 0
            self.degraded_counter += 1
            if self.degraded_counter >= self.n_degrade:
                self.mode = NavigationMode.DEGRADED

        else:
            # Signal lost or blackout
            self.recovery_counter = 0
            self.degraded_counter = 0
            self.lost_counter += 1
            if self.lost_counter >= self.n_lost:
                self.mode = NavigationMode.LOST

        # 3. Derive Control Parameters (Trust Factor & AI Activation)
        if self.mode == NavigationMode.GOOD:
            trust_factor = 1.0
            skip_gnss = False
            ai_active = not self.ai_sleep_on_good
        elif self.mode == NavigationMode.RECOVERING:
            # Gradual recovery scaling over n_recovery cycles
            blend_frac = min(1.0, max(0.1, self.recovery_counter / float(self.n_recovery)))
            trust_factor = 0.2 + 0.8 * blend_frac
            skip_gnss = False
            ai_active = True
        elif self.mode == NavigationMode.DEGRADED:
            trust_factor = max(0.1, instant_score)
            skip_gnss = False
            ai_active = True
        else: # NavigationMode.LOST
            trust_factor = 0.0
            skip_gnss = True
            ai_active = True

        if ai_active:
            self.ai_active_cycles += 1

        return {
            "mode": self.mode,
            "quality_score": instant_score,
            "trust_factor": trust_factor,
            "ai_active": ai_active,
            "skip_gnss": skip_gnss,
            "recovery_counter": self.recovery_counter,
        }

    def get_battery_savings_summary(self) -> dict:
        """
        Return statistics on AI duty cycle and energy savings.
        """
        if self.total_cycles == 0:
            return {"duty_cycle_pct": 0.0, "battery_savings_pct": 100.0}

        duty_cycle = (self.ai_active_cycles / float(self.total_cycles)) * 100.0
        savings = 100.0 - duty_cycle
        return {
            "total_cycles": self.total_cycles,
            "ai_active_cycles": self.ai_active_cycles,
            "ai_sleeping_cycles": self.total_cycles - self.ai_active_cycles,
            "duty_cycle_pct": duty_cycle,
            "battery_savings_pct": savings
        }


def check_innovation_gate(innovation: np.ndarray, S: np.ndarray, gate_threshold: float = config.EKF_INNOVATION_GATE_CHI2) -> bool:
    """
    Mahalanobis distance innovation gating to reject multipath/false GNSS jumps.
    Returns True if measurement is valid (passes gate), False if anomaly/rejected.
    """
    try:
        S_inv = np.linalg.inv(S)
        mahalanobis_sq = float(innovation.T @ S_inv @ innovation)
        return mahalanobis_sq <= gate_threshold
    except np.linalg.LinAlgError:
        return False
