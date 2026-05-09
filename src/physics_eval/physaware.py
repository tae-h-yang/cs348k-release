"""Physical-awareness critic and simple repair operators for qpos plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inverse_dynamics import InverseDynamicsEvaluator
from .metrics import InverseDynamicsClipMetrics
from .validation import validate_qpos_sequence


@dataclass
class RiskReport:
    clip_name: str
    motion_type: str
    variant: str
    n_frames: int
    risk_score: float
    torque_risk: float
    root_force_risk: float
    root_torque_risk: float
    velocity_risk: float
    acceleration_risk: float
    smoothness_risk: float
    p95_torque_limit_ratio: float
    exceeded_joint_pct: float
    p95_root_force_N: float
    p95_root_torque_Nm: float
    p95_joint_vel_rad_s: float
    p95_joint_acc_rad_s2: float
    mean_joint_jerk_rad_s3: float
    recommended_action: str

    def summary(self) -> dict:
        return {
            "clip": self.clip_name,
            "type": self.motion_type,
            "variant": self.variant,
            "frames": self.n_frames,
            "risk_score": self.risk_score,
            "torque_risk": self.torque_risk,
            "root_force_risk": self.root_force_risk,
            "root_torque_risk": self.root_torque_risk,
            "velocity_risk": self.velocity_risk,
            "acceleration_risk": self.acceleration_risk,
            "smoothness_risk": self.smoothness_risk,
            "p95_torque_limit_ratio": self.p95_torque_limit_ratio,
            "exceeded_joint_pct": self.exceeded_joint_pct,
            "p95_root_force_N": self.p95_root_force_N,
            "p95_root_torque_Nm": self.p95_root_torque_Nm,
            "p95_joint_vel_rad_s": self.p95_joint_vel_rad_s,
            "p95_joint_acc_rad_s2": self.p95_joint_acc_rad_s2,
            "mean_joint_jerk_rad_s3": self.mean_joint_jerk_rad_s3,
            "recommended_action": self.recommended_action,
        }


class PhysicalAwarenessCritic:
    """Scores kinematic plans using inverse dynamics and derivative limits."""

    def __init__(self, evaluator: InverseDynamicsEvaluator | None = None):
        self.evaluator = evaluator or InverseDynamicsEvaluator()
        self.dt = self.evaluator.dt

    def score(
        self,
        qpos_seq: np.ndarray,
        clip_name: str,
        motion_type: str,
        variant: str = "original",
    ) -> tuple[RiskReport, InverseDynamicsClipMetrics]:
        qpos_seq = validate_qpos_sequence(qpos_seq, name=clip_name, min_frames=2)
        inv = self.evaluator.evaluate_clip(qpos_seq, clip_name=clip_name, motion_type=motion_type)

        qvel = self.evaluator.compute_qvel(qpos_seq)
        qacc = self.evaluator.compute_qacc(qvel)
        qjerk = np.diff(qacc[:, 6:], axis=0) / self.dt if len(qacc) > 1 else np.zeros((1, 29))

        p95_vel = float(np.percentile(np.abs(qvel[:, 6:]), 95))
        p95_acc = float(np.percentile(np.abs(qacc[:, 6:]), 95))
        mean_jerk = float(np.mean(np.abs(qjerk)))

        torque_risk = _soft_excess(inv.p95_torque_limit_ratio, 1.0)
        root_force_risk = _soft_excess(inv.p95_root_force_N, 10000.0)
        root_torque_risk = _soft_excess(inv.p95_root_torque_Nm, 2000.0)
        velocity_risk = _soft_excess(p95_vel, 10.0)
        acceleration_risk = _soft_excess(p95_acc, 120.0)
        smoothness_risk = _soft_excess(mean_jerk, 1200.0)

        risk_score = float(
            35.0 * torque_risk
            + 20.0 * root_force_risk
            + 15.0 * root_torque_risk
            + 10.0 * velocity_risk
            + 10.0 * acceleration_risk
            + 10.0 * smoothness_risk
        )

        action = "accept"
        if risk_score > 80 or inv.p95_torque_limit_ratio > 5.0:
            action = "reject_or_regenerate"
        elif risk_score > 25 or inv.p95_torque_limit_ratio > 1.25:
            action = "repair_or_rerank"

        report = RiskReport(
            clip_name=clip_name,
            motion_type=motion_type,
            variant=variant,
            n_frames=len(qpos_seq),
            risk_score=risk_score,
            torque_risk=torque_risk,
            root_force_risk=root_force_risk,
            root_torque_risk=root_torque_risk,
            velocity_risk=velocity_risk,
            acceleration_risk=acceleration_risk,
            smoothness_risk=smoothness_risk,
            p95_torque_limit_ratio=inv.p95_torque_limit_ratio,
            exceeded_joint_pct=100.0 * inv.mean_exceeded_joint_fraction,
            p95_root_force_N=inv.p95_root_force_N,
            p95_root_torque_Nm=inv.p95_root_torque_Nm,
            p95_joint_vel_rad_s=p95_vel,
            p95_joint_acc_rad_s2=p95_acc,
            mean_joint_jerk_rad_s3=mean_jerk,
            recommended_action=action,
        )
        return report, inv


def _soft_excess(value: float, threshold: float) -> float:
    """Map threshold excess to a bounded-ish positive risk component."""
    if value <= threshold:
        return 0.0
    return float(np.log1p((value - threshold) / threshold))


def smooth_qpos(qpos_seq: np.ndarray, passes: int = 2, joint_weight: float = 0.25) -> np.ndarray:
    """Low-pass filter root translation and joints while preserving endpoints."""
    out = validate_qpos_sequence(qpos_seq, min_frames=3)
    for _ in range(passes):
        prev = out.copy()
        out[1:-1, :3] = (1 - joint_weight) * prev[1:-1, :3] + 0.5 * joint_weight * (
            prev[:-2, :3] + prev[2:, :3]
        )
        out[1:-1, 7:] = (1 - joint_weight) * prev[1:-1, 7:] + 0.5 * joint_weight * (
            prev[:-2, 7:] + prev[2:, 7:]
        )
    return out


def time_scale_qpos(qpos_seq: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """Resample a qpos sequence to slow the motion down by `scale`."""
    if scale <= 1.0:
        raise ValueError("time scale must be > 1.0")
    qpos_seq = validate_qpos_sequence(qpos_seq, min_frames=2)
    old_t = np.arange(len(qpos_seq), dtype=np.float64)
    new_len = max(len(qpos_seq) + 1, int(np.ceil(len(qpos_seq) * scale)))
    new_t = np.linspace(0, len(qpos_seq) - 1, new_len)
    out = np.zeros((new_len, qpos_seq.shape[1]), dtype=np.float64)
    for j in range(qpos_seq.shape[1]):
        out[:, j] = np.interp(new_t, old_t, qpos_seq[:, j])
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1)[:, None]
    return out


def repair_candidates(qpos_seq: np.ndarray) -> dict[str, np.ndarray]:
    """Generate deterministic test-time repair candidates."""
    return {
        "original": validate_qpos_sequence(qpos_seq, min_frames=2),
        "smooth": smooth_qpos(qpos_seq, passes=3, joint_weight=0.35),
        "slow_1p5x": time_scale_qpos(qpos_seq, scale=1.5),
        "slow_2x": time_scale_qpos(qpos_seq, scale=2.0),
        "smooth_slow_1p5x": time_scale_qpos(smooth_qpos(qpos_seq, passes=3, joint_weight=0.35), scale=1.5),
    }
