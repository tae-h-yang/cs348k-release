"""
Physics plausibility metrics for kinematic-to-dynamic gap analysis.

Each metric is computed per-frame and aggregated over a clip.
All functions assume MuJoCo coordinate convention: Z-up, X-forward.
"""

import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import List, Optional


# G1-specific thresholds
FALL_HEIGHT_THRESHOLD = 0.45   # root Z below this → fall (nominal standing ~0.793m)
N_JOINTS = 29


@dataclass
class FrameMetrics:
    tracking_rmse: float        # RMSE of 29 joint angles vs kinematic target
    per_joint_error: np.ndarray # (29,) absolute joint angle errors (rad)
    root_pos_error: float       # Euclidean distance of root pos vs kinematic target
    root_height: float          # Absolute root Z (to detect falls)
    fell: bool                  # True if root dropped below FALL_HEIGHT_THRESHOLD
    n_joint_limit_violations: int   # joints outside their MuJoCo range
    foot_penetration_left: float    # left foot ground penetration depth (m)
    foot_penetration_right: float   # right foot ground penetration depth (m)
    mechanical_power: float     # sum |tau_i * dq_i| across all joints (W)
    contact_forces: np.ndarray  # (ncon,) normal contact forces (N)


@dataclass
class ClipMetrics:
    clip_name: str
    motion_type: str
    mode: str                   # "physics" or "kinematic"
    n_frames: int
    time_to_fall: Optional[int]     # frame index of first fall, None if no fall
    fell: bool
    mean_tracking_rmse: float
    mean_per_joint_error: np.ndarray    # (29,) mean abs error per joint
    mean_root_pos_error: float
    mean_mechanical_power: float
    max_joint_limit_violations: int
    mean_foot_penetration: float
    frame_metrics: List[FrameMetrics] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "clip":             self.clip_name,
            "type":             self.motion_type,
            "mode":             self.mode,
            "frames":           self.n_frames,
            "fell":             self.fell,
            "time_to_fall":     self.time_to_fall,
            "tracking_rmse":    self.mean_tracking_rmse,
            "root_pos_error":   self.mean_root_pos_error,
            "mech_power_W":     self.mean_mechanical_power,
            "max_joint_viol":   self.max_joint_limit_violations,
            "foot_penetration_m": self.mean_foot_penetration,
        }


@dataclass
class InverseDynamicsFrameMetrics:
    required_torques: np.ndarray        # (29,) inverse-dynamics joint torques
    torque_limit_ratio: np.ndarray      # (29,) |tau| / actuator limit
    root_wrench: np.ndarray             # (6,) generalized force on free root
    root_force_norm: float              # ||root translational generalized force||
    root_torque_norm: float             # ||root rotational generalized force||
    max_torque_limit_ratio: float
    n_torque_limit_exceeded: int


@dataclass
class InverseDynamicsClipMetrics:
    clip_name: str
    motion_type: str
    mode: str
    n_frames: int
    mean_torque_limit_ratio: float
    p95_torque_limit_ratio: float
    max_torque_limit_ratio: float
    mean_exceeded_joint_fraction: float
    mean_root_force_N: float
    p95_root_force_N: float
    mean_root_torque_Nm: float
    p95_root_torque_Nm: float
    mean_per_joint_ratio: np.ndarray
    p95_per_joint_ratio: np.ndarray
    frame_metrics: List[InverseDynamicsFrameMetrics] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "clip": self.clip_name,
            "type": self.motion_type,
            "mode": self.mode,
            "frames": self.n_frames,
            "mean_torque_limit_ratio": self.mean_torque_limit_ratio,
            "p95_torque_limit_ratio": self.p95_torque_limit_ratio,
            "max_torque_limit_ratio": self.max_torque_limit_ratio,
            "exceeded_joint_pct": 100.0 * self.mean_exceeded_joint_fraction,
            "mean_root_force_N": self.mean_root_force_N,
            "p95_root_force_N": self.p95_root_force_N,
            "mean_root_torque_Nm": self.mean_root_torque_Nm,
            "p95_root_torque_Nm": self.p95_root_torque_Nm,
        }


def compute_frame_metrics(
    model,
    data,
    q_target: np.ndarray,
    torques: np.ndarray,
    joint_ranges: np.ndarray,
    foot_geom_ids: tuple,
) -> FrameMetrics:
    """
    Args:
        model:         mujoco.MjModel
        data:          mujoco.MjData (after mj_step or mj_forward)
        q_target:      (29,) kinematic target joint angles for this frame
        torques:       (29,) applied torques (zeros for kinematic replay)
        joint_ranges:  (29, 2) joint limits [[lo, hi], ...]
        foot_geom_ids: (left_ids, right_ids) collision geoms under each foot
    """
    q = data.qpos[7:]
    dq = data.qvel[6:]

    per_joint_error = np.abs(q - q_target)
    tracking_rmse = float(np.sqrt(np.mean(per_joint_error ** 2)))
    root_height = float(data.qpos[2])

    fell = root_height < FALL_HEIGHT_THRESHOLD

    violations = int(np.sum((q < joint_ranges[:, 0]) | (q > joint_ranges[:, 1])))

    foot_pen_left = _foot_penetration(model, data, foot_geom_ids[0])
    foot_pen_right = _foot_penetration(model, data, foot_geom_ids[1])

    mech_power = float(np.sum(np.abs(torques * dq)))

    # Normal contact forces (N): mj_contactForce wrench[0] is normal component
    contact_forces = np.zeros(data.ncon, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, wrench)
        contact_forces[i] = wrench[0]

    return FrameMetrics(
        tracking_rmse=tracking_rmse,
        per_joint_error=per_joint_error,
        root_pos_error=0.0,   # filled in by simulator (needs full qpos target)
        root_height=root_height,
        fell=fell,
        n_joint_limit_violations=violations,
        foot_penetration_left=foot_pen_left,
        foot_penetration_right=foot_pen_right,
        mechanical_power=mech_power,
        contact_forces=contact_forces,
    )


def _foot_penetration(model, data, geom_ids: np.ndarray) -> float:
    """Return positive ground penetration depth for the lowest foot collision geom."""
    if len(geom_ids) == 0:
        return 0.0
    lowest = float("inf")
    for gid in geom_ids:
        z = float(data.geom_xpos[gid, 2])
        # Foot collision geoms are spheres in the G1 asset. For other geom
        # types this is still a conservative lower-bound proxy.
        z_bottom = z - float(model.geom_size[gid, 0])
        lowest = min(lowest, z_bottom)
    return max(0.0, -lowest)


def aggregate_clip_metrics(clip_name: str, motion_type: str, mode: str,
                           frame_list: List[FrameMetrics]) -> ClipMetrics:
    n = len(frame_list)
    fell_frames = [i for i, f in enumerate(frame_list) if f.fell]
    fell = len(fell_frames) > 0
    time_to_fall = fell_frames[0] if fell else None

    # Average over frames before or including the first fall. Including the
    # fall frame makes frame-0 failures deterministic instead of averaging over
    # padded post-fall copies.
    active = frame_list[:time_to_fall + 1] if fell and time_to_fall is not None else frame_list

    valid = [f for f in active if np.isfinite(f.tracking_rmse)]
    if not valid:
        valid = active

    mean_tracking   = float(np.mean([f.tracking_rmse for f in valid]))
    mean_root_err   = float(np.mean([f.root_pos_error for f in valid]))
    mean_power      = float(np.mean([f.mechanical_power for f in valid]))
    max_violations  = int(max(f.n_joint_limit_violations for f in valid))
    mean_per_joint  = np.mean(np.stack([f.per_joint_error for f in valid]), axis=0)

    foot_pen_per_frame = [
        (f.foot_penetration_left + f.foot_penetration_right) / 2.0
        for f in valid
    ]
    mean_foot_pen = float(np.mean(foot_pen_per_frame))

    return ClipMetrics(
        clip_name=clip_name,
        motion_type=motion_type,
        mode=mode,
        n_frames=n,
        time_to_fall=time_to_fall,
        fell=fell,
        mean_tracking_rmse=mean_tracking,
        mean_per_joint_error=mean_per_joint,
        mean_root_pos_error=mean_root_err,
        mean_mechanical_power=mean_power,
        max_joint_limit_violations=max_violations,
        mean_foot_penetration=mean_foot_pen,
        frame_metrics=frame_list,
    )


def aggregate_inverse_dynamics_metrics(
    clip_name: str,
    motion_type: str,
    frame_list: List[InverseDynamicsFrameMetrics],
) -> InverseDynamicsClipMetrics:
    ratios = np.stack([f.torque_limit_ratio for f in frame_list], axis=0)
    root_forces = np.array([f.root_force_norm for f in frame_list], dtype=np.float64)
    root_torques = np.array([f.root_torque_norm for f in frame_list], dtype=np.float64)
    exceeded_fraction = np.mean(ratios > 1.0, axis=1)

    return InverseDynamicsClipMetrics(
        clip_name=clip_name,
        motion_type=motion_type,
        mode="inverse_dynamics",
        n_frames=len(frame_list),
        mean_torque_limit_ratio=float(np.mean(ratios)),
        p95_torque_limit_ratio=float(np.percentile(ratios, 95)),
        max_torque_limit_ratio=float(np.max(ratios)),
        mean_exceeded_joint_fraction=float(np.mean(exceeded_fraction)),
        mean_root_force_N=float(np.mean(root_forces)),
        p95_root_force_N=float(np.percentile(root_forces, 95)),
        mean_root_torque_Nm=float(np.mean(root_torques)),
        p95_root_torque_Nm=float(np.percentile(root_torques, 95)),
        mean_per_joint_ratio=np.mean(ratios, axis=0),
        p95_per_joint_ratio=np.percentile(ratios, 95, axis=0),
        frame_metrics=frame_list,
    )
