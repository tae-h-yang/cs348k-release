"""
Physics execution simulator for kinematic-to-dynamic gap analysis.

Two evaluation modes:
  evaluate_clip()           — mj_step + PD control (physics execution)
  evaluate_clip_kinematic() — mj_forward only (zero-gap kinematic replay)

Comparing the two gives the kinematic-to-dynamic gap directly.
"""

import numpy as np
import mujoco
from pathlib import Path

from .pd_controller import PDController, KP, KD, ACTUATOR_FORCE_LIMITS
from .metrics import (
    FrameMetrics, ClipMetrics,
    compute_frame_metrics, aggregate_clip_metrics
)
from .validation import validate_qpos_sequence

ASSETS_DIR = Path(__file__).parents[2] / "assets" / "g1"
SCENE_XML = str(ASSETS_DIR / "scene_29dof.xml")


class PhysicsSimulator:
    """
    Runs kinematic qpos sequences under MuJoCo physics via a PD tracking controller,
    and also provides a kinematic replay baseline (mj_forward only).

    Usage:
        sim = PhysicsSimulator()
        phys = sim.evaluate_clip(qpos_seq, clip_name="walk_01", motion_type="locomotion")
        kin  = sim.evaluate_clip_kinematic(qpos_seq, clip_name="walk_01", motion_type="locomotion")
    """

    MAX_VEL = 50.0  # rad/s or m/s — clip qvel after each step to prevent NaN cascade

    def __init__(self, xml_path: str = SCENE_XML,
                 pd_kp_scale: float = 0.5,
                 pd_kd_scale: float = 1.0,
                 pd_force_scale: float = 1.0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.physics_dt = self.model.opt.timestep          # 0.002 s
        self.ctrl_dt    = 1.0 / 30.0                       # MotionBricks frame rate
        self.substeps   = max(1, round(self.ctrl_dt / self.physics_dt))  # compatibility

        self.controller = PDController(
            kp=KP * pd_kp_scale,
            kd=KD * pd_kd_scale,
            force_limits=ACTUATOR_FORCE_LIMITS * pd_force_scale,
        )

        self._joint_ranges  = self._get_joint_ranges()
        self._foot_geom_ids = self._get_foot_geom_ids()

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_joint_ranges(self) -> np.ndarray:
        """Returns (29, 2) array of [lo, hi] joint limits in radians."""
        ranges = []
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                ranges.append([self.model.jnt_range[i, 0], self.model.jnt_range[i, 1]])
        return np.array(ranges, dtype=np.float64)

    def _get_foot_geom_ids(self) -> tuple:
        left  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        right = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        assert left >= 0 and right >= 0, "Foot body IDs not found in model"
        left_geoms = np.array([
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] == left and self.model.geom_contype[i] != 0
        ], dtype=np.int32)
        right_geoms = np.array([
            i for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] == right and self.model.geom_contype[i] != 0
        ], dtype=np.int32)
        assert len(left_geoms) > 0 and len(right_geoms) > 0, "Foot collision geoms not found"
        return (left_geoms, right_geoms)

    def _substeps_for_frame(self, t: int) -> int:
        """Number of physics steps for control frame t with exact 30 Hz average."""
        start = round(t * self.ctrl_dt / self.physics_dt)
        end = round((t + 1) * self.ctrl_dt / self.physics_dt)
        return max(1, end - start)

    def _settle(self, init_qpos: np.ndarray, n_steps: int = 200):
        """Hold initial pose under gravity until contact forces stabilize."""
        q_target  = init_qpos[7:]
        dq_target = np.zeros(29)
        for _ in range(n_steps):
            torques = self.controller.compute_torques(
                q_target, self.data.qpos[7:], dq_target, self.data.qvel[6:],
                gravity_comp=self.data.qfrc_bias[6:]
            )
            self.data.ctrl[:] = torques
            mujoco.mj_step(self.model, self.data)
            np.clip(self.data.qvel, -self.MAX_VEL, self.MAX_VEL, out=self.data.qvel)

    MAX_TARGET_VEL = 10.0  # rad/s — clamp finite-diff velocity

    def _finite_diff_velocity(self, q_seq: np.ndarray, t: int) -> np.ndarray:
        if t == 0:
            vel = (q_seq[1, 7:] - q_seq[0, 7:]) / self.ctrl_dt
        elif t >= len(q_seq) - 1:
            vel = (q_seq[-1, 7:] - q_seq[-2, 7:]) / self.ctrl_dt
        else:
            vel = (q_seq[t + 1, 7:] - q_seq[t - 1, 7:]) / (2 * self.ctrl_dt)
        return np.clip(vel, -self.MAX_TARGET_VEL, self.MAX_TARGET_VEL)

    def _make_nan_frame(self, t: int) -> FrameMetrics:
        return FrameMetrics(
            tracking_rmse=float("nan"),
            per_joint_error=np.full(29, float("nan")),
            root_pos_error=float("nan"),
            root_height=0.0, fell=True,
            n_joint_limit_violations=0,
            foot_penetration_left=0.0, foot_penetration_right=0.0,
            mechanical_power=float("nan"), contact_forces=np.array([]),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate_clip(self, qpos_seq: np.ndarray, clip_name: str = "unknown",
                      motion_type: str = "unknown",
                      early_stop_on_fall: bool = True) -> ClipMetrics:
        """
        Physics execution: mj_step + PD tracking controller.

        Args:
            qpos_seq:    (T, 36) kinematic qpos sequence.
            clip_name:   Identifier for logging/plotting.
            motion_type: Category label (e.g. "locomotion", "expressive").
            early_stop_on_fall: Stop and pad remaining frames once the robot falls.
        """
        qpos_seq = validate_qpos_sequence(qpos_seq, name=clip_name, min_frames=2)

        T = len(qpos_seq)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos_seq[0]
        mujoco.mj_forward(self.model, self.data)
        self._settle(qpos_seq[0])

        frame_metrics: list[FrameMetrics] = []

        for t in range(T):
            q_target  = qpos_seq[t, 7:]
            dq_target = self._finite_diff_velocity(qpos_seq, t)

            torques = self.controller.compute_torques(
                q_target, self.data.qpos[7:].copy(), dq_target, self.data.qvel[6:].copy(),
                gravity_comp=self.data.qfrc_bias[6:].copy()
            )
            self.data.ctrl[:] = torques

            for _ in range(self._substeps_for_frame(t)):
                mujoco.mj_step(self.model, self.data)
            np.clip(self.data.qvel, -self.MAX_VEL, self.MAX_VEL, out=self.data.qvel)

            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                nan_fm = self._make_nan_frame(t)
                frame_metrics.append(nan_fm)
                frame_metrics.extend([nan_fm] * (T - t - 1))
                print(f"      [NaN at frame {t} — unstable]")
                break

            fm = compute_frame_metrics(
                self.model, self.data, q_target, torques,
                self._joint_ranges, self._foot_geom_ids
            )
            fm.root_pos_error = float(np.linalg.norm(self.data.qpos[:3] - qpos_seq[t, :3]))
            frame_metrics.append(fm)

            if early_stop_on_fall and fm.fell:
                fall_fm = FrameMetrics(
                    tracking_rmse=fm.tracking_rmse,
                    per_joint_error=fm.per_joint_error,
                    root_pos_error=fm.root_pos_error,
                    root_height=fm.root_height, fell=True,
                    n_joint_limit_violations=fm.n_joint_limit_violations,
                    foot_penetration_left=fm.foot_penetration_left,
                    foot_penetration_right=fm.foot_penetration_right,
                    mechanical_power=0.0, contact_forces=fm.contact_forces,
                )
                frame_metrics.extend([fall_fm] * (T - t - 1))
                break

        return aggregate_clip_metrics(clip_name, motion_type, "physics", frame_metrics)

    def evaluate_clip_kinematic(self, qpos_seq: np.ndarray, clip_name: str = "unknown",
                                motion_type: str = "unknown") -> ClipMetrics:
        """
        Kinematic replay baseline: mj_forward only (no dynamics, no control).

        Tracking RMSE is identically 0 (qpos is set directly from the sequence).
        Useful for checking joint limit violations and foot penetration in the
        raw kinematic data before any physics is applied.
        """
        qpos_seq = validate_qpos_sequence(qpos_seq, name=clip_name, min_frames=1)

        T = len(qpos_seq)
        zeros_torque = np.zeros(29)
        frame_metrics: list[FrameMetrics] = []

        for t in range(T):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = qpos_seq[t]
            mujoco.mj_forward(self.model, self.data)

            fm = compute_frame_metrics(
                self.model, self.data, qpos_seq[t, 7:], zeros_torque,
                self._joint_ranges, self._foot_geom_ids
            )
            fm.root_pos_error = 0.0  # kinematic replay has zero root error by definition
            frame_metrics.append(fm)

        return aggregate_clip_metrics(clip_name, motion_type, "kinematic", frame_metrics)
