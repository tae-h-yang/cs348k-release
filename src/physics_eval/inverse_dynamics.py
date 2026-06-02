"""Inverse-dynamics feasibility analysis for kinematic humanoid clips."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .metrics import (
    InverseDynamicsFrameMetrics,
    InverseDynamicsClipMetrics,
    aggregate_inverse_dynamics_metrics,
)
from .simulator import SCENE_XML
from .validation import validate_qpos_sequence


class InverseDynamicsEvaluator:
    """
    Replays a qpos trajectory and asks what generalized forces would be needed
    to execute it exactly.

    The 29 actuated joint torques are compared against the model actuator
    limits. The first 6 inverse-dynamics components are the free-root wrench;
    nonzero values there are the external support/balance forces the kinematic
    trajectory implicitly assumes but the real robot cannot actuate directly.
    """

    def __init__(self, xml_path: str | Path = SCENE_XML, fps: float = 30.0):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.dt = 1.0 / fps

        if self.model.nq != 36 or self.model.nv != 35 or self.model.nu != 29:
            raise ValueError(
                f"Expected G1 layout nq=36, nv=35, nu=29; got "
                f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}"
            )
        self.actuator_limits = np.maximum(np.abs(self.model.actuator_ctrlrange).max(axis=1), 1e-8)

    def evaluate_clip(
        self,
        qpos_seq: np.ndarray,
        clip_name: str = "unknown",
        motion_type: str = "unknown",
    ) -> InverseDynamicsClipMetrics:
        qpos_seq = validate_qpos_sequence(qpos_seq, name=clip_name, min_frames=2)
        qvel_seq = self.compute_qvel(qpos_seq)
        qacc_seq = self.compute_qacc(qvel_seq)

        frames: list[InverseDynamicsFrameMetrics] = []
        for t in range(len(qpos_seq)):
            self.data.qpos[:] = qpos_seq[t]
            self.data.qvel[:] = qvel_seq[t]
            self.data.qacc[:] = qacc_seq[t]
            mujoco.mj_inverse(self.model, self.data)

            required_torques = self.data.qfrc_inverse[6:].copy()
            root_wrench = self.data.qfrc_inverse[:6].copy()
            ratio = np.abs(required_torques) / self.actuator_limits

            frames.append(InverseDynamicsFrameMetrics(
                required_torques=required_torques,
                torque_limit_ratio=ratio,
                root_wrench=root_wrench,
                root_force_norm=float(np.linalg.norm(root_wrench[:3])),
                root_torque_norm=float(np.linalg.norm(root_wrench[3:])),
                max_torque_limit_ratio=float(np.max(ratio)),
                n_torque_limit_exceeded=int(np.sum(ratio > 1.0)),
            ))

        return aggregate_inverse_dynamics_metrics(clip_name, motion_type, frames)

    def compute_qvel(self, qpos_seq: np.ndarray) -> np.ndarray:
        """Quaternion-safe finite-difference qvel from qpos."""
        qpos_seq = validate_qpos_sequence(qpos_seq, min_frames=2)
        qvel_seq = np.zeros((len(qpos_seq), self.model.nv), dtype=np.float64)
        for t in range(len(qpos_seq)):
            if t == 0:
                qa, qb, h = qpos_seq[0], qpos_seq[1], self.dt
            elif t == len(qpos_seq) - 1:
                qa, qb, h = qpos_seq[-2], qpos_seq[-1], self.dt
            else:
                qa, qb, h = qpos_seq[t - 1], qpos_seq[t + 1], 2.0 * self.dt
            mujoco.mj_differentiatePos(self.model, qvel_seq[t], h, qa, qb)
        return qvel_seq

    def compute_qacc(self, qvel_seq: np.ndarray) -> np.ndarray:
        """Finite-difference acceleration from qvel."""
        if qvel_seq.ndim != 2 or qvel_seq.shape[1] != self.model.nv or len(qvel_seq) < 2:
            raise ValueError(f"Expected qvel shape (T, {self.model.nv}) with T >= 2")
        qacc_seq = np.zeros_like(qvel_seq)
        qacc_seq[0] = (qvel_seq[1] - qvel_seq[0]) / self.dt
        qacc_seq[-1] = (qvel_seq[-1] - qvel_seq[-2]) / self.dt
        if len(qvel_seq) > 2:
            qacc_seq[1:-1] = (qvel_seq[2:] - qvel_seq[:-2]) / (2.0 * self.dt)
        return qacc_seq
