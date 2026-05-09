"""
Online physics critic for segment-level scoring during MotionBricks generation.

At each generation step, MotionBricks produces a segment of ~15 frames.
This module scores each candidate segment with the inverse-dynamics critic
so the generation loop can pick the best one without scoring the full clip.
"""

from __future__ import annotations

import numpy as np

from .inverse_dynamics import InverseDynamicsEvaluator
from .physaware import _soft_excess


_WEIGHTS = {
    "torque": 35.0,
    "root_force": 20.0,
    "root_torque": 15.0,
    "velocity": 10.0,
    "acceleration": 10.0,
    "jerk": 10.0,
}
_THRESHOLDS = {
    "torque": 1.0,         # p95 torque/limit ratio
    "root_force": 10000.0, # N
    "root_torque": 2000.0, # Nm
    "velocity": 10.0,      # rad/s
    "acceleration": 120.0, # rad/s^2
    "jerk": 1200.0,        # rad/s^3
}


class OnlineSegmentCritic:
    """
    Scores a single MotionBricks generation segment with the physics critic.

    Designed for low latency: only computes on the new frames (typically 15-20
    frames), not the full accumulated trajectory.
    """

    def __init__(self, evaluator: InverseDynamicsEvaluator | None = None):
        self._evaluator = evaluator or InverseDynamicsEvaluator()
        self.dt = self._evaluator.dt

    def score_segment(self, qpos_segment: np.ndarray) -> float:
        """
        Score a short qpos segment. Returns the heuristic risk scalar.
        Lower is better. Returns inf for degenerate input.
        """
        if qpos_segment is None or len(qpos_segment) < 3:
            return float("inf")

        qpos = qpos_segment.astype(np.float64)

        try:
            inv = self._evaluator.evaluate_clip(qpos)
        except Exception:
            return float("inf")

        qvel = self._evaluator.compute_qvel(qpos)
        qacc = self._evaluator.compute_qacc(qvel)
        qjerk = (
            np.diff(qacc[:, 6:], axis=0) / self.dt if len(qacc) > 1
            else np.zeros((1, qacc.shape[1] - 6))
        )

        p95_vel = float(np.percentile(np.abs(qvel[:, 6:]), 95))
        p95_acc = float(np.percentile(np.abs(qacc[:, 6:]), 95))
        mean_jerk = float(np.mean(np.abs(qjerk)))

        risk = (
            _WEIGHTS["torque"]       * _soft_excess(inv.p95_torque_limit_ratio, _THRESHOLDS["torque"])
            + _WEIGHTS["root_force"] * _soft_excess(inv.p95_root_force_N,       _THRESHOLDS["root_force"])
            + _WEIGHTS["root_torque"]* _soft_excess(inv.p95_root_torque_Nm,     _THRESHOLDS["root_torque"])
            + _WEIGHTS["velocity"]   * _soft_excess(p95_vel,                    _THRESHOLDS["velocity"])
            + _WEIGHTS["acceleration"]* _soft_excess(p95_acc,                   _THRESHOLDS["acceleration"])
            + _WEIGHTS["jerk"]       * _soft_excess(mean_jerk,                  _THRESHOLDS["jerk"])
        )
        return float(risk)

    def select_best_candidate(
        self,
        segments: list[np.ndarray],
    ) -> tuple[int, float, list[float]]:
        """
        Score a list of candidate segments and return the index of the best one.

        Returns: (best_index, best_risk, all_risks)
        """
        risks = [self.score_segment(seg) for seg in segments]
        best_idx = int(np.argmin(risks))
        return best_idx, risks[best_idx], risks
