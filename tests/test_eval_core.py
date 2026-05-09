import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from run_eval import load_motion_dir
from physics_eval.inverse_dynamics import InverseDynamicsEvaluator
from physics_eval.metrics import FrameMetrics, aggregate_clip_metrics
from physics_eval.pd_controller import ACTUATOR_FORCE_LIMITS, PDController
from physics_eval.simulator import PhysicsSimulator
from physics_eval.validation import validate_qpos_sequence


def _qpos_sequence(n_frames=3):
    q = np.zeros((n_frames, 36), dtype=np.float64)
    q[:, 2] = 0.78
    q[:, 3] = 1.0
    return q


def test_loader_skips_invalid_arrays(tmp_path):
    np.save(tmp_path / "valid.npy", _qpos_sequence())
    np.save(tmp_path / "bad_shape.npy", np.zeros((3, 35)))
    np.save(tmp_path / "motion_labels.npy", {"valid": "static"})

    motions = load_motion_dir(tmp_path)

    assert len(motions) == 1
    assert motions[0][0] == "valid"
    assert motions[0][1] == "static"
    assert motions[0][2].shape == (3, 36)


def test_g1_model_layout():
    sim = PhysicsSimulator()

    assert sim.model.nq == 36
    assert sim.model.nv == 35
    assert sim.model.nu == 29


def test_pd_controller_clamps_torques():
    controller = PDController()
    tau = controller.compute_torques(
        q_target=np.ones(29) * 100.0,
        q=np.zeros(29),
        dq_target=np.zeros(29),
        dq=np.zeros(29),
    )

    np.testing.assert_allclose(np.abs(tau), ACTUATOR_FORCE_LIMITS)


def test_validation_rejects_empty_and_one_frame_dynamic_clip():
    with pytest.raises(ValueError, match="at least 1 frames"):
        validate_qpos_sequence(np.zeros((0, 36)), min_frames=1)

    sim = PhysicsSimulator()
    with pytest.raises(ValueError, match="at least 2 frames"):
        sim.evaluate_clip(_qpos_sequence(1), clip_name="one_frame")


def test_validation_normalizes_root_quaternions_without_mutating_input():
    q = _qpos_sequence(2)
    q[:, 3] = 2.0

    normalized = validate_qpos_sequence(q)

    assert np.allclose(q[:, 3], 2.0)
    np.testing.assert_allclose(np.linalg.norm(normalized[:, 3:7], axis=1), 1.0)


def test_inverse_dynamics_outputs_finite_values():
    evaluator = InverseDynamicsEvaluator()
    q = _qpos_sequence(4)
    q[:, 0] = np.linspace(0.0, 0.03, len(q))
    q[:, 7 + 3] = np.linspace(0.25, 0.35, len(q))

    qvel = evaluator.compute_qvel(q)
    result = evaluator.evaluate_clip(q, clip_name="tiny", motion_type="static")

    assert qvel.shape == (4, 35)
    assert np.isfinite(qvel).all()
    assert result.n_frames == 4
    assert np.isfinite(result.p95_torque_limit_ratio)
    assert result.mean_per_joint_ratio.shape == (29,)


def test_fall_at_zero_aggregation_uses_first_frame_only():
    first = FrameMetrics(
        tracking_rmse=1.0,
        per_joint_error=np.ones(29),
        root_pos_error=0.5,
        root_height=0.2,
        fell=True,
        n_joint_limit_violations=0,
        foot_penetration_left=0.01,
        foot_penetration_right=0.03,
        mechanical_power=10.0,
        contact_forces=np.array([]),
    )
    padded = FrameMetrics(
        tracking_rmse=100.0,
        per_joint_error=np.ones(29) * 100.0,
        root_pos_error=100.0,
        root_height=0.1,
        fell=True,
        n_joint_limit_violations=0,
        foot_penetration_left=1.0,
        foot_penetration_right=1.0,
        mechanical_power=100.0,
        contact_forces=np.array([]),
    )

    result = aggregate_clip_metrics("fall0", "test", "physics", [first, padded])

    assert result.time_to_fall == 0
    assert result.mean_tracking_rmse == 1.0
    assert result.mean_foot_penetration == pytest.approx(0.02)
