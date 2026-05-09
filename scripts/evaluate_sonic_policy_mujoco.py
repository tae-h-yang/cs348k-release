"""Run SONIC's released G1 policy on exported MotionBricks references in MuJoCo.

This script is a local tracking-policy bridge for the MotionBricks screening
project.  It uses the public SONIC deployment ONNX encoder/decoder weights from
GR00T-WholeBodyControl, reconstructs the G1 reference-motion observation path,
and drives the 29-DoF MuJoCo G1 with the policy's target-joint outputs.

It is deliberately labeled as an approximate harness: the native SONIC deploy
binary expects Unitree LowState DDS messages, while this script supplies the
same observation tensors from MuJoCo state.  The resulting metrics are still far
more relevant than inverse dynamics alone because a learned tracking policy is
closed around the simulated robot.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
GEAR = Path("/home/rewardai/repos/GR00T-WholeBodyControl/gear_sonic_deploy")
DEFAULT_REF_DIR = ROOT / "results" / "sonic_references"
DEFAULT_OUT = ROOT / "results" / "sonic_policy_mujoco_tracking.csv"
DEFAULT_SUMMARY = ROOT / "results" / "sonic_policy_mujoco_summary.csv"
DEFAULT_XML = ROOT / "assets" / "g1" / "scene_29dof.xml"
DEFAULT_ENCODER = GEAR / "policy" / "release" / "model_encoder.onnx"
DEFAULT_POLICY = GEAR / "policy" / "release" / "model_decoder.onnx"

CTRL_DT = 1.0 / 50.0
FALL_HEIGHT = 0.55
MAX_QVEL = 60.0

ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

LOWER_BODY_ISAAC = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int64)
WRIST_ISAAC = np.array([23, 25, 27, 24, 26, 28], dtype=np.int64)

ONE_DEGREE = 0.0174533
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

EFFORT_5020 = 25.0
EFFORT_7520_14 = 88.0
EFFORT_7520_22 = 139.0
EFFORT_4010 = 5.0

DEFAULT_ANGLES = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

KP = np.array(
    [
        STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
        STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
        STIFFNESS_7520_14, 2 * STIFFNESS_5020, 2 * STIFFNESS_5020,
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_4010, STIFFNESS_4010,
    ],
    dtype=np.float64,
)
KD = np.array(
    [
        DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2 * DAMPING_5020, 2 * DAMPING_5020,
        DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22, 2 * DAMPING_5020, 2 * DAMPING_5020,
        DAMPING_7520_14, 2 * DAMPING_5020, 2 * DAMPING_5020,
        DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_4010, DAMPING_4010,
        DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_4010, DAMPING_4010,
    ],
    dtype=np.float64,
)
ACTION_SCALE = np.array(
    [
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_7520_22 / STIFFNESS_7520_22,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_7520_14 / STIFFNESS_7520_14,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_4010 / STIFFNESS_4010,
        0.25 * EFFORT_4010 / STIFFNESS_4010,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_5020 / STIFFNESS_5020,
        0.25 * EFFORT_4010 / STIFFNESS_4010,
        0.25 * EFFORT_4010 / STIFFNESS_4010,
    ],
    dtype=np.float64,
)


def qnormalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return qmul(qmul(q, np.array([0.0, *v], dtype=np.float64)), qconj(q))[1:]


def heading_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = qnormalize(q)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def quat_to_6d(q: np.ndarray) -> np.ndarray:
    w, x, y, z = qnormalize(q)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    return np.array([r00, r01, r10, r11, r20, r21], dtype=np.float64)


def read_csv_array(path: Path) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.asarray(arr, dtype=np.float64)


@dataclass
class Reference:
    name: str
    joint_isaac: np.ndarray
    joint_vel_isaac: np.ndarray
    body_pos: np.ndarray
    body_quat: np.ndarray


def load_reference(path: Path) -> Reference:
    joint = read_csv_array(path / "joint_pos.csv")
    joint_vel = read_csv_array(path / "joint_vel.csv")
    body_pos = read_csv_array(path / "body_pos.csv").reshape(len(joint), -1, 3)[:, 0]
    body_quat = read_csv_array(path / "body_quat.csv").reshape(len(joint), -1, 4)[:, 0]
    body_quat = np.array([qnormalize(q) for q in body_quat], dtype=np.float64)
    return Reference(path.name, joint, joint_vel, body_pos, body_quat)


def make_sessions(encoder_path: Path, policy_path: Path, provider: str) -> tuple[ort.InferenceSession, ort.InferenceSession]:
    providers = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    }[provider]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return (
        ort.InferenceSession(str(encoder_path), sess_options=so, providers=providers),
        ort.InferenceSession(str(policy_path), sess_options=so, providers=providers),
    )


def future_rows(arr: np.ndarray, t: int, n: int, step: int) -> np.ndarray:
    idx = np.minimum(t + np.arange(n) * step, len(arr) - 1)
    return arr[idx]


def build_encoder_obs(ref: Reference, t: int, base_quat: np.ndarray, init_ref_quat: np.ndarray) -> np.ndarray:
    obs = np.zeros(1762, dtype=np.float32)
    obs[0] = 0.0  # encoder_mode_4: mode id=0 (g1), remaining 3 slots zero
    obs[4:294] = future_rows(ref.joint_isaac, t, 10, 5).reshape(-1)
    obs[294:584] = future_rows(ref.joint_vel_isaac, t, 10, 5).reshape(-1)

    # motion_root_z_position_10frame_step5 [584:594] and motion_root_z_position [594:595]
    future_z = future_rows(ref.body_pos, t, 10, 5)[:, 2]
    obs[584:594] = future_z
    obs[594] = ref.body_pos[min(t, len(ref.body_pos) - 1), 2]

    apply_delta_heading = qmul(heading_quat(base_quat), qconj(heading_quat(init_ref_quat)))

    # motion_anchor_orientation (current frame) [595:601]
    cur_ref_quat = ref.body_quat[min(t, len(ref.body_quat) - 1)]
    cur_new_ref = qmul(apply_delta_heading, cur_ref_quat)
    cur_rel = qmul(qconj(base_quat), cur_new_ref)
    obs[595:601] = quat_to_6d(cur_rel)

    # motion_anchor_orientation_10frame_step5 [601:661]
    for i, qref in enumerate(future_rows(ref.body_quat, t, 10, 5)):
        new_ref = qmul(apply_delta_heading, qref)
        rel = qmul(qconj(base_quat), new_ref)
        obs[601 + i * 6:601 + (i + 1) * 6] = quat_to_6d(rel)
    return obs[None, :]


def sim_state_for_policy(data: mujoco.MjData, last_action: np.ndarray) -> dict[str, np.ndarray]:
    q_mujoco = data.qpos[7:].copy()
    dq_mujoco = data.qvel[6:].copy()
    body_q = q_mujoco[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]
    body_dq = dq_mujoco[MUJOCO_TO_ISAACLAB]
    base_quat = qnormalize(data.qpos[3:7])
    return {
        "base_quat": base_quat,
        "base_ang_vel": qrotate(qconj(base_quat), data.qvel[3:6]),
        "body_q": body_q,
        "body_dq": body_dq,
        "last_action": last_action.copy(),
        "gravity": qrotate(qconj(base_quat), np.array([0.0, 0.0, -1.0])),
    }


def history_stack(history: list[dict[str, np.ndarray]], key: str, dim: int) -> np.ndarray:
    rows = [h[key] for h in history[-10:]]
    if len(rows) < 10:
        rows = [np.zeros(dim, dtype=np.float64) for _ in range(10 - len(rows))] + rows
    return np.asarray(rows[-10:], dtype=np.float64).reshape(-1)


def build_policy_obs(token: np.ndarray, history: list[dict[str, np.ndarray]]) -> np.ndarray:
    obs = np.zeros(994, dtype=np.float32)
    obs[0:64] = token.reshape(-1)
    obs[64:94] = history_stack(history, "base_ang_vel", 3)
    obs[94:384] = history_stack(history, "body_q", 29)
    obs[384:674] = history_stack(history, "body_dq", 29)
    obs[674:964] = history_stack(history, "last_action", 29)
    obs[964:994] = history_stack(history, "gravity", 3)
    return obs[None, :]


def parse_name(name: str) -> tuple[str, int]:
    match = re.search(r"_K(\d+)$", name)
    if match is None:
        return name, 1
    k = int(match.group(1))
    base = name.rsplit("_K", 1)[0]
    return base, k


def find_best_start_frame(ref: Reference) -> int:
    """Find the reference frame whose lower-body pose is closest to DEFAULT_ANGLES."""
    default_isaac = DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]
    lb_diff = np.abs(ref.joint_isaac[:, :12] - default_isaac[:12]).sum(axis=1)
    return int(np.argmin(lb_diff))


def render_side_by_side_video(
    model: mujoco.MjModel,
    sim_qpos_arr: np.ndarray,
    ref_qpos_arr: np.ndarray,
    out_path: Path,
    fps: float = 50.0,
    height: int = 480,
    width: int = 640,
) -> None:
    import imageio
    renderer = mujoco.Renderer(model, height=height, width=width)
    data_sim = mujoco.MjData(model)
    data_ref = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -15
    cam.azimuth = 90
    frames = []
    for i in range(len(sim_qpos_arr)):
        # Sim side — camera tracks the root
        data_sim.qpos[:] = sim_qpos_arr[i]
        mujoco.mj_forward(model, data_sim)
        cam.lookat[:] = [data_sim.qpos[0], data_sim.qpos[1], data_sim.qpos[2]]
        renderer.update_scene(data_sim, camera=cam)
        img_sim = renderer.render().copy()

        # Reference side
        data_ref.qpos[:] = ref_qpos_arr[i]
        mujoco.mj_forward(model, data_ref)
        cam.lookat[:] = [data_ref.qpos[0], data_ref.qpos[1], data_ref.qpos[2]]
        renderer.update_scene(data_ref, camera=cam)
        img_ref = renderer.render().copy()

        frame = np.concatenate([img_sim, img_ref], axis=1)
        frames.append(frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264", quality=8)
    renderer.close()


def rollout_reference(
    ref: Reference,
    model: mujoco.MjModel,
    encoder: ort.InferenceSession,
    policy: ort.InferenceSession,
    max_seconds: float,
    kp_scale: float,
    kd_scale: float,
    save_npz: Path | None = None,
    video_path: Path | None = None,
    start_frame: int | None = None,
) -> dict[str, object]:
    # In real SONIC deployment, tracking always starts with the robot at DEFAULT
    # standing pose (body_q ≈ 0), NOT at reference frame 0.  start_frame is only
    # used to pick which part of the reference to track (we always track from 0).
    # The robot is placed at the reference's root xy/z but with DEFAULT joint angles.
    start_frame = 0

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = ref.body_pos[0]          # root at reference start position
    data.qpos[3:7] = ref.body_quat[0]        # root orientation from reference
    data.qpos[7:] = DEFAULT_ANGLES            # joints at DEFAULT (standing pose)
    mujoco.mj_forward(model, data)

    substeps = max(1, round(CTRL_DT / model.opt.timestep))
    ctrl_lo = model.actuator_ctrlrange[:, 0]
    ctrl_hi = model.actuator_ctrlrange[:, 1]
    last_action = np.zeros(29, dtype=np.float64)
    # Pre-warm history with 10 frames of DEFAULT standing state (body_q ≈ 0),
    # matching how the real deployment's state logger is populated before tracking.
    init_state = sim_state_for_policy(data, last_action)
    history: list[dict[str, np.ndarray]] = [init_state for _ in range(10)]
    tracking = []
    root_err = []
    action_norm = []
    torque_sat = []
    sim_qpos = []
    ref_qpos = []
    fell = False
    n_ref = len(ref.joint_isaac)
    max_frames = min(n_ref, int(round(max_seconds / CTRL_DT)))
    fall_frame = max_frames
    init_ref_quat = ref.body_quat[0]

    for local_t in range(max_frames):
        abs_t = local_t
        history.append(sim_state_for_policy(data, last_action))
        enc_obs = build_encoder_obs(ref, abs_t, history[-1]["base_quat"], init_ref_quat)
        token = encoder.run(None, {"obs_dict": enc_obs.astype(np.float32)})[0]
        pol_obs = build_policy_obs(token, history)
        action = policy.run(None, {"obs_dict": pol_obs.astype(np.float32)})[0].reshape(-1).astype(np.float64)

        q_target = DEFAULT_ANGLES + action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        # Apply PD at each physics substep (matches IsaacLab training and hardware).
        for _ in range(substeps):
            tau = KP * kp_scale * (q_target - data.qpos[7:]) - KD * kd_scale * data.qvel[6:]
            data.ctrl[:] = np.clip(tau, ctrl_lo, ctrl_hi)
            mujoco.mj_step(model, data)
        np.clip(data.qvel, -MAX_QVEL, MAX_QVEL, out=data.qvel)

        abs_t_ref = min(abs_t, n_ref - 1)
        ref_q_mujoco = ref.joint_isaac[abs_t_ref][ISAACLAB_TO_MUJOCO]
        ref_full = np.zeros(36, dtype=np.float64)
        ref_full[:3] = ref.body_pos[abs_t_ref]
        ref_full[3:7] = ref.body_quat[abs_t_ref]
        ref_full[7:] = ref_q_mujoco
        sim_qpos.append(data.qpos.copy())
        ref_qpos.append(ref_full)
        tracking.append(float(np.sqrt(np.mean((data.qpos[7:] - ref_q_mujoco) ** 2))))
        root_err.append(float(np.linalg.norm(data.qpos[:3] - ref.body_pos[abs_t_ref])))
        action_norm.append(float(np.linalg.norm(action)))
        torque_sat.append(float(np.mean((np.abs(tau - ctrl_lo) < 1e-6) | (np.abs(tau - ctrl_hi) < 1e-6))))
        last_action = action

        if not np.isfinite(data.qpos).all() or data.qpos[2] < FALL_HEIGHT:
            fell = True
            fall_frame = local_t + 1
            break

    base, k = parse_name(ref.name)
    sim_arr = np.asarray(sim_qpos, dtype=np.float64)
    ref_arr = np.asarray(ref_qpos, dtype=np.float64)

    if save_npz is not None:
        save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_npz,
            reference=ref.name,
            clip=base,
            K=k,
            sim_qpos=sim_arr,
            ref_qpos=ref_arr,
            fell=fell,
            fall_frame=fall_frame,
            ctrl_dt=CTRL_DT,
        )

    if video_path is not None:
        render_side_by_side_video(model, sim_arr, ref_arr, video_path)

    return {
        "clip": base,
        "reference": ref.name,
        "K": k,
        "frames": max_frames,
        "active_frames": len(tracking),
        "fell": fell,
        "fall_frame": fall_frame,
        "track_seconds": fall_frame * CTRL_DT,
        "mean_tracking_rmse": float(np.mean(tracking)) if tracking else float("nan"),
        "mean_root_error": float(np.mean(root_err)) if root_err else float("nan"),
        "mean_action_norm": float(np.mean(action_norm)) if action_norm else float("nan"),
        "mean_torque_saturation_frac": float(np.mean(torque_sat)) if torque_sat else float("nan"),
        "final_root_z": float(data.qpos[2]),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    ks = sorted({int(r["K"]) for r in rows})
    for k in ks:
        group = [r for r in rows if int(r["K"]) == k]
        if not group:
            continue
        out.append(
            {
                "group": f"K{k}",
                "n": len(group),
                "fell_count": sum(bool(r["fell"]) for r in group),
                "mean_track_seconds": float(np.mean([float(r["track_seconds"]) for r in group])),
                "mean_tracking_rmse": float(np.nanmean([float(r["mean_tracking_rmse"]) for r in group])),
                "mean_root_error": float(np.nanmean([float(r["mean_root_error"]) for r in group])),
                "mean_torque_saturation_frac": float(np.nanmean([float(r["mean_torque_saturation_frac"]) for r in group])),
            }
        )
    by_clip: dict[str, dict[int, dict[str, object]]] = {}
    for r in rows:
        by_clip.setdefault(str(r["clip"]), {})[int(r["K"])] = r
    paired = [v for v in by_clip.values() if 1 in v and 8 in v]
    if paired:
        out.append(
            {
                "group": "paired_K8_minus_K1",
                "n": len(paired),
                "fell_count": "",
                "mean_track_seconds": float(np.mean([float(p[8]["track_seconds"]) - float(p[1]["track_seconds"]) for p in paired])),
                "mean_tracking_rmse": float(np.nanmean([float(p[8]["mean_tracking_rmse"]) - float(p[1]["mean_tracking_rmse"]) for p in paired])),
                "mean_root_error": float(np.nanmean([float(p[8]["mean_root_error"]) - float(p[1]["mean_root_error"]) for p in paired])),
                "mean_torque_saturation_frac": float(np.nanmean([float(p[8]["mean_torque_saturation_frac"]) - float(p[1]["mean_torque_saturation_frac"]) for p in paired])),
            }
        )
    complete = [v for v in by_clip.values() if all(k in v for k in ks)]
    if len(ks) > 2 and complete:
        selected = [
            max(
                variants.values(),
                key=lambda r: (float(r["track_seconds"]), -float(r["mean_tracking_rmse"])),
            )
            for variants in complete
        ]
        out.append(
            {
                "group": "policy_selector_all_K",
                "n": len(selected),
                "fell_count": sum(bool(r["fell"]) for r in selected),
                "mean_track_seconds": float(np.mean([float(r["track_seconds"]) for r in selected])),
                "mean_tracking_rmse": float(np.nanmean([float(r["mean_tracking_rmse"]) for r in selected])),
                "mean_root_error": float(np.nanmean([float(r["mean_root_error"]) for r in selected])),
                "mean_torque_saturation_frac": float(np.nanmean([float(r["mean_torque_saturation_frac"]) for r in selected])),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--provider", choices=["cpu", "cuda", "tensorrt"], default="cuda")
    parser.add_argument("--max_seconds", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--kp_scale", type=float, default=1.0)
    parser.add_argument("--kd_scale", type=float, default=1.0)
    parser.add_argument("--save_rollouts_dir", type=Path, default=None)
    parser.add_argument("--save_names", nargs="*", default=[])
    parser.add_argument("--video_dir", type=Path, default=None, help="Save side-by-side .mp4 videos here")
    parser.add_argument("--video_names", nargs="*", default=[], help="Only render video for these refs (empty=all)")
    parser.add_argument("--start_frame", type=int, default=None, help="Fixed start frame (default: auto best-frame)")
    args = parser.parse_args()

    refs = [load_reference(p) for p in sorted(args.reference_dir.iterdir()) if (p / "joint_pos.csv").exists()]
    if args.limit:
        refs = refs[: args.limit]
    if not refs:
        raise SystemExit(f"No SONIC references found under {args.reference_dir}")

    encoder, policy = make_sessions(args.encoder, args.policy, args.provider)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    rows: list[dict[str, object]] = []
    for i, ref in enumerate(refs, start=1):
        save_npz = None
        if args.save_rollouts_dir is not None and (not args.save_names or ref.name in set(args.save_names)):
            save_npz = args.save_rollouts_dir / f"{ref.name}.npz"
        video_path = None
        if args.video_dir is not None and (not args.video_names or ref.name in set(args.video_names)):
            video_path = args.video_dir / f"{ref.name}.mp4"
        row = rollout_reference(
            ref, model, encoder, policy, args.max_seconds, args.kp_scale, args.kd_scale,
            save_npz=save_npz, video_path=video_path, start_frame=args.start_frame,
        )
        rows.append(row)
        print(
            f"[{i:03d}/{len(refs):03d}] {ref.name}: K={row['K']} "
            f"fell={row['fell']} seconds={float(row['track_seconds']):.2f} "
            f"rmse={float(row['mean_tracking_rmse']):.3f}"
        )

    write_csv(args.out_csv, rows)
    summary = summarize(rows)
    write_csv(args.summary_csv, summary)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.summary_csv}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
