"""
Video rendering for kinematic-to-dynamic gap visualization.

Produces side-by-side MP4s: kinematic replay (left) vs physics execution (right).
This makes the gap immediately visible — the kinematic robot tracks perfectly
while the physics robot stumbles and falls.

Usage:
    from analysis.render import render_clip_video
    render_clip_video(sim, qpos_seq, clip_name="walk_01", motion_type="locomotion")
"""

import numpy as np
import mujoco
import imageio
from pathlib import Path
from typing import Optional
from physics_eval.metrics import FALL_HEIGHT_THRESHOLD
from physics_eval.validation import validate_qpos_sequence

RESULTS_DIR = Path(__file__).parents[2] / "results"
VIDEO_DIR = RESULTS_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Render settings
WIDTH, HEIGHT = 640, 480
FPS_OUT = 30        # output video fps (matches MotionBricks frame rate)
SIDE_BY_SIDE = True


def _make_camera(root_xy: np.ndarray = np.zeros(2)) -> mujoco.MjvCamera:
    """Side-view camera tracking the robot's root XY position."""
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat    = np.array([root_xy[0], root_xy[1], 0.75])
    cam.azimuth   = 90.0     # side view — robot walks left→right in frame
    cam.elevation = -15.0
    cam.distance  = 3.5
    return cam


def _render_frame(renderer: mujoco.Renderer, data,
                  root_xy: np.ndarray) -> np.ndarray:
    """Render one frame, camera locked to robot's XY. Returns (H, W, 3) uint8."""
    cam = _make_camera(root_xy)
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def render_clip_video(
    sim,
    qpos_seq: np.ndarray,
    clip_name: str = "clip",
    motion_type: str = "unknown",
    side_by_side: bool = SIDE_BY_SIDE,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Render a side-by-side video: kinematic replay (left) vs physics execution (right).

    Args:
        sim:         PhysicsSimulator instance (provides model + controller)
        qpos_seq:    (T, 36) kinematic qpos sequence
        clip_name:   Used in output filename
        motion_type: Used for display annotation
        side_by_side: If True, render both modes; otherwise physics only
        output_path: Override output path (default: results/videos/<clip_name>.mp4)

    Returns:
        Path to the saved video file.
    """
    qpos_seq = validate_qpos_sequence(qpos_seq, name=clip_name, min_frames=2)

    if output_path is None:
        output_path = VIDEO_DIR / f"{clip_name}.mp4"

    model = sim.model
    T = len(qpos_seq)

    w = WIDTH * 2 if side_by_side else WIDTH
    camera = _make_camera()

    # We need two independent MjData objects for simultaneous kinematic + physics
    data_phys = mujoco.MjData(model)
    data_kin  = mujoco.MjData(model)

    # Set up physics: settle into initial pose
    mujoco.mj_resetData(model, data_phys)
    data_phys.qpos[:] = qpos_seq[0]
    mujoco.mj_forward(model, data_phys)
    # Run settle using the simulator's controller
    q_t0 = qpos_seq[0, 7:]
    dq_zero = np.zeros(29)
    for _ in range(200):
        torques = sim.controller.compute_torques(
            q_t0, data_phys.qpos[7:], dq_zero, data_phys.qvel[6:],
            gravity_comp=data_phys.qfrc_bias[6:]
        )
        data_phys.ctrl[:] = torques
        mujoco.mj_step(model, data_phys)
        np.clip(data_phys.qvel, -sim.MAX_VEL, sim.MAX_VEL, out=data_phys.qvel)

    renderer_phys = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    renderer_kin  = mujoco.Renderer(model, height=HEIGHT, width=WIDTH) if side_by_side else None

    frames = []
    fell = False

    for t in range(T):
        # ── kinematic replay ────────────────────────────────────────────────
        if side_by_side:
            mujoco.mj_resetData(model, data_kin)
            data_kin.qpos[:] = qpos_seq[t]
            mujoco.mj_forward(model, data_kin)
            kin_xy = data_kin.qpos[:2].copy()
            frame_kin = _render_frame(renderer_kin, data_kin, kin_xy)

        # ── physics step ────────────────────────────────────────────────────
        if not fell:
            q_target  = qpos_seq[t, 7:]
            dq_target = sim._finite_diff_velocity(qpos_seq, t)
            torques = sim.controller.compute_torques(
                q_target, data_phys.qpos[7:].copy(), dq_target,
                data_phys.qvel[6:].copy(), gravity_comp=data_phys.qfrc_bias[6:].copy()
            )
            data_phys.ctrl[:] = torques
            for _ in range(sim._substeps_for_frame(t)):
                mujoco.mj_step(model, data_phys)
            np.clip(data_phys.qvel, -sim.MAX_VEL, sim.MAX_VEL, out=data_phys.qvel)
            if not np.isfinite(data_phys.qpos).all():
                fell = True

        tracking_rmse = float(np.sqrt(np.mean((data_phys.qpos[7:] - qpos_seq[t, 7:]) ** 2)))

        if data_phys.qpos[2] < FALL_HEIGHT_THRESHOLD:
            fell = True

        phys_xy = data_phys.qpos[:2].copy()
        frame_phys = _render_frame(renderer_phys, data_phys, phys_xy)

        # ── compose frame ───────────────────────────────────────────────────
        if side_by_side:
            # Add thin divider
            divider = np.full((HEIGHT, 4, 3), 60, dtype=np.uint8)
            frame = np.concatenate([frame_kin, divider, frame_phys], axis=1)
            _add_label(frame, "Kinematic (no physics)", x=10)
            _add_label(frame, f"Physics PD  [{clip_name}]", x=WIDTH + 14)
            _add_label(frame, f"t={t / FPS_OUT:.2f}s", x=10, y=44)
            _add_label(frame, f"RMSE={tracking_rmse:.2f} rad", x=WIDTH + 14, y=44)
        else:
            frame = frame_phys
            _add_label(frame, f"Physics PD  [{clip_name}]", x=10)
            _add_label(frame, f"t={t / FPS_OUT:.2f}s  RMSE={tracking_rmse:.2f} rad", x=10, y=44)

        if fell:
            _add_label(frame, "FELL", x=w // 2 - 20, y=HEIGHT - 20, color=(255, 60, 60))

        frames.append(frame)

    renderer_phys.close()
    if renderer_kin:
        renderer_kin.close()

    imageio.mimwrite(str(output_path), frames, fps=FPS_OUT, macro_block_size=1)
    return output_path


def render_reference_video(
    sim,
    qpos_seq: np.ndarray,
    clip_name: str = "clip",
    output_path: Optional[Path] = None,
) -> Path:
    """Render a clean kinematic-reference video with no physics rollout."""
    qpos_seq = validate_qpos_sequence(qpos_seq, name=clip_name, min_frames=1)
    if output_path is None:
        output_path = VIDEO_DIR / f"reference_{clip_name}.mp4"

    model = sim.model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    frames = []

    for t in range(len(qpos_seq)):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(model, data)
        frame = _render_frame(renderer, data, data.qpos[:2].copy())
        _add_label(frame, f"Kinematic reference [{clip_name}]", x=10)
        _add_label(frame, f"t={t / FPS_OUT:.2f}s", x=10, y=44)
        frames.append(frame)

    renderer.close()
    imageio.mimwrite(str(output_path), frames, fps=FPS_OUT, macro_block_size=1)
    return output_path


def _add_label(frame: np.ndarray, text: str, x: int = 10, y: int = 20,
               color=(255, 255, 255)):
    """Burn a simple text label into the frame using a bitmap font."""
    try:
        import cv2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
    except ImportError:
        pass  # skip labels if cv2 not installed — video still saves fine
