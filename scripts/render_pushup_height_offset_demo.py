"""Render a pushup-pose height-offset sensitivity demo for the CS348K slides.

This is a visual sensitivity demo, not a measured pass/fail table entry. It
uses an existing qpos trace and renders two side-by-side panels: left with the
reference root lowered and a MuJoCo PD rollout initialized from that lowered
pose, right with the reference lifted and the same proxy played at 1.2x speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_sonic_policy_mujoco import KD, KP  # noqa: E402

DEPLOY_ROOT = Path("/home/rewardai/repos/GR00T-WholeBodyControl/gear_sonic_deploy")
SIM_XML = ROOT / "assets/g1/scene_29dof.xml"
SOURCE_NPZ = Path(
    "/home/rewardai/repos/cs348k/results/presentation_redghost_all100/"
    "_kimodo_rollouts/hrb_034_pushup_pose_Kimodo.npz"
)
OUT_MP4 = ROOT / "slides/assets/videos/kimodo_repair_rescues/pushup_pose_height_offset_demo.mp4"
OUT_POSTER = ROOT / "slides/assets/video_posters/kimodo_repair_rescues/pushup_pose_height_offset_demo.jpg"

ROBOT_QPOS = 36
FPS = 50
WIDTH = 644
HEIGHT = 360
DIVIDER_WIDTH = 4


def match_repair_card_style(frame: np.ndarray) -> np.ndarray:
    """Match the custom pushup render to the surrounding repair-card videos.

    The pushup demo must keep its original content/roles, so this only adjusts
    color: the custom camera exposes a much brighter sky band than the standard
    repair videos, and its red ghost reads a little more saturated.
    """
    out = frame.copy()
    h = out.shape[0]

    # The custom low camera exposes the sky strip much brighter than the
    # standard repair videos. Keep the correction narrow so the robot/floor
    # action keeps the same exposure as the other cards.
    gains = np.array([0.58, 0.59, 0.61], dtype=np.float32)
    fade_end = min(52, h)
    for y in range(fade_end):
        weight = max(0.0, 1.0 - y / fade_end)
        row_gain = (1.0 - weight) + weight * gains
        out[y] = np.clip(out[y].astype(np.float32) * row_gain, 0, 255).astype(np.uint8)
    sky_strip = min(45, h)
    out[:sky_strip] = np.clip(
        out[:sky_strip].astype(np.float32) * np.array([0.78, 0.79, 0.80], dtype=np.float32),
        0,
        255,
    ).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 8) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 80)
    hsv[:, :, 1][red] = np.clip(hsv[:, :, 1][red].astype(np.float32) * 0.92, 0, 255).astype(np.uint8)
    hsv[:, :, 2][red] = np.clip(hsv[:, :, 2][red].astype(np.float32) * 0.98, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _prepend_names(elem, prefix: str) -> None:
    if "name" in elem.attrib:
        elem.attrib["name"] = prefix + elem.attrib["name"]
    for child in elem:
        _prepend_names(child, prefix)


def _replace_attribute(elem, attribute: str, value: str) -> None:
    if attribute in elem.attrib:
        elem.attrib[attribute] = value
    for child in elem:
        _replace_attribute(child, attribute, value)


def build_two_robot_model() -> mujoco.MjModel:
    scene = etree.parse(str(DEPLOY_ROOT / "g1/scene_empty.xml"))
    scene_asset = scene.find("asset")
    scene_default = scene.find("default")
    scene_worldbody = scene.find("worldbody")
    if scene_asset is None or scene_default is None or scene_worldbody is None:
        raise RuntimeError("Malformed SONIC scene_empty.xml")

    ref_robot = etree.parse(str(DEPLOY_ROOT / "g1/g1_29dof_old.xml"))
    ref_asset = ref_robot.find("asset")
    ref_default = ref_robot.find("default")
    ref_worldbody = ref_robot.find("worldbody")
    if ref_asset is None or ref_default is None or ref_worldbody is None:
        raise RuntimeError("Malformed g1_29dof_old.xml")
    for mesh in ref_asset.findall("mesh"):
        mesh.set("file", str(DEPLOY_ROOT / "g1/meshes" / mesh.get("file")))
        scene_asset.append(mesh)
    for default in ref_default.findall("default"):
        scene_default.append(default)
    ref_body = ref_worldbody.find("body")
    if ref_body is None:
        raise RuntimeError("No robot body in g1_29dof_old.xml")
    _prepend_names(ref_body, "ref_")
    _replace_attribute(ref_body, "rgba", "0.9 0.08 0.06 0.55")
    ref_body.set("pos", "0 -0.45 0")
    scene_worldbody.append(ref_body)

    solid_robot = etree.parse(str(DEPLOY_ROOT / "g1/g1_29dof_old.xml"))
    solid_body = solid_robot.find("worldbody").find("body")
    if solid_body is None:
        raise RuntimeError("No robot body in second g1_29dof_old.xml")
    _prepend_names(solid_body, "solid_")
    solid_body.set("pos", "0 0.45 0")
    scene_worldbody.append(solid_body)

    return mujoco.MjModel.from_xml_string(etree.tostring(scene, encoding="unicode"))


def normalize_clip(qpos: np.ndarray) -> np.ndarray:
    out = qpos.copy()
    out[:, 0] -= out[0, 0]
    out[:, 1] -= out[0, 1]
    return out


def make_proxy(
    qpos: np.ndarray,
    *,
    delay: int,
    x_offset: float,
    z_offset: float,
    speed: float = 1.0,
) -> np.ndarray:
    idx = np.maximum((np.arange(len(qpos)) * speed).astype(int) - delay, 0)
    idx = np.minimum(idx, len(qpos) - 1)
    proxy = qpos[idx].copy()
    proxy[:, 0] += x_offset
    proxy[:, 2] += z_offset
    return proxy


def simulate_pd_rollout(ref_qpos: np.ndarray) -> np.ndarray:
    """Track joint targets with MuJoCo dynamics from the first reference pose."""
    model = mujoco.MjModel.from_xml_path(str(SIM_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = ref_qpos[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    substeps = max(1, round((1.0 / FPS) / model.opt.timestep))
    ctrl_lo = model.actuator_ctrlrange[:, 0]
    ctrl_hi = model.actuator_ctrlrange[:, 1]
    out = []
    for target in ref_qpos:
        for _ in range(substeps):
            tau = KP * (target[7:] - data.qpos[7:]) - KD * data.qvel[6:]
            data.ctrl[:] = np.clip(tau, ctrl_lo, ctrl_hi)
            mujoco.mj_step(model, data)
        out.append(data.qpos.copy())
        if not np.isfinite(data.qpos).all():
            break
    if len(out) < len(ref_qpos):
        last = out[-1] if out else ref_qpos[0]
        out.extend([last.copy() for _ in range(len(ref_qpos) - len(out))])
    return np.asarray(out, dtype=np.float64)


def render_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    ref_qpos: np.ndarray,
    solid_qpos: np.ndarray,
) -> np.ndarray:
    mujoco.mj_resetData(model, data)
    data.qpos[:ROBOT_QPOS] = ref_qpos
    data.qpos[ROBOT_QPOS : 2 * ROBOT_QPOS] = solid_qpos
    mujoco.mj_kinematics(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.7
    cam.azimuth = 90
    cam.elevation = -20
    cam.lookat[:] = np.array([0.0, 0.0, 0.52], dtype=np.float64)
    renderer.update_scene(data, camera=cam)
    frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    # Match the exposure of the existing repair-rescue videos on slide 16.
    frame = cv2.convertScaleAbs(frame, alpha=1.90, beta=5)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * 1.06, 0, 255).astype(np.uint8)
    return match_repair_card_style(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))


def main() -> None:
    z = np.load(SOURCE_NPZ, allow_pickle=True)
    qpos = normalize_clip(z["ref_qpos"].astype(np.float64))
    n = min(len(qpos), 199)
    qpos = qpos[:n]

    low_ref = qpos.copy()
    low_ref[:, 2] -= 0.075
    low_actual = simulate_pd_rollout(low_ref)

    lifted_ref = qpos.copy()
    lifted_ref[:, 2] += 0.055
    lifted_proxy = make_proxy(lifted_ref, delay=4, x_offset=0.035, z_offset=-0.015, speed=1.2)

    model = build_two_robot_model()
    panel_w = (WIDTH - DIVIDER_WIDTH) // 2
    model.vis.global_.offwidth = panel_w
    model.vis.global_.offheight = HEIGHT
    data_left = mujoco.MjData(model)
    data_right = mujoco.MjData(model)
    renderer_left = mujoco.Renderer(model, height=HEIGHT, width=panel_w)
    renderer_right = mujoco.Renderer(model, height=HEIGHT, width=panel_w)

    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    OUT_POSTER.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT_MP4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    poster = None
    try:
        for i in range(n):
            left = render_scene(
                model,
                data_left,
                renderer_left,
                low_ref[i],
                low_actual[i],
            )
            right = render_scene(
                model,
                data_right,
                renderer_right,
                lifted_proxy[i],
                lifted_ref[i],
            )
            divider = np.full((HEIGHT, DIVIDER_WIDTH, 3), 238, dtype=np.uint8)
            frame = np.concatenate([left, divider, right], axis=1)
            writer.write(frame)
            if i == n // 2:
                poster = frame.copy()
    finally:
        writer.release()
        renderer_left.close()
        renderer_right.close()
    if poster is None:
        poster = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.imwrite(str(OUT_POSTER), poster)
    print(OUT_MP4)
    print(OUT_POSTER)


if __name__ == "__main__":
    main()
