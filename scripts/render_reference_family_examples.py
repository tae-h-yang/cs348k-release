"""Render one KIMODO reference-only G1 clip for each prompt family.

These videos are for the early deck section that introduces what generated
reference motions look like before any SONIC rollout is shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets/g1/scene_29dof.xml"
SOURCE_DIR = Path("/home/rewardai/repos/cs348k/results/presentation_redghost_all100/_kimodo_rollouts")
OUT_DIR = ROOT / "slides/assets/videos/kimodo_reference_family_examples"
POSTER_DIR = ROOT / "slides/assets/video_posters/kimodo_reference_family_examples"

FPS = 50
WIDTH = 640
HEIGHT = 360


@dataclass(frozen=True)
class Example:
    family: str
    clip_id: str
    subcategory: str


EXAMPLES = [
    Example("locomotion + recovery", "hrb_001_forward_walk", "forward_walk"),
    Example("manipulation + loco-manipulation", "hrb_056_open_door", "open_door"),
    Example("floor / low posture", "hrb_027_hand_crawl", "hand_crawl"),
    Example("dance / expressive", "hrb_018_happy_dance", "happy_dance"),
    Example("athletic + terrain stress", "hrb_097_split_squat_jump", "split_squat_jump"),
    Example("communication / safety", "hrb_077_wave", "wave"),
]


def load_ref_qpos(example: Example) -> np.ndarray:
    path = SOURCE_DIR / f"{example.clip_id}_Kimodo.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    qpos = np.asarray(np.load(path, allow_pickle=True)["ref_qpos"], dtype=np.float64)
    qpos = qpos.copy()
    qpos[:, 0] -= qpos[0, 0]
    qpos[:, 1] -= qpos[0, 1]
    return qpos[:199]


def make_camera(qpos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(qpos[0]), float(qpos[1]), max(0.75, float(qpos[2]))]
    cam.azimuth = 100
    cam.elevation = -18
    cam.distance = 3.4
    return cam


def render_example(example: Example) -> None:
    qpos = load_ref_qpos(example)
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    out_path = OUT_DIR / f"{example.clip_id}_{example.subcategory}_reference.mp4"
    poster_path = POSTER_DIR / f"{example.clip_id}_{example.subcategory}_reference.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    poster = None
    try:
        for i, q in enumerate(qpos):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            cam = make_camera(q)
            renderer.update_scene(data, camera=cam)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            writer.write(frame)
            if i == len(qpos) // 2:
                poster = frame.copy()
    finally:
        writer.release()
        renderer.close()

    if poster is None:
        poster = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.imwrite(str(poster_path), poster)
    print(out_path)


def main() -> None:
    for example in EXAMPLES:
        render_example(example)


if __name__ == "__main__":
    main()
