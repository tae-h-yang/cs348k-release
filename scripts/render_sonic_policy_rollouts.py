"""Render side-by-side SONIC policy rollout traces saved by evaluate_sonic_policy_mujoco.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets" / "g1" / "scene_29dof.xml"
H, W = 480, 640


def make_camera(qpos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(qpos[0]), float(qpos[1]), 0.75]
    cam.distance = 4.0
    cam.azimuth = 90
    cam.elevation = -18
    return cam


def draw_label(img: np.ndarray, text: str, xy: tuple[int, int], color=(255, 255, 255)) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def fit_text(text: str, max_chars: int = 54) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def render_qpos(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer, qpos: np.ndarray) -> np.ndarray:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=make_camera(qpos))
    return renderer.render()


def render_npz(path: Path, out: Path, fps: int) -> None:
    z = np.load(path, allow_pickle=True)
    sim = z["sim_qpos"]
    ref = z["ref_qpos"]
    n = min(len(sim), len(ref))
    if n == 0:
        raise ValueError(f"{path} has no frames")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data_ref = mujoco.MjData(model)
    data_sim = mujoco.MjData(model)
    ren_ref = mujoco.Renderer(model, height=H, width=W)
    ren_sim = mujoco.Renderer(model, height=H, width=W)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, H))
    name = str(z["reference"])
    fell = bool(z["fell"])
    fall_frame = int(z["fall_frame"])
    for i in range(n):
        left = render_qpos(model, data_ref, ren_ref, ref[i])
        right = render_qpos(model, data_sim, ren_sim, sim[i])
        frame = np.concatenate([left, right], axis=1)
        frame[:78, :, :] = (0.72 * frame[:78, :, :]).astype(np.uint8)
        draw_label(frame, "reference", (18, 26))
        draw_label(frame, "SONIC policy", (W + 18, 26))
        draw_label(frame, fit_text(name), (18, 56))
        draw_label(frame, f"t={i / fps:.2f}s", (W - 98, 56))
        status = f"fell at {fall_frame / fps:.2f}s" if fell else "no fall"
        draw_label(frame, status, (W + 18, 56), color=(255, 180, 120) if fell else (160, 255, 160))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    ren_ref.close()
    ren_sim.close()
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", nargs="+", type=Path)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "videos" / "sonic_policy")
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    for path in args.rollouts:
        render_npz(path, args.out_dir / f"{path.stem}.mp4", args.fps)


if __name__ == "__main__":
    main()
