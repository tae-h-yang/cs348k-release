"""
Render side-by-side K=1 vs K=8 comparison videos from saved .npy ablation clips.

Usage:
    # Render a few representative clips
    conda run -n base python scripts/render_comparison.py

    # Specific clip
    conda run -n base python scripts/render_comparison.py --clip walk_seed0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco
import imageio

REPO = Path(__file__).parent.parent
DATA_DIR = REPO / "data" / "guided_ablation"
OUT_DIR = REPO / "results" / "videos" / "comparison"
MODEL_XML = REPO / "assets" / "g1" / "scene_29dof.xml"

WIDTH, HEIGHT = 640, 480
FPS = 30


def _make_cam(xy: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([xy[0], xy[1], 0.75])
    cam.azimuth = 90.0
    cam.elevation = -15.0
    cam.distance = 3.5
    return cam


def _render_qpos(renderer, model, data, qpos):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    cam = _make_cam(data.qpos[:2])
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def _label(frame, text, x=10, y=24, color=(255, 255, 255)):
    try:
        import cv2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
    except ImportError:
        pass


def render_comparison(clip_name: str, model, risk_k1: float, risk_k8: float):
    p1 = DATA_DIR / f"{clip_name}_K1.npy"
    p8 = DATA_DIR / f"{clip_name}_K8.npy"
    if not p1.exists() or not p8.exists():
        print(f"  skip {clip_name} (missing files)")
        return None

    q1 = np.load(p1)
    q8 = np.load(p8)
    T = min(len(q1), len(q8))

    data1 = mujoco.MjData(model)
    data8 = mujoco.MjData(model)
    r1 = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    r8 = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    divider = np.full((HEIGHT, 6, 3), 40, dtype=np.uint8)
    frames = []

    for t in range(T):
        f1 = _render_qpos(r1, model, data1, q1[t])
        f8 = _render_qpos(r8, model, data8, q8[t])

        _label(f1, f"K=1  risk={risk_k1:.1f}", color=(255, 160, 80))
        _label(f8, f"K=8  risk={risk_k8:.1f}", color=(100, 220, 100))
        _label(f1, clip_name, y=HEIGHT - 10)
        _label(f1, f"t={t/FPS:.2f}s", y=48)

        frames.append(np.concatenate([f1, divider, f8], axis=1))

    r1.close(); r8.close()

    out = OUT_DIR / f"{clip_name}_K1_vs_K8.mp4"
    imageio.mimwrite(str(out), frames, fps=FPS, macro_block_size=1)
    print(f"  saved: {out.name}  ({T} frames)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=None, help="Single clip name, e.g. walk_seed0")
    parser.add_argument("--all", action="store_true", help="Render all clips")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load risk table
    import csv
    csv_path = REPO / "results" / "guided_ablation_full.csv"
    risks = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            k = int(row["K"])
            if k in (1, 8):
                risks[(row["clip"], k)] = float(row["full_risk"])

    print(f"Loading model: {MODEL_XML}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    print("Model loaded.")

    # Representative clips: one per type, lowest + highest improvement
    default_clips = [
        "walk_seed0",
        "walk_seed1",
        "walk_boxing_seed0",
        "walk_happy_dance_seed0",
        "walk_zombie_seed0",
        "injured_walk_seed0",
        "hand_crawling_seed0",
    ]

    if args.clip:
        clips = [args.clip]
    elif args.all:
        clips = sorted({p.stem.rsplit("_K", 1)[0] for p in DATA_DIR.glob("*_K1.npy")})
    else:
        clips = default_clips

    for clip in clips:
        r1 = risks.get((clip, 1), float("nan"))
        r8 = risks.get((clip, 8), float("nan"))
        print(f"\n{clip}  K1_risk={r1:.1f}  K8_risk={r8:.1f}")
        render_comparison(clip, model, r1, r8)

    print(f"\nDone. Videos in: {OUT_DIR}")


if __name__ == "__main__":
    main()
