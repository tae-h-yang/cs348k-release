"""Render risk-explainer videos for MotionBricks qpos clips.

The standard comparison videos show a single clip-level risk number. These
videos make the score inspectable by displaying per-frame inverse-dynamics
signals:

  - status color and instantaneous risk,
  - timeline spikes,
  - component bars for torque/root/velocity/acceleration/jerk,
  - top torque-limited joints at the current frame.

Usage:
    python scripts/render_risk_explainer.py --clip walk_seed0
    python scripts/render_risk_explainer.py --clip hand_crawling_seed0 --K 1 8
    python scripts/render_risk_explainer.py --default_set
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio
import mujoco
import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - cv2 is expected in the project env
    raise SystemExit("render_risk_explainer.py requires opencv-python/cv2") from exc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from physics_eval.inverse_dynamics import InverseDynamicsEvaluator
from physics_eval.physaware import _soft_excess
from physics_eval.validation import validate_qpos_sequence

DATA_DIR = REPO / "data" / "guided_ablation"
RESULTS_CSV = REPO / "results" / "guided_ablation_full.csv"
OUT_DIR = REPO / "results" / "videos" / "risk_explainer"
MODEL_XML = REPO / "assets" / "g1" / "scene_29dof.xml"

WIDTH = 560
HEIGHT = 480
INFO_W = 340
PANEL_W = WIDTH + INFO_W
FPS = 30


@dataclass
class FrameRisk:
    risk: float
    torque: float
    root_force: float
    root_torque: float
    velocity: float
    acceleration: float
    smoothness: float
    max_torque_ratio: float
    root_force_N: float
    root_torque_Nm: float
    top_joints: list[tuple[str, float]]


def risk_color_bgr(risk: float) -> tuple[int, int, int]:
    if risk <= 25:
        return (60, 190, 80)
    if risk <= 80:
        return (45, 150, 235)
    return (45, 55, 230)


def risk_label(risk: float) -> str:
    if risk <= 25:
        return "ACCEPT"
    if risk <= 80:
        return "REVIEW"
    return "REJECT"


def load_clip_risks() -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    if not RESULTS_CSV.exists():
        return out
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            if int(row["K"]) in (1, 4, 8, 16):
                out[(row["clip"], int(row["K"]))] = float(row["full_risk"])
    return out


def joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for i in range(model.nu):
        joint_id = int(model.actuator_trnid[i, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        names.append(name or f"joint_{i}")
    return names


def analyze_frame_risk(qpos_seq: np.ndarray, evaluator: InverseDynamicsEvaluator) -> list[FrameRisk]:
    qpos_seq = validate_qpos_sequence(qpos_seq, min_frames=2)
    qvel = evaluator.compute_qvel(qpos_seq)
    qacc = evaluator.compute_qacc(qvel)
    jerk = np.zeros((len(qpos_seq), evaluator.model.nu), dtype=np.float64)
    if len(qacc) > 1:
        jerk[1:] = np.abs(np.diff(qacc[:, 6:], axis=0) / evaluator.dt)
        jerk[0] = jerk[1]

    names = joint_names(evaluator.model)
    frames: list[FrameRisk] = []
    for t in range(len(qpos_seq)):
        evaluator.data.qpos[:] = qpos_seq[t]
        evaluator.data.qvel[:] = qvel[t]
        evaluator.data.qacc[:] = qacc[t]
        mujoco.mj_inverse(evaluator.model, evaluator.data)

        torques = evaluator.data.qfrc_inverse[6:].copy()
        root = evaluator.data.qfrc_inverse[:6].copy()
        ratio = np.abs(torques) / evaluator.actuator_limits

        max_ratio = float(np.max(ratio))
        root_force = float(np.linalg.norm(root[:3]))
        root_torque = float(np.linalg.norm(root[3:]))
        max_vel = float(np.percentile(np.abs(qvel[t, 6:]), 95))
        max_acc = float(np.percentile(np.abs(qacc[t, 6:]), 95))
        mean_jerk = float(np.mean(jerk[t]))

        torque_risk = 35.0 * _soft_excess(max_ratio, 1.0)
        root_force_risk = 20.0 * _soft_excess(root_force, 10000.0)
        root_torque_risk = 15.0 * _soft_excess(root_torque, 2000.0)
        velocity_risk = 10.0 * _soft_excess(max_vel, 10.0)
        acceleration_risk = 10.0 * _soft_excess(max_acc, 120.0)
        smoothness_risk = 10.0 * _soft_excess(mean_jerk, 1200.0)
        risk = float(
            torque_risk
            + root_force_risk
            + root_torque_risk
            + velocity_risk
            + acceleration_risk
            + smoothness_risk
        )

        top_idx = np.argsort(-ratio)[:3]
        top = [(names[i], float(ratio[i])) for i in top_idx]
        frames.append(FrameRisk(
            risk=risk,
            torque=torque_risk,
            root_force=root_force_risk,
            root_torque=root_torque_risk,
            velocity=velocity_risk,
            acceleration=acceleration_risk,
            smoothness=smoothness_risk,
            max_torque_ratio=max_ratio,
            root_force_N=root_force,
            root_torque_Nm=root_torque,
            top_joints=top,
        ))
    return frames


def make_camera(root_xy: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([root_xy[0], root_xy[1], 0.78])
    cam.azimuth = 90.0
    cam.elevation = -15.0
    cam.distance = 3.5
    return cam


def render_qpos(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> np.ndarray:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=make_camera(data.qpos[:2]))
    return renderer.render().copy()


def put_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.52,
             color: tuple[int, int, int] = (235, 235, 235), thick: int = 1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_bar(img: np.ndarray, label: str, value: float, x: int, y: int, w: int, max_value: float,
             color: tuple[int, int, int]) -> None:
    h = 11
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill = int(w * min(max(value, 0.0) / max_value, 1.0))
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    put_text(img, f"{label}: {value:.1f}", x, y - 5, scale=0.42, color=(215, 215, 215))


def draw_timeline(img: np.ndarray, risks: list[FrameRisk], t: int, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), (38, 38, 38), -1)
    vals = np.array([r.risk for r in risks], dtype=np.float64)
    cap = max(85.0, float(np.percentile(vals, 98)) if len(vals) else 85.0)
    n = len(vals)
    for i, risk in enumerate(vals):
        xx = x + int(i * w / max(n - 1, 1))
        yy = y + h - int(h * min(risk / cap, 1.0))
        cv2.line(img, (xx, y + h), (xx, yy), risk_color_bgr(float(risk)), 1)
    cur_x = x + int(t * w / max(n - 1, 1))
    cv2.line(img, (cur_x, y - 4), (cur_x, y + h + 4), (255, 255, 255), 1)
    put_text(img, "risk over time", x, y - 8, scale=0.38, color=(215, 215, 215))


def annotate_panel(frame: np.ndarray, risks: list[FrameRisk], t: int, title: str, clip_risk: float) -> np.ndarray:
    out = np.zeros((HEIGHT, PANEL_W, 3), dtype=np.uint8)
    out[:, :WIDTH] = frame
    panel = out[:, WIDTH:]
    panel[:] = (24, 24, 24)

    r = risks[t]
    color = risk_color_bgr(r.risk)
    cv2.rectangle(out, (0, 0), (PANEL_W - 1, HEIGHT - 1), color, 3)
    cv2.rectangle(out, (0, 0), (PANEL_W, 52), (0, 0, 0), -1)
    put_text(out, title, 10, 22, scale=0.58, color=(245, 245, 245), thick=1)
    put_text(out, f"clip risk {clip_risk:.1f} | frame {t / FPS:.2f}s", 10, 45, scale=0.46, color=(210, 210, 210))

    cv2.rectangle(panel, (16, 14), (INFO_W - 16, 64), color, -1)
    put_text(panel, risk_label(r.risk), 30, 45, scale=0.78, color=(20, 20, 20), thick=2)
    put_text(panel, f"frame risk: {r.risk:.1f}", 20, 92, scale=0.54, color=(235, 235, 235))
    put_text(panel, f"max torque/limit: {r.max_torque_ratio:.2f}x", 20, 120, scale=0.46)
    put_text(panel, f"root force: {r.root_force_N / 1000:.1f} kN", 20, 146, scale=0.46)

    y = 170
    components = [
        ("torque", r.torque, (70, 190, 240)),
        ("root F", r.root_force, (60, 150, 230)),
        ("root M", r.root_torque, (80, 120, 220)),
        ("vel", r.velocity, (110, 200, 120)),
        ("acc", r.acceleration, (180, 170, 80)),
        ("jerk", r.smoothness, (220, 120, 120)),
    ]
    for label, value, bar_color in components:
        draw_bar(panel, label, value, 20, y, INFO_W - 42, 40.0, bar_color)
        y += 34

    put_text(panel, "top torque-limited joints", 20, y + 6, scale=0.42, color=(220, 220, 220))
    for name, ratio in r.top_joints:
        y += 23
        joint_color = (70, 210, 90) if ratio <= 1.25 else (45, 150, 235) if ratio <= 5.0 else (45, 55, 230)
        put_text(panel, f"{name[:23]:23s} {ratio:4.1f}x", 20, y, scale=0.40, color=joint_color)

    draw_timeline(out, risks, t, 12, HEIGHT - 42, WIDTH - 24, 26)
    return out


def render_risk_comparison(clip: str, ks: list[int], args: argparse.Namespace) -> Path | None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    evaluator = InverseDynamicsEvaluator(MODEL_XML)
    clip_risks = load_clip_risks()
    loaded = []

    for k in ks:
        path = Path(args.data_dir) / f"{clip}_K{k}.npy"
        if not path.exists():
            print(f"skip {clip} K={k}: missing {path}")
            return None
        qpos = np.load(path)
        risks = analyze_frame_risk(qpos, evaluator)
        loaded.append((k, qpos, risks, clip_risks.get((clip, k), float(np.mean([r.risk for r in risks])))))

    T = min(len(qpos) for _, qpos, _, _ in loaded)
    renderers = [mujoco.Renderer(model, height=HEIGHT, width=WIDTH) for _ in loaded]
    datas = [mujoco.MjData(model) for _ in loaded]
    divider = np.full((HEIGHT, 4, 3), 35, dtype=np.uint8)
    frames = []

    for t in range(T):
        panels = []
        for idx, (k, qpos, risks, clip_risk) in enumerate(loaded):
            frame = render_qpos(renderers[idx], model, datas[idx], qpos[t])
            panel = annotate_panel(frame, risks, t, f"{clip}  K={k}", clip_risk)
            panels.append(panel)
            if idx < len(loaded) - 1:
                panels.append(divider)
        frames.append(np.concatenate(panels, axis=1))

    for renderer in renderers:
        renderer.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_vs_".join(f"K{k}" for k in ks)
    out_path = OUT_DIR / f"{clip}_{suffix}_risk_explainer.mp4"
    imageio.mimwrite(str(out_path), frames, fps=FPS, macro_block_size=1,
                     quality=8, output_params=["-vf", "format=yuv420p"])
    print(f"saved {out_path} ({T} frames)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default=None, help="Clip stem, e.g. walk_seed0")
    parser.add_argument("--K", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--default_set", action="store_true")
    args = parser.parse_args()

    if args.default_set:
        clips = [
            "walk_seed0",
            "walk_happy_dance_seed0",
            "hand_crawling_seed0",
        ]
    elif args.clip:
        clips = [args.clip]
    else:
        clips = ["walk_seed0"]

    for clip in clips:
        render_risk_comparison(clip, args.K, args)
    print(f"done: {OUT_DIR}")


if __name__ == "__main__":
    main()
