"""Render selected SONIC rollouts as actual G1 mesh plus red reference ghost."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets" / "g1" / "scene_29dof.xml"
DEFAULT_ROLLOUT_DIR = ROOT / "results" / "humanoid100_final_eval" / "final_100_selected_rollouts"
DEFAULT_SELECTED = ROOT / "results" / "humanoid100_final_eval" / "final_selector" / "selected_methods.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "humanoid100_final_eval" / "final_100_selected_overlay_videos"
DEFAULT_INDEX = ROOT / "results" / "humanoid100_final_eval" / "final_100_selected_overlay_videos.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def selected_references(path: Path, selector: str) -> list[dict[str, str]]:
    rows = [row for row in read_rows(path) if row["selector"] == selector]
    rows.sort(key=lambda row: row["prompt_id"])
    return rows


def make_ref_model() -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(XML))
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name == "floor":
            model.geom_rgba[gid] = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            model.geom_rgba[gid] = np.array([1.0, 0.02, 0.02, 0.50])
    return model


def camera_for(sim_qpos: np.ndarray, ref_qpos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    root = 0.5 * (sim_qpos[:3] + ref_qpos[:3])
    cam.lookat[:] = [float(root[0]), float(root[1]), max(0.65, float(root[2]))]
    sep = float(np.linalg.norm(sim_qpos[:2] - ref_qpos[:2]))
    cam.distance = max(3.1, min(6.0, 3.2 + sep))
    cam.azimuth = 90
    cam.elevation = -18
    return cam


def render_rgb(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer, qpos: np.ndarray, cam: mujoco.MjvCamera) -> np.ndarray:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def red_mask(img: np.ndarray) -> np.ndarray:
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    mask = (r > 35) & (r > 1.25 * g + 8) & (r > 1.25 * b + 8)
    return mask


def put(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.48, color=(245, 245, 245)) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_overlay(npz_path: Path, out_path: Path, label: dict[str, str], fps: int, width: int, height: int) -> dict[str, object]:
    z = np.load(npz_path, allow_pickle=True)
    sim = np.asarray(z["sim_qpos"], dtype=np.float64)
    ref = np.asarray(z["ref_qpos"], dtype=np.float64)
    n = min(len(sim), len(ref))
    if n == 0:
        raise ValueError(f"{npz_path} has no rollout frames")

    actual_model = mujoco.MjModel.from_xml_path(str(XML))
    ref_model = make_ref_model()
    actual_data = mujoco.MjData(actual_model)
    ref_data = mujoco.MjData(ref_model)
    actual_renderer = mujoco.Renderer(actual_model, height=height, width=width)
    ref_renderer = mujoco.Renderer(ref_model, height=height, width=width)

    initial_sim_qpos = np.asarray(z["initial_sim_qpos"], dtype=np.float64)
    initial_ref_qpos = np.asarray(z["initial_ref_qpos"], dtype=np.float64)
    initial_sim_qvel = np.asarray(z["initial_sim_qvel"], dtype=np.float64)
    init_qpos_error = float(np.max(np.abs(initial_sim_qpos - initial_ref_qpos)))
    init_qvel_norm = float(np.linalg.norm(initial_sim_qvel))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    fell = bool(z["fell"])
    fall_frame = int(z["fall_frame"])
    for i in range(n):
        cam = camera_for(sim[i], ref[i])
        actual = render_rgb(actual_model, actual_data, actual_renderer, sim[i], cam)
        ghost = render_rgb(ref_model, ref_data, ref_renderer, ref[i], cam)
        mask = red_mask(ghost)
        frame = actual.copy()
        frame[mask] = (0.58 * actual[mask] + 0.42 * ghost[mask]).astype(np.uint8)
        frame[:88, :, :] = (0.60 * frame[:88, :, :]).astype(np.uint8)
        put(frame, f"{label['prompt_id']} {label['subcategory']} | {label['selected_method']}", 12, 25)
        put(frame, "solid robot: SONIC/MuJoCo physics   red ghost: selected reference", 12, 52, 0.44)
        status = f"t={i / fps:.2f}s  track={float(label['sonic_track_seconds']):.2f}s  rmse={float(label['sonic_mean_tracking_rmse']):.3f}"
        if fell:
            status += f"  fall={fall_frame / fps:.2f}s"
            color = (255, 205, 150)
        else:
            status += "  no fall"
            color = (170, 255, 185)
        put(frame, status, 12, 78, 0.42, color=color)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    actual_renderer.close()
    ref_renderer.close()

    return {
        "prompt_id": label["prompt_id"],
        "subcategory": label["subcategory"],
        "selected_method": label["selected_method"],
        "selected_reference": label["selected_reference"],
        "video_path": str(out_path),
        "frames": n,
        "fell": fell,
        "fall_frame": fall_frame,
        "track_seconds": label["sonic_track_seconds"],
        "mean_tracking_rmse": label["sonic_mean_tracking_rmse"],
        "initial_qpos_max_abs_error": init_qpos_error,
        "initial_qvel_norm": init_qvel_norm,
        "init_reference_pose": bool(z["init_reference_pose"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_csv", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--selector", default="sonic_verified_best")
    parser.add_argument("--rollout_dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--index_csv", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    rows = selected_references(args.selected_csv, args.selector)
    outputs: list[dict[str, object]] = []
    for i, row in enumerate(rows, start=1):
        npz = args.rollout_dir / f"{row['selected_reference']}.npz"
        out = args.out_dir / f"{row['prompt_id']}_{row['subcategory']}_{row['selected_method']}.mp4"
        outputs.append(render_overlay(npz, out, row, args.fps, args.width, args.height))
        print(f"[{i:03d}/{len(rows):03d}] {out}")
    write_csv(args.index_csv, outputs)
    print(f"Wrote {len(outputs)} videos to {args.out_dir}")
    print(f"Index: {args.index_csv}")


if __name__ == "__main__":
    main()
