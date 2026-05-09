"""Export MotionBricks qpos clips to GEAR-SONIC reference-motion CSV folders.

This does not run the SONIC policy. It prepares the reference data that the
deployment/tracking stack expects: 50 Hz joint_pos/joint_vel plus root body
position/quaternion and metadata. The local MotionBricks clips are MuJoCo qpos
at 30 Hz; SONIC reference docs expect IsaacLab joint ordering, so this exporter
uses the mapping from gear_sonic_deploy's policy_parameters.hpp.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "guided_ablation_extended"
OUT_DIR = ROOT / "results" / "sonic_references"
SOURCE_FPS = 30.0
SONIC_FPS = 50.0

# From gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
     5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)


def read_guided_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(values)


def resample_linear(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if len(values) < 2:
        raise ValueError("Need at least two frames to resample")
    duration = (len(values) - 1) / source_fps
    n_out = max(2, int(round(duration * target_fps)) + 1)
    t_in = np.linspace(0.0, duration, len(values))
    t_out = np.linspace(0.0, duration, n_out)
    out = np.empty((n_out, values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(t_out, t_in, values[:, j])
    return out


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(norms, 1e-8)


def export_clip(qpos_path: Path, out_dir: Path, source_fps: float, target_fps: float) -> dict[str, object]:
    qpos = np.load(qpos_path).astype(np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2:
        raise ValueError(f"{qpos_path}: expected qpos shape (T, 36), got {qpos.shape}")

    root_pos = resample_linear(qpos[:, :3], source_fps, target_fps)
    root_quat = normalize_quat(resample_linear(qpos[:, 3:7], source_fps, target_fps))
    joints_mujoco = resample_linear(qpos[:, 7:], source_fps, target_fps)
    joints_isaac = joints_mujoco[:, MUJOCO_TO_ISAACLAB]
    joint_vel = np.gradient(joints_isaac, 1.0 / target_fps, axis=0)
    body_lin_vel = np.gradient(root_pos, 1.0 / target_fps, axis=0)
    body_ang_vel = np.zeros_like(body_lin_vel)

    write_csv(out_dir / "joint_pos.csv", [f"joint_{i}" for i in range(29)], joints_isaac)
    write_csv(out_dir / "joint_vel.csv", [f"joint_vel_{i}" for i in range(29)], joint_vel)
    write_csv(out_dir / "body_pos.csv", ["body_0_x", "body_0_y", "body_0_z"], root_pos)
    write_csv(out_dir / "body_quat.csv", ["body_0_w", "body_0_x", "body_0_y", "body_0_z"], root_quat)
    write_csv(
        out_dir / "body_lin_vel.csv",
        ["body_0_vel_x", "body_0_vel_y", "body_0_vel_z"],
        body_lin_vel,
    )
    write_csv(
        out_dir / "body_ang_vel.csv",
        ["body_0_angvel_x", "body_0_angvel_y", "body_0_angvel_z"],
        body_ang_vel,
    )

    with open(out_dir / "metadata.txt", "w") as f:
        f.write(f"Metadata for: {out_dir.name}\n")
        f.write("==============================\n\n")
        f.write("Body part indexes:\n")
        f.write("[0]\n\n")
        f.write(f"Total timesteps: {len(joints_isaac)}\n")

    with open(out_dir / "info.txt", "w") as f:
        f.write(f"MotionBricks qpos source: {qpos_path}\n")
        f.write(f"Source fps: {source_fps}\n")
        f.write(f"Export fps: {target_fps}\n")
        f.write("Joint order: IsaacLab order via ISAACLAB_TO_MUJOCO mapping\n")
        f.write("Body data: root-only pelvis position/quaternion\n")

    return {
        "motion": out_dir.name,
        "source_path": str(qpos_path),
        "out_dir": str(out_dir),
        "source_frames": len(qpos),
        "export_frames": len(joints_isaac),
        "duration_s": (len(joints_isaac) - 1) / target_fps,
        "joint_order": "isaaclab",
        "body_count": 1,
    }


def clip_mode_seed(row: dict[str, str]) -> tuple[str, int]:
    if "mode" in row and "seed_idx" in row:
        return row["mode"], int(row["seed_idx"])
    clip = row["clip"]
    match = re.search(r"_seed(\d+)$", clip)
    if match is None:
        return clip, 0
    return clip[: match.start()], int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results_csv", type=Path, default=ROOT / "results" / "guided_ablation_extended.csv")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--modes", nargs="*", default=["walk", "slow_walk", "walk_happy_dance", "walk_gun", "hand_crawling"])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--k_values", type=int, nargs="*", default=[1, 8])
    args = parser.parse_args()

    rows = read_guided_rows(args.results_csv)
    keep = []
    for row in rows:
        mode, seed_idx = clip_mode_seed(row)
        if mode not in args.modes:
            continue
        if seed_idx >= args.seeds:
            continue
        if int(row["K"]) not in args.k_values:
            continue
        keep.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in keep:
        raw_path = row.get("path", "")
        qpos_path = Path(raw_path) if raw_path else args.data_dir / f"{row['clip']}_K{row['K']}.npy"
        if not qpos_path.exists() or qpos_path.is_dir():
            qpos_path = args.data_dir / f"{row['clip']}_K{row['K']}.npy"
        motion_name = f"{row['clip']}_K{row['K']}"
        manifest.append(export_clip(qpos_path, args.out_dir / motion_name, SOURCE_FPS, SONIC_FPS))

    manifest_path = args.out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Exported {len(manifest)} SONIC references to {args.out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
