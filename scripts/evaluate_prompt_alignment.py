"""Evaluate task/prompt preservation for generated MotionBricks clips.

This is a proxy evaluator for the local mode-control preview release. It does
not claim learned text-motion retrieval; it checks whether the generated root
trajectory and body-motion statistics remain compatible with the requested
task after best-of-K reranking.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROMPT_CSV = ROOT / "configs" / "prompt_suite_105.csv"
DATA_DIR = ROOT / "data" / "guided_ablation_extended"
RESULTS_CSV = ROOT / "results" / "prompt_alignment.csv"
SUMMARY_CSV = ROOT / "results" / "prompt_alignment_summary.csv"
PLOT_PATH = ROOT / "results" / "prompt_alignment_by_category.png"
SCATTER_PATH = ROOT / "results" / "risk_vs_prompt_alignment.png"


ARM_SLICE = slice(15, 29)
LEG_SLICE = slice(0, 12)
FPS = 30.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def direction_vec(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    return np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)


def bounded_exp_error(value: float, target: float, scale: float) -> float:
    return float(math.exp(-abs(value - target) / max(scale, 1e-6)))


def clip_features(qpos: np.ndarray) -> dict[str, float]:
    root_xy = qpos[:, :2].astype(np.float64)
    joints = qpos[:, 7:].astype(np.float64)
    duration_s = max((len(qpos) - 1) / FPS, 1.0 / FPS)

    delta = root_xy[-1] - root_xy[0]
    step_xy = np.diff(root_xy, axis=0)
    path_len = float(np.sum(np.linalg.norm(step_xy, axis=1)))
    planar_disp = float(np.linalg.norm(delta))
    mean_speed = path_len / duration_s

    dj = np.diff(joints, axis=0) * FPS
    arm_energy = float(np.mean(np.abs(dj[:, ARM_SLICE]))) if len(dj) else 0.0
    leg_energy = float(np.mean(np.abs(dj[:, LEG_SLICE]))) if len(dj) else 0.0
    joint_energy = float(np.mean(np.abs(dj))) if len(dj) else 0.0
    arm_leg_ratio = arm_energy / max(leg_energy, 1e-6)

    yaw = yaw_from_quat_wxyz(qpos[:, 3:7].astype(np.float64))
    yaw_change = float(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0])

    return {
        "duration_s": duration_s,
        "path_length_m": path_len,
        "planar_displacement_m": planar_disp,
        "mean_speed_mps": mean_speed,
        "delta_x_m": float(delta[0]),
        "delta_y_m": float(delta[1]),
        "mean_root_height_m": float(np.mean(qpos[:, 2])),
        "min_root_height_m": float(np.min(qpos[:, 2])),
        "mean_joint_speed_rad_s": joint_energy,
        "mean_arm_speed_rad_s": arm_energy,
        "mean_leg_speed_rad_s": leg_energy,
        "arm_leg_motion_ratio": arm_leg_ratio,
        "yaw_change_rad": yaw_change,
    }


def alignment_score(row: dict[str, str], feat: dict[str, float]) -> dict[str, float]:
    category = row["category"]
    target_speed = float(row["target_speed_mps"])
    target_dir = float(row["target_direction_deg"])
    direction = direction_vec(target_dir)
    displacement = np.array([feat["delta_x_m"], feat["delta_y_m"]], dtype=np.float64)
    disp_norm = float(np.linalg.norm(displacement))

    if target_speed <= 1e-6:
        direction_score = bounded_exp_error(disp_norm, 0.0, 0.30)
        speed_score = bounded_exp_error(feat["mean_speed_mps"], 0.0, 0.15)
    else:
        progress = float(np.dot(displacement, direction))
        progress_speed = progress / feat["duration_s"]
        direction_cos = progress / max(disp_norm, 1e-6)
        direction_score = float(np.clip(0.5 * (direction_cos + 1.0), 0.0, 1.0))
        speed_score = bounded_exp_error(max(progress_speed, 0.0), target_speed, 0.35)

    static_score = bounded_exp_error(feat["planar_displacement_m"], 0.0, 0.20) * bounded_exp_error(
        feat["mean_joint_speed_rad_s"], 0.0, 0.25
    )
    low_posture_score = float(np.clip((0.78 - feat["mean_root_height_m"]) / 0.28, 0.0, 1.0))
    expressive_score = float(np.clip((feat["arm_leg_motion_ratio"] - 0.55) / 0.75, 0.0, 1.0))
    upright_score = float(np.clip((feat["mean_root_height_m"] - 0.50) / 0.25, 0.0, 1.0))

    if category == "static":
        style_score = static_score
        weights = (0.25, 0.25, 0.50)
    elif category == "whole_body_low":
        style_score = low_posture_score
        weights = (0.30, 0.25, 0.45)
    elif category == "expressive_locomotion":
        style_score = expressive_score
        weights = (0.30, 0.30, 0.40)
    else:
        style_score = upright_score
        weights = (0.40, 0.35, 0.25)

    total = (
        weights[0] * direction_score
        + weights[1] * speed_score
        + weights[2] * style_score
    )
    return {
        "alignment_score": float(np.clip(total, 0.0, 1.0)),
        "direction_score": float(np.clip(direction_score, 0.0, 1.0)),
        "speed_score": float(np.clip(speed_score, 0.0, 1.0)),
        "style_proxy_score": float(np.clip(style_score, 0.0, 1.0)),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for k in sorted({int(r["K"]) for r in rows}):
        group = [r for r in rows if int(r["K"]) == k]
        out.append({
            "group": f"K={k}",
            "n": len(group),
            "mean_alignment": float(np.mean([r["alignment_score"] for r in group])),
            "median_alignment": float(np.median([r["alignment_score"] for r in group])),
            "mean_speed_error_mps": float(np.mean([r["speed_error_mps"] for r in group])),
            "mean_displacement_m": float(np.mean([r["planar_displacement_m"] for r in group])),
        })
    for category in sorted({r["category"] for r in rows}):
        for k in sorted({int(r["K"]) for r in rows}):
            group = [r for r in rows if r["category"] == category and int(r["K"]) == k]
            if not group:
                continue
            out.append({
                "group": f"{category}_K={k}",
                "n": len(group),
                "mean_alignment": float(np.mean([r["alignment_score"] for r in group])),
                "median_alignment": float(np.median([r["alignment_score"] for r in group])),
                "mean_speed_error_mps": float(np.mean([r["speed_error_mps"] for r in group])),
                "mean_displacement_m": float(np.mean([r["planar_displacement_m"] for r in group])),
            })
    pairs: dict[str, dict[int, dict[str, object]]] = {}
    for r in rows:
        pairs.setdefault(str(r["prompt_id"]), {})[int(r["K"])] = r
    paired = [v for v in pairs.values() if 1 in v and 8 in v]
    if paired:
        delta = np.array([p[8]["alignment_score"] - p[1]["alignment_score"] for p in paired], dtype=float)
        disp_ratio = np.array([
            p[8]["planar_displacement_m"] / max(p[1]["planar_displacement_m"], 1e-6)
            for p in paired
        ])
        out.append({
            "group": "K8_minus_K1_paired",
            "n": len(paired),
            "mean_alignment_delta": float(np.mean(delta)),
            "median_alignment_delta": float(np.median(delta)),
            "alignment_improved_or_equal": int(np.sum(delta >= -0.05)),
            "displacement_ratio_in_0p5_1p5": int(np.sum((disp_ratio >= 0.5) & (disp_ratio <= 1.5))),
            "mean_displacement_ratio": float(np.mean(disp_ratio)),
        })
    return out


def plot_alignment(rows: list[dict[str, object]], out_path: Path) -> None:
    categories = sorted({r["category"] for r in rows})
    x = np.arange(len(categories))
    width = 0.36
    means = {}
    for k in [1, 8]:
        means[k] = [
            np.mean([r["alignment_score"] for r in rows if r["category"] == c and int(r["K"]) == k])
            for c in categories
        ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, means[1], width, label="MotionBricks K=1")
    ax.bar(x + width / 2, means[8], width, label="Screened K=8")
    ax.set_ylabel("Proxy prompt-alignment score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Task preservation proxies across 105 local MotionBricks prompts")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_risk_scatter(rows: list[dict[str, object]], risk_csv: Path, out_path: Path) -> None:
    if not risk_csv.exists():
        return
    risks = {}
    for r in load_rows(risk_csv):
        risks[(r["clip"], int(r["K"]))] = float(r["full_risk"])
    xs = []
    ys = []
    colors = []
    for r in rows:
        key = (str(r["clip"]), int(r["K"]))
        if key not in risks:
            continue
        xs.append(risks[key])
        ys.append(float(r["alignment_score"]))
        colors.append(int(r["K"]))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    color_arr = np.array(["#555555" if k == 1 else "#1f77b4" for k in colors])
    ax.scatter(xs, ys, c=color_arr, s=28, alpha=0.75, edgecolor="none")
    ax.set_xlabel("Inverse-dynamics heuristic risk")
    ax.set_ylabel("Proxy prompt-alignment score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Screening should lower risk without erasing task intent")
    ax.text(0.98, 0.05, "gray=K1, blue=K8", transform=ax.transAxes, ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_csv", type=Path, default=PROMPT_CSV)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--k_values", type=int, nargs="*", default=[1, 8])
    parser.add_argument("--out_csv", type=Path, default=RESULTS_CSV)
    args = parser.parse_args()

    rows = []
    for prompt in load_rows(args.prompt_csv):
        mode = prompt["motionbricks_mode"]
        seed = int(prompt["seed_idx"])
        clip = f"{mode}_seed{seed}"
        for k in args.k_values:
            path = args.data_dir / f"{clip}_K{k}.npy"
            if not path.exists():
                continue
            qpos = np.load(path)
            feat = clip_features(qpos)
            scores = alignment_score(prompt, feat)
            target_speed = float(prompt["target_speed_mps"])
            rows.append({
                "prompt_id": prompt["prompt_id"],
                "clip": clip,
                "category": prompt["category"],
                "mode": mode,
                "seed_idx": seed,
                "K": k,
                "prompt_text": prompt["prompt_text"],
                "target_speed_mps": target_speed,
                "target_direction_deg": float(prompt["target_direction_deg"]),
                "speed_error_mps": abs(feat["mean_speed_mps"] - target_speed),
                **feat,
                **scores,
                "path": str(path),
            })

    write_csv(args.out_csv, rows)
    write_csv(SUMMARY_CSV, summarize(rows))
    plot_alignment(rows, PLOT_PATH)
    plot_risk_scatter(rows, ROOT / "results" / "guided_ablation_extended.csv", SCATTER_PATH)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(f"Wrote summary to {SUMMARY_CSV}")
    print(f"Wrote plots to {PLOT_PATH} and {SCATTER_PATH}")


if __name__ == "__main__":
    main()
