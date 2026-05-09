"""
Sweep PD gain scales on representative MotionBricks clips.

This is a reproducible way to justify the default weak-controller baseline:
we tune only for qualitative survival time, while inverse dynamics remains the
main quantitative result.

Usage:
    python scripts/sweep_pd_gains.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_eval.pd_controller import ACTUATOR_FORCE_LIMITS, KD, KP, PDController
from physics_eval.simulator import PhysicsSimulator

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_CLIPS = (
    "idle_seed0",
    "walk_seed2",
    "walk_gun_seed1",
    "hand_crawling_seed2",
)


def evaluate_config(clip_paths: list[Path], kp_scale: float, kd_scale: float) -> dict:
    sim = PhysicsSimulator(pd_kp_scale=kp_scale, pd_kd_scale=kd_scale)
    # Reassign explicitly so this script documents the controller equation source.
    sim.controller = PDController(
        kp=KP * kp_scale,
        kd=KD * kd_scale,
        force_limits=ACTUATOR_FORCE_LIMITS,
    )

    fall_frames = []
    tracking = []
    survived = 0
    for path in clip_paths:
        seq = np.load(path)
        metrics = sim.evaluate_clip(seq, clip_name=path.stem, motion_type="sweep")
        fall_frame = metrics.time_to_fall if metrics.fell else metrics.n_frames
        fall_frames.append(fall_frame)
        tracking.append(metrics.mean_tracking_rmse)
        survived += int(not metrics.fell)

    return {
        "kp_scale": kp_scale,
        "kd_scale": kd_scale,
        "mean_fall_frame": float(np.mean(fall_frames)),
        "min_fall_frame": int(np.min(fall_frames)),
        "max_fall_frame": int(np.max(fall_frames)),
        "mean_tracking_rmse": float(np.mean(tracking)),
        "survived_full_clip_count": survived,
    }


def main():
    data_dir = ROOT / "data" / "motionbricks"
    clip_paths = [data_dir / f"{name}.npy" for name in DEFAULT_CLIPS]
    missing = [str(p) for p in clip_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sweep clips: {missing}")

    kp_scales = np.round(np.linspace(0.1, 1.0, 10), 2)
    kd_scales = np.round(np.linspace(0.5, 2.75, 10), 2)

    rows = []
    total = len(kp_scales) * len(kd_scales)
    i = 0
    for kp in kp_scales:
        for kd in kd_scales:
            i += 1
            print(f"[{i:03d}/{total}] kp={kp:.2f} kd={kd:.2f}")
            rows.append(evaluate_config(clip_paths, float(kp), float(kd)))

    csv_path = RESULTS_DIR / "pd_gain_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")

    plot_path = RESULTS_DIR / "pd_gain_sweep_heatmap.png"
    plot_heatmap(rows, kp_scales, kd_scales, plot_path)
    print(f"Saved: {plot_path}")

    best = sorted(rows, key=lambda r: (r["mean_fall_frame"], -r["mean_tracking_rmse"]), reverse=True)[:5]
    print("Top configs by mean fall frame:")
    for row in best:
        print(row)


def plot_heatmap(rows: list[dict], kp_scales: np.ndarray, kd_scales: np.ndarray, path: Path):
    grid = np.zeros((len(kp_scales), len(kd_scales)), dtype=np.float64)
    for row in rows:
        i = int(np.where(kp_scales == row["kp_scale"])[0][0])
        j = int(np.where(kd_scales == row["kd_scale"])[0][0])
        grid[i, j] = row["mean_fall_frame"] / 30.0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="Mean time-to-fall on 4 demo clips (s)")
    ax.set_xticks(range(len(kd_scales)))
    ax.set_xticklabels([f"{x:.2f}" for x in kd_scales], rotation=45, ha="right")
    ax.set_yticks(range(len(kp_scales)))
    ax.set_yticklabels([f"{x:.2f}" for x in kp_scales])
    ax.set_xlabel("KD scale")
    ax.set_ylabel("KP scale")
    ax.set_title("PD Gain Sweep: Qualitative Baseline Stability")

    # Mark the repo default.
    default_i = int(np.where(kp_scales == 0.5)[0][0])
    default_j = int(np.where(kd_scales == 1.0)[0][0])
    ax.scatter(default_j, default_i, marker="*", s=180, c="white",
               edgecolors="black", linewidths=1.0, label="repo default")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
