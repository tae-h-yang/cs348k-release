"""Measure how much K=8 changes semantics relative to K=1.

The physics critic can lower inverse-dynamics risk by selecting a different
candidate. This script quantifies the tradeoff instead of relying on visual
claims: root displacement, path length, average speed, joint trajectory RMSE,
duration difference, and risk reduction.
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
DATA_DIR = ROOT / "data" / "guided_ablation"
GUIDED_CSV = ROOT / "results" / "guided_ablation_full.csv"
OUT_CSV = ROOT / "results" / "semantic_preservation.csv"
OUT_SUMMARY = ROOT / "results" / "semantic_preservation_summary.csv"
OUT_PLOT = ROOT / "results" / "semantic_preservation.png"


def path_metrics(qpos: np.ndarray) -> dict:
    xy = qpos[:, :2]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    path_length = float(np.sum(step))
    duration_s = len(qpos) / 30.0
    return {
        "frames": len(qpos),
        "duration_s": duration_s,
        "root_displacement_m": displacement,
        "root_path_length_m": path_length,
        "avg_speed_mps": path_length / max(duration_s, 1e-8),
    }


def load_rows():
    with open(GUIDED_CSV) as f:
        rows = list(csv.DictReader(f))
    by_clip = {}
    for row in rows:
        k = int(row["K"])
        if k in (1, 8):
            by_clip.setdefault(row["clip"], {})[k] = row
    return by_clip


def aligned_joint_rmse(q1: np.ndarray, q8: np.ndarray) -> float:
    n = min(len(q1), len(q8))
    return float(np.sqrt(np.mean((q1[:n, 7:] - q8[:n, 7:]) ** 2)))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for mtype in ["all", *sorted({r["type"] for r in rows})]:
        group = rows if mtype == "all" else [r for r in rows if r["type"] == mtype]
        if not group:
            continue
        risk1 = np.array([r["risk_K1"] for r in group], dtype=np.float64)
        risk8 = np.array([r["risk_K8"] for r in group], dtype=np.float64)
        valid_pct = [r["risk_reduction_pct"] for r in group if np.isfinite(r["risk_reduction_pct"])]
        out.append({
            "type": mtype,
            "n": len(group),
            "aggregate_risk_reduction_pct": float(
                100.0 * (np.mean(risk1) - np.mean(risk8)) / max(np.mean(risk1), 1e-8)
            ),
            "mean_risk_reduction_pct_finite_baseline": float(np.mean(valid_pct)) if valid_pct else float("nan"),
            "median_displacement_ratio": float(np.median([r["displacement_ratio"] for r in group])),
            "median_path_length_ratio": float(np.median([r["path_length_ratio"] for r in group])),
            "median_speed_ratio": float(np.median([r["speed_ratio"] for r in group])),
            "median_joint_rmse_rad": float(np.median([r["joint_rmse_rad"] for r in group])),
            "changed_displacement_count": int(sum(
                (r["displacement_ratio"] < 0.5 or r["displacement_ratio"] > 1.5)
                for r in group
            )),
        })
    return out


def plot(rows: list[dict]):
    risk = np.array([r["risk_reduction_pct"] for r in rows])
    disp = np.array([r["displacement_ratio"] for r in rows])
    types = [r["type"] for r in rows]
    colors = {
        "static": "#7E57C2",
        "locomotion": "#42A5F5",
        "expressive": "#26A69A",
        "whole_body": "#EF5350",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for mtype in sorted(set(types)):
        idx = [i for i, t in enumerate(types) if t == mtype]
        ax.scatter(disp[idx], risk[idx], label=mtype, s=45,
                   color=colors.get(mtype, "gray"), edgecolor="black", alpha=0.8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.axvspan(0.5, 1.5, color="gray", alpha=0.12, label="0.5x-1.5x displacement")
    ax.set_xlabel("K=8 / K=1 Root Displacement Ratio")
    ax.set_ylabel("Risk Reduction (%)")
    ax.set_title("Risk Reduction vs. Motion-Scale Preservation")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PLOT}")


def main():
    rows = []
    for clip, meta in load_rows().items():
        if 1 not in meta or 8 not in meta:
            continue
        q1_path = DATA_DIR / f"{clip}_K1.npy"
        q8_path = DATA_DIR / f"{clip}_K8.npy"
        if not q1_path.exists() or not q8_path.exists():
            continue
        q1 = np.load(q1_path)
        q8 = np.load(q8_path)
        m1 = path_metrics(q1)
        m8 = path_metrics(q8)
        risk1 = float(meta[1]["full_risk"])
        risk8 = float(meta[8]["full_risk"])
        risk_reduction_pct = (
            100.0 * (risk1 - risk8) / risk1
            if risk1 > 1e-6 else float("nan")
        )
        rows.append({
            "clip": clip,
            "type": meta[1]["type"],
            "risk_K1": risk1,
            "risk_K8": risk8,
            "risk_reduction_pct": risk_reduction_pct,
            "displacement_K1_m": m1["root_displacement_m"],
            "displacement_K8_m": m8["root_displacement_m"],
            "displacement_ratio": m8["root_displacement_m"] / max(m1["root_displacement_m"], 1e-8),
            "path_length_K1_m": m1["root_path_length_m"],
            "path_length_K8_m": m8["root_path_length_m"],
            "path_length_ratio": m8["root_path_length_m"] / max(m1["root_path_length_m"], 1e-8),
            "speed_ratio": m8["avg_speed_mps"] / max(m1["avg_speed_mps"], 1e-8),
            "joint_rmse_rad": aligned_joint_rmse(q1, q8),
            "frames_K1": m1["frames"],
            "frames_K8": m8["frames"],
        })

    write_csv(OUT_CSV, rows)
    write_csv(OUT_SUMMARY, summarize(rows))
    plot(rows)


if __name__ == "__main__":
    main()
