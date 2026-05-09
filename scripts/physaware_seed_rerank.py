"""Group MotionBricks seeds by style and show physical-awareness reranking."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from physics_eval.physaware import PhysicalAwarenessCritic
from run_eval import load_motion_dir

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def style_name(clip_name: str) -> str:
    return re.sub(r"_seed\d+$", "", clip_name)


def main():
    critic = PhysicalAwarenessCritic()
    groups: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for name, motion_type, qpos in load_motion_dir(ROOT / "data" / "motionbricks"):
        groups.setdefault(style_name(name), []).append((name, motion_type, qpos))

    rows = []
    for style, clips in sorted(groups.items()):
        if len(clips) < 2:
            continue
        reports = []
        for name, motion_type, qpos in clips:
            report, _ = critic.score(qpos, clip_name=name, motion_type=motion_type)
            reports.append(report)
        original_mean = float(np.mean([r.risk_score for r in reports]))
        best = min(reports, key=lambda r: r.risk_score)
        action_counts = {
            action: sum(r.recommended_action == action for r in reports)
            for action in ("accept", "repair_or_rerank", "reject_or_regenerate")
        }
        rows.append({
            "style": style,
            "type": best.motion_type,
            "n_seeds": len(reports),
            "mean_seed_risk": original_mean,
            "best_seed": best.clip_name,
            "best_seed_risk": best.risk_score,
            "risk_reduction_vs_mean_pct": 100.0 * (original_mean - best.risk_score) / max(original_mean, 1e-8),
            "best_action": best.recommended_action,
            "random_accept_probability": action_counts["accept"] / len(reports),
            "random_repair_probability": action_counts["repair_or_rerank"] / len(reports),
            "random_reject_probability": action_counts["reject_or_regenerate"] / len(reports),
        })
        print(style, "mean", f"{original_mean:.2f}", "best", best.clip_name, f"{best.risk_score:.2f}")

    path = RESULTS_DIR / "physaware_seed_rerank.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")

    summary = summarize(rows)
    summary_path = RESULTS_DIR / "physaware_seed_rerank_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved: {summary_path}")

    plot(rows)


def summarize(rows: list[dict]) -> list[dict]:
    mean_seed = np.array([float(r["mean_seed_risk"]) for r in rows])
    best_seed = np.array([float(r["best_seed_risk"]) for r in rows])
    reduction = np.array([float(r["risk_reduction_vs_mean_pct"]) for r in rows])
    return [{
        "n_styles": len(rows),
        "random_expected_mean_risk": float(np.mean(mean_seed)),
        "critic_selected_mean_risk": float(np.mean(best_seed)),
        "aggregate_reduction_pct": float(
            100.0 * (np.mean(mean_seed) - np.mean(best_seed)) / max(np.mean(mean_seed), 1e-8)
        ),
        "mean_per_style_reduction_pct": float(np.mean(reduction)),
        "median_per_style_reduction_pct": float(np.median(reduction)),
        "random_expected_accept_styles": float(np.sum([float(r["random_accept_probability"]) for r in rows])),
        "random_expected_repair_styles": float(np.sum([float(r["random_repair_probability"]) for r in rows])),
        "random_expected_reject_styles": float(np.sum([float(r["random_reject_probability"]) for r in rows])),
        "critic_selected_accept_styles": sum(r["best_action"] == "accept" for r in rows),
        "critic_selected_repair_styles": sum(r["best_action"] == "repair_or_rerank" for r in rows),
        "critic_selected_reject_styles": sum(r["best_action"] == "reject_or_regenerate" for r in rows),
    }]


def plot(rows: list[dict]):
    styles = [r["style"] for r in rows]
    mean_risk = np.array([float(r["mean_seed_risk"]) for r in rows])
    best_risk = np.array([float(r["best_seed_risk"]) for r in rows])

    x = np.arange(len(styles))
    fig, ax = plt.subplots(figsize=(max(10, len(styles) * 0.75), 5))
    ax.bar(x - 0.18, mean_risk, width=0.36, color="#B0BEC5", edgecolor="black",
           label="mean over generated seeds")
    ax.bar(x + 0.18, best_risk, width=0.36, color="#42A5F5", edgecolor="black",
           label="best seed selected by critic")
    ax.set_xticks(x)
    ax.set_xticklabels(styles, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Heuristic Feasibility Risk Score")
    ax.set_title("PhysAware Seed Reranking: Lowest-Risk Candidate Selection")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    path = RESULTS_DIR / "physaware_seed_rerank.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
