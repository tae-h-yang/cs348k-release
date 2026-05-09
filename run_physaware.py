"""
Run the physical-awareness critic and repair layer on MotionBricks clips.

This is the project-facing experiment: given kinematic plans, score physical
risk, try deterministic test-time repairs, and report before/after changes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.render import render_reference_video
from analysis.visualize import RESULTS_DIR
from physics_eval.physaware import PhysicalAwarenessCritic, repair_candidates
from physics_eval.simulator import PhysicsSimulator
from run_eval import load_motion_dir

DEFAULT_EXAMPLES = (
    "idle_seed0",
    "walk_seed0",
    "walk_seed2",
    "slow_walk_seed0",
    "stealth_walk_seed0",
    "walk_boxing_seed0",
    "walk_gun_seed1",
    "walk_happy_dance_seed1",
    "walk_scared_seed1",
    "hand_crawling_seed2",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/motionbricks")
    parser.add_argument("--examples", nargs="*", default=list(DEFAULT_EXAMPLES))
    parser.add_argument("--all", action="store_true",
                        help="Run on every valid clip in data_dir instead of the default 10 examples")
    parser.add_argument("--render", action="store_true",
                        help="Render original and best-repaired reference videos for selected examples")
    args = parser.parse_args()

    motions = {name: (typ, qpos) for name, typ, qpos in load_motion_dir(Path(args.data_dir))}
    if args.all:
        args.examples = sorted(motions)
    missing = [name for name in args.examples if name not in motions]
    if missing:
        raise ValueError(f"Missing requested examples: {missing}")

    critic = PhysicalAwarenessCritic()
    sim = PhysicsSimulator()
    all_rows = []
    best_rows = []

    repaired_dir = Path(args.data_dir) / "physaware_repaired"
    repaired_dir.mkdir(parents=True, exist_ok=True)

    for name in args.examples:
        motion_type, qpos = motions[name]
        print(f"[{name}] {motion_type}")
        reports = []
        candidates = repair_candidates(qpos)
        for variant, candidate in candidates.items():
            report, _ = critic.score(candidate, name, motion_type, variant=variant)
            reports.append((report, candidate))
            all_rows.append(report.summary())
            print(f"  {variant:18s} risk={report.risk_score:6.2f} "
                  f"tau95={report.p95_torque_limit_ratio:5.2f} "
                  f"rootF95={report.p95_root_force_N:8.0f} "
                  f"action={report.recommended_action}")

        original = reports[0][0]
        best_report, best_qpos = min(reports, key=lambda item: item[0].risk_score)
        improvement = 100.0 * (original.risk_score - best_report.risk_score) / max(original.risk_score, 1e-8)
        best_summary = best_report.summary()
        best_summary["original_risk_score"] = original.risk_score
        best_summary["risk_reduction_pct"] = improvement
        best_summary["selected_variant"] = best_report.variant
        best_rows.append(best_summary)

        out_path = repaired_dir / f"{name}_{best_report.variant}.npy"
        np.save(out_path, best_qpos.astype(np.float32))
        print(f"  selected={best_report.variant} reduction={improvement:.1f}% -> {out_path}")

        if args.render:
            render_reference_video(sim, qpos, clip_name=f"{name}_original")
            render_reference_video(sim, best_qpos, clip_name=f"{name}_{best_report.variant}")

    write_csv(RESULTS_DIR / "physaware_candidates.csv", all_rows)
    write_csv(RESULTS_DIR / "physaware_best.csv", best_rows)
    write_csv(RESULTS_DIR / "physaware_summary.csv", summarize_overall(best_rows))
    write_csv(RESULTS_DIR / "physaware_by_type.csv", summarize_by_type(best_rows))
    write_csv(RESULTS_DIR / "physaware_variant_baselines.csv", summarize_variants(all_rows))
    write_csv(RESULTS_DIR / "physaware_action_counts.csv", summarize_actions(best_rows))
    plot_physaware_results(best_rows)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def summarize_by_type(rows: list[dict]) -> list[dict]:
    out = []
    for motion_type in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == motion_type]
        original = np.array([float(r["original_risk_score"]) for r in group])
        selected = np.array([float(r["risk_score"]) for r in group])
        reduction = np.array([float(r["risk_reduction_pct"]) for r in group])
        out.append({
            "type": motion_type,
            "n": len(group),
            "mean_original_risk": float(np.mean(original)),
            "mean_selected_risk": float(np.mean(selected)),
            "aggregate_reduction_pct": float(100.0 * (np.mean(original) - np.mean(selected)) / max(np.mean(original), 1e-8)),
            "mean_per_clip_reduction_pct": float(np.mean(reduction)),
            "median_per_clip_reduction_pct": float(np.median(reduction)),
            "improved_count": int(np.sum(reduction > 1e-8)),
            "mean_reduction_pct_ci_low": bootstrap_mean_ci(reduction)[0],
            "mean_reduction_pct_ci_high": bootstrap_mean_ci(reduction)[1],
        })
    return out


def summarize_overall(rows: list[dict]) -> list[dict]:
    original = np.array([float(r["original_risk_score"]) for r in rows])
    selected = np.array([float(r["risk_score"]) for r in rows])
    reduction = np.array([float(r["risk_reduction_pct"]) for r in rows])
    return [{
        "n": len(rows),
        "mean_original_risk": float(np.mean(original)),
        "mean_selected_risk": float(np.mean(selected)),
        "mean_absolute_reduction": float(np.mean(original - selected)),
        "aggregate_reduction_pct": float(
            100.0 * (np.mean(original) - np.mean(selected)) / max(np.mean(original), 1e-8)
        ),
        "mean_per_clip_reduction_pct": float(np.mean(reduction)),
        "median_per_clip_reduction_pct": float(np.median(reduction)),
        "improved_count": int(np.sum(reduction > 1e-8)),
        "mean_reduction_pct_ci_low": bootstrap_mean_ci(reduction)[0],
        "mean_reduction_pct_ci_high": bootstrap_mean_ci(reduction)[1],
    }]


def summarize_variants(rows: list[dict]) -> list[dict]:
    out = []
    for variant in sorted({r["variant"] for r in rows}):
        group = [r for r in rows if r["variant"] == variant]
        risk = np.array([float(r["risk_score"]) for r in group])
        torque = np.array([float(r["p95_torque_limit_ratio"]) for r in group])
        out.append({
            "variant": variant,
            "n": len(group),
            "mean_risk": float(np.mean(risk)),
            "median_risk": float(np.median(risk)),
            "mean_p95_torque_limit_ratio": float(np.mean(torque)),
            "accept_count": sum(r["recommended_action"] == "accept" for r in group),
            "repair_or_rerank_count": sum(r["recommended_action"] == "repair_or_rerank" for r in group),
            "reject_or_regenerate_count": sum(r["recommended_action"] == "reject_or_regenerate" for r in group),
        })
    return out


def summarize_actions(rows: list[dict]) -> list[dict]:
    original_actions = []
    selected_actions = []
    candidate_rows = list(csv.DictReader(open(RESULTS_DIR / "physaware_candidates.csv")))
    original_by_clip = {
        r["clip"]: r["recommended_action"]
        for r in candidate_rows if r["variant"] == "original"
    }
    for r in rows:
        original_actions.append(original_by_clip[r["clip"]])
        selected_actions.append(r["recommended_action"])
    actions = ["accept", "repair_or_rerank", "reject_or_regenerate"]
    return [{
        "stage": "original",
        **{a: original_actions.count(a) for a in actions},
        "n": len(original_actions),
    }, {
        "stage": "selected",
        **{a: selected_actions.count(a) for a in actions},
        "n": len(selected_actions),
    }]


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 7) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = np.mean(samples, axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def plot_physaware_results(rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["clip"] for r in rows]
    original = np.array([float(r["original_risk_score"]) for r in rows])
    repaired = np.array([float(r["risk_score"]) for r in rows])
    selected = [r["selected_variant"] for r in rows]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 5))
    ax.bar(x - 0.18, original, width=0.36, label="original", color="#90A4AE", edgecolor="black")
    ax.bar(x + 0.18, repaired, width=0.36, label="selected repair", color="#26A69A", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Heuristic Feasibility Risk Score")
    ax.set_title("Critic-Selected Retiming/Smoothing Variants")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    for i, label in enumerate(selected):
        ax.text(i, repaired[i], label, ha="center", va="bottom", fontsize=7, rotation=90)
    fig.tight_layout()
    path = RESULTS_DIR / "physaware_before_after_risk.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    reductions = [float(r["risk_reduction_pct"]) for r in rows]
    ax.hist(reductions, bins=8, color="#5C6BC0", edgecolor="black")
    ax.set_xlabel("Risk Reduction (%)")
    ax.set_ylabel("Number of Clips")
    ax.set_title(f"Critic-Selected Risk Reduction Across {len(rows)} Clips")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = RESULTS_DIR / "physaware_risk_reduction_hist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
