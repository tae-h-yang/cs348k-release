"""Extended 100+ clip best-of-K ablation for MotionBricks.

The earlier paper results use 13 modes x 3 seeds = 39 clip identities. This
script expands the supported MotionBricks mode set to all 15 exposed G1 demo
modes and defaults to 7 seeds per mode, yielding 105 distinct style/seed
identities. For each identity it evaluates K=1 and K=8.

The script is resumable: generated K outputs are saved under
data/guided_ablation_extended/, and existing outputs are scored/reused unless
--overwrite is provided.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from physics_eval.online_critic import OnlineSegmentCritic
from physics_eval.physaware import PhysicalAwarenessCritic
from generate_guided import generate_clip

MOTIONBRICKS_DIR = ROOT.parent / "GR00T-WholeBodyControl" / "motionbricks"
OUT_DIR = ROOT / "data" / "guided_ablation_extended"
RESULTS_CSV = ROOT / "results" / "guided_ablation_extended.csv"
SUMMARY_CSV = ROOT / "results" / "guided_ablation_extended_summary.csv"

EXTENDED_CONFIGS = [
    {"mode": "idle",             "n_frames": 150, "type": "static"},
    {"mode": "walk",             "n_frames": 200, "type": "locomotion"},
    {"mode": "slow_walk",        "n_frames": 200, "type": "locomotion"},
    {"mode": "stealth_walk",     "n_frames": 200, "type": "locomotion"},
    {"mode": "injured_walk",     "n_frames": 200, "type": "locomotion"},
    {"mode": "walk_zombie",      "n_frames": 200, "type": "locomotion"},
    {"mode": "walk_stealth",     "n_frames": 180, "type": "locomotion"},
    {"mode": "walk_left",        "n_frames": 180, "type": "locomotion"},
    {"mode": "walk_right",       "n_frames": 180, "type": "locomotion"},
    {"mode": "walk_boxing",      "n_frames": 180, "type": "expressive"},
    {"mode": "walk_happy_dance", "n_frames": 180, "type": "expressive"},
    {"mode": "walk_gun",         "n_frames": 180, "type": "expressive"},
    {"mode": "walk_scared",      "n_frames": 180, "type": "expressive"},
    {"mode": "hand_crawling",    "n_frames": 150, "type": "whole_body"},
    {"mode": "elbow_crawling",   "n_frames": 150, "type": "whole_body"},
]


def load_motionbricks_agent():
    original_dir = Path.cwd()
    os.chdir(MOTIONBRICKS_DIR)
    import argparse as ap
    from motionbricks.motion_backbone.demo.utils import navigation_demo

    mb_args = ap.Namespace(
        explicit_dataset_folder=None, reprocess_clips=0, controller="random",
        lookat_movement_direction=0, has_viewer=0, pre_filter_qpos=1,
        source_root_realignment=1, target_root_realignment=1,
        force_canonicalization=1, skip_ending_target_cond=0,
        random_speed_scale=0, speed_scale=[0.8, 1.2], generate_dt=2.0,
        max_steps=10000, random_seed=42, num_runs=1, use_qpos=1,
        planner="default", allowed_mode=None, clips="G1",
        return_model_configs=True, return_dataloader=True,
        recording_dir=None, EXP="default",
    )
    print("Loading MotionBricks model...")
    demo_agent = navigation_demo(mb_args)
    os.chdir(original_dir)
    return demo_agent


def generate_best(
    demo_agent,
    cfg: dict,
    seed_idx: int,
    K: int,
    segment_critic: OnlineSegmentCritic,
):
    clip = f"{cfg['mode']}_seed{seed_idx}"
    base_seed = seed_idx * 1000
    best_qpos = None
    best_seg_risk = float("inf")
    all_seg_risks = []

    for k in range(K):
        seed_k = base_seed + k * 137
        stochastic = k > 0
        qpos = generate_clip(
            demo_agent,
            cfg["mode"],
            cfg["n_frames"],
            seed=seed_k,
            stochastic=stochastic,
        )
        seg_risk = float(segment_critic.score_segment(qpos))
        all_seg_risks.append(seg_risk)
        print(
            f"    {clip} K={K} candidate={k:02d} seed={seed_k:<5d} "
            f"stoch={stochastic} seg_risk={seg_risk:.2f}"
        )
        if seg_risk < best_seg_risk:
            best_seg_risk = seg_risk
            best_qpos = qpos

    return best_qpos, best_seg_risk, all_seg_risks


def write_csv(path: Path, rows: list[dict]):
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
    print(f"Saved: {path}")


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for k in sorted({int(r["K"]) for r in rows}):
        group = [r for r in rows if int(r["K"]) == k and np.isfinite(float(r["full_risk"]))]
        risks = np.array([float(r["full_risk"]) for r in group])
        out.append({
            "K": k,
            "n": len(group),
            "mean_risk": float(np.mean(risks)) if len(risks) else float("nan"),
            "median_risk": float(np.median(risks)) if len(risks) else float("nan"),
            "accept": sum(r["action"] == "accept" for r in group),
            "repair_or_rerank": sum(r["action"] == "repair_or_rerank" for r in group),
            "reject_or_regenerate": sum(r["action"] == "reject_or_regenerate" for r in group),
        })
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], {})[int(r["K"])] = r
    paired = [v for v in by_clip.values() if 1 in v and 8 in v]
    if paired:
        k1 = np.array([float(p[1]["full_risk"]) for p in paired])
        k8 = np.array([float(p[8]["full_risk"]) for p in paired])
        finite_mask = k1 > 1e-6
        out.append({
            "K": "K8_vs_K1_paired",
            "n": len(paired),
            "mean_risk": float(np.mean(k8 - k1)),
            "median_risk": float(np.median(k8 - k1)),
            "aggregate_reduction_pct": float(100.0 * (np.mean(k1) - np.mean(k8)) / max(np.mean(k1), 1e-8)),
            "mean_per_clip_reduction_pct": (
                float(np.mean(100.0 * (k1[finite_mask] - k8[finite_mask]) / k1[finite_mask]))
                if np.any(finite_mask) else float("nan")
            ),
            "median_per_clip_reduction_pct": (
                float(np.median(100.0 * (k1[finite_mask] - k8[finite_mask]) / k1[finite_mask]))
                if np.any(finite_mask) else float("nan")
            ),
            "finite_baseline_count": int(np.sum(finite_mask)),
            "improved_count": int(np.sum(k8 < k1)),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--k_values", type=int, nargs="*", default=[1, 8])
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    configs = EXTENDED_CONFIGS
    if args.modes:
        configs = [c for c in configs if c["mode"] in args.modes]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    demo_agent = load_motionbricks_agent()
    segment_critic = OnlineSegmentCritic()
    full_critic = PhysicalAwarenessCritic()
    rows = []

    for cfg in configs:
        for seed_idx in range(args.seeds):
            clip = f"{cfg['mode']}_seed{seed_idx}"
            for K in args.k_values:
                out_path = OUT_DIR / f"{clip}_K{K}.npy"
                if out_path.exists() and not args.overwrite:
                    qpos = np.load(out_path)
                    seg_risks = []
                    best_seg_risk = float("nan")
                    print(f"Reusing {out_path}")
                else:
                    print(f"\n[{clip}] type={cfg['type']} K={K}")
                    qpos, best_seg_risk, seg_risks = generate_best(
                        demo_agent, cfg, seed_idx, K, segment_critic
                    )
                    np.save(out_path, qpos.astype(np.float32))

                report, _ = full_critic.score(qpos, clip, cfg["type"], variant=f"K{K}")
                rows.append({
                    "clip": clip,
                    "mode": cfg["mode"],
                    "type": cfg["type"],
                    "seed_idx": seed_idx,
                    "K": K,
                    "frames": len(qpos),
                    "full_risk": report.risk_score,
                    "p95_torque_limit_ratio": report.p95_torque_limit_ratio,
                    "action": report.recommended_action,
                    "best_seg_risk": best_seg_risk,
                    "all_seg_risks": ";".join(f"{r:.3f}" for r in seg_risks),
                    "path": str(out_path),
                })
                write_csv(RESULTS_CSV, rows)
                write_csv(SUMMARY_CSV, summarize(rows))

    write_csv(RESULTS_CSV, rows)
    write_csv(SUMMARY_CSV, summarize(rows))


if __name__ == "__main__":
    main()
