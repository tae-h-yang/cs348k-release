"""Audit segment-critic selection against full-critic selection.

Original WC-K data saved only the selected clip per K, not every candidate.
This script reruns a small K=8 audit set, saves every candidate, scores each
with both OnlineSegmentCritic and the full PhysicalAwarenessCritic, and reports
winner agreement. It directly checks whether the paper can say "full critic
best-of-K" or must say "segment critic proposes, full critic evaluates."
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

from generate_guided import MOTION_CONFIGS, generate_clip
from physics_eval.online_critic import OnlineSegmentCritic
from physics_eval.physaware import PhysicalAwarenessCritic

MOTIONBRICKS_DIR = ROOT.parent / "GR00T-WholeBodyControl" / "motionbricks"
OUT_DIR = ROOT / "data" / "candidate_audit"
OUT_CSV = ROOT / "results" / "candidate_audit.csv"
OUT_SUMMARY = ROOT / "results" / "candidate_audit_summary.csv"

DEFAULT_MODES = ["walk", "slow_walk", "walk_gun", "walk_happy_dance", "hand_crawling"]


def load_agent():
    original = Path.cwd()
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
    agent = navigation_demo(mb_args)
    os.chdir(original)
    return agent


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


def summarize(rows: list[dict]):
    groups = {}
    for row in rows:
        groups.setdefault(row["clip"], []).append(row)
    summary_rows = []
    for clip, group in groups.items():
        seg_best = min(group, key=lambda r: float(r["segment_risk"]))
        full_best = min(group, key=lambda r: float(r["full_risk"]))
        summary_rows.append({
            "clip": clip,
            "type": seg_best["type"],
            "segment_best_k": seg_best["candidate_k"],
            "segment_best_full_risk": seg_best["full_risk"],
            "full_best_k": full_best["candidate_k"],
            "full_best_full_risk": full_best["full_risk"],
            "winner_agrees": seg_best["candidate_k"] == full_best["candidate_k"],
            "full_risk_gap": float(seg_best["full_risk"]) - float(full_best["full_risk"]),
        })
    if summary_rows:
        summary_rows.append({
            "clip": "OVERALL",
            "type": "all",
            "segment_best_k": "",
            "segment_best_full_risk": "",
            "full_best_k": "",
            "full_best_full_risk": "",
            "winner_agrees": sum(r["winner_agrees"] for r in summary_rows),
            "full_risk_gap": float(np.mean([r["full_risk_gap"] for r in summary_rows])),
            "n": len(summary_rows),
        })
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="*", default=DEFAULT_MODES)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    configs = [c for c in MOTION_CONFIGS if c["mode"] in args.modes]
    agent = load_agent()
    seg_critic = OnlineSegmentCritic()
    full_critic = PhysicalAwarenessCritic()
    rows = []

    for cfg in configs:
        for seed_idx in range(args.seeds):
            clip = f"{cfg['mode']}_seed{seed_idx}"
            base_seed = seed_idx * 1000
            for k in range(args.K):
                path = OUT_DIR / f"{clip}_cand{k}.npy"
                if path.exists() and not args.overwrite:
                    qpos = np.load(path)
                    print(f"Reusing {path}")
                else:
                    qpos = generate_clip(
                        agent,
                        cfg["mode"],
                        cfg["n_frames"],
                        seed=base_seed + k * 137,
                        stochastic=(k > 0),
                    )
                    np.save(path, qpos.astype(np.float32))
                segment_risk = seg_critic.score_segment(qpos)
                report, _ = full_critic.score(qpos, clip, cfg["type"], variant=f"candidate_{k}")
                rows.append({
                    "clip": clip,
                    "mode": cfg["mode"],
                    "type": cfg["type"],
                    "seed_idx": seed_idx,
                    "candidate_k": k,
                    "segment_risk": segment_risk,
                    "full_risk": report.risk_score,
                    "action": report.recommended_action,
                    "path": str(path),
                })
                print(f"{clip} cand={k} seg={segment_risk:.2f} full={report.risk_score:.2f}")
                write_csv(OUT_CSV, rows)
                write_csv(OUT_SUMMARY, summarize(rows))


if __name__ == "__main__":
    main()
