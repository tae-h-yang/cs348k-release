"""
Diversity source ablation: addresses Reviewer R4.

Compares three strategies for generating K=4 candidate clips:
  A) seed-only:    argmax sampling, 4 different seeds (seed+0, +137, +274, +411)
  B) gumbel-only:  stochastic Gumbel sampling, same base seed × 4 draws
  C) combined:     k=0 argmax + k=1..3 stochastic with offset seeds [CURRENT METHOD]

This isolates the contribution of each diversity source to the 59% risk reduction.

Runs on all 13 styles × 3 seeds = 39 clips at K=4.
Results saved to: results/diversity_ablation.csv
"""
import sys, os, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MOTIONBRICKS_DIR = Path(__file__).parent.parent.parent / "GR00T-WholeBodyControl" / "motionbricks"
os.chdir(MOTIONBRICKS_DIR)

import argparse
from motionbricks.motion_backbone.demo.utils import navigation_demo

mb_args = argparse.Namespace(
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
os.chdir(Path(__file__).parent.parent)

from physics_eval.online_critic import OnlineSegmentCritic
from physics_eval.physaware import PhysicalAwarenessCritic

import importlib.util
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

guided_mod = _load("generate_guided", "generate_guided.py")
generate_clip = guided_mod.generate_clip

segment_critic = OnlineSegmentCritic()
full_critic = PhysicalAwarenessCritic()

ALL_CONFIGS = [
    {"mode": "idle",             "n_frames": 150, "type": "static"},
    {"mode": "walk",             "n_frames": 200, "type": "locomotion"},
    {"mode": "slow_walk",        "n_frames": 200, "type": "locomotion"},
    {"mode": "stealth_walk",     "n_frames": 200, "type": "locomotion"},
    {"mode": "injured_walk",     "n_frames": 200, "type": "locomotion"},
    {"mode": "walk_zombie",      "n_frames": 200, "type": "locomotion"},
    {"mode": "walk_stealth",     "n_frames": 180, "type": "locomotion"},
    {"mode": "walk_boxing",      "n_frames": 180, "type": "expressive"},
    {"mode": "walk_happy_dance", "n_frames": 180, "type": "expressive"},
    {"mode": "walk_gun",         "n_frames": 180, "type": "expressive"},
    {"mode": "walk_scared",      "n_frames": 180, "type": "expressive"},
    {"mode": "hand_crawling",    "n_frames": 150, "type": "whole_body"},
    {"mode": "elbow_crawling",   "n_frames": 150, "type": "whole_body"},
]
N_SEEDS = 3
K = 4

SEED_OFFSETS = [0, 137, 274, 411]  # prime-spaced, same as current WC-K

def run_strategy(strategy, mode, n_frames, base_seed, K=4):
    """
    strategy: "seed_only" | "gumbel_only" | "combined"
    Returns (best_qpos, best_risk, all_risks)
    """
    candidates = []
    risks = []

    for k in range(K):
        if strategy == "seed_only":
            # argmax (deterministic), K different seeds
            seed_k = base_seed + SEED_OFFSETS[k]
            stochastic = False
        elif strategy == "gumbel_only":
            # Gumbel stochastic, same base seed (pure Gumbel variance)
            seed_k = base_seed
            stochastic = True
        elif strategy == "combined":
            # k=0: argmax base_seed; k>0: stochastic offset seeds
            seed_k = base_seed + SEED_OFFSETS[k]
            stochastic = (k > 0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        try:
            qpos_k = generate_clip(demo_agent, mode, n_frames, seed=seed_k, stochastic=stochastic)
            risk_k = segment_critic.score_segment(qpos_k)
        except Exception as e:
            print(f"      k={k} ({strategy}) FAILED: {e}")
            qpos_k = None
            risk_k = float("inf")

        candidates.append(qpos_k)
        risks.append(risk_k)

    valid = [(i, r) for i, r in enumerate(risks)
             if candidates[i] is not None and np.isfinite(r)]
    if not valid:
        return None, float("nan"), risks

    best_k, best_risk = min(valid, key=lambda x: x[1])
    return candidates[best_k], best_risk, risks


rows = []
out_base = Path("data/diversity_ablation")
out_base.mkdir(parents=True, exist_ok=True)

for cfg in ALL_CONFIGS:
    mode, n_frames, mtype = cfg["mode"], cfg["n_frames"], cfg["type"]
    for seed_idx in range(N_SEEDS):
        base_seed = seed_idx * 1000
        clip_name = f"{mode}_seed{seed_idx}"
        print(f"\n[{clip_name}]")

        row = {"clip": clip_name, "type": mtype, "K": K}

        for strategy in ["seed_only", "gumbel_only", "combined"]:
            best_qpos, best_seg_risk, all_risks = run_strategy(
                strategy, mode, n_frames, base_seed, K=K
            )
            if best_qpos is not None:
                report, _ = full_critic.score(best_qpos, clip_name, mtype)
                full_risk = report.risk_score
                action = report.recommended_action
            else:
                full_risk = float("nan")
                action = "failed"
            print(f"  {strategy:12s}: seg_best={best_seg_risk:.2f}  full_risk={full_risk:.2f}  action={action}")

            row[f"risk_{strategy}"] = full_risk
            row[f"action_{strategy}"] = action
            row[f"all_seg_risks_{strategy}"] = str([round(r, 2) for r in all_risks])

        rows.append(row)

csv_path = Path("results/diversity_ablation.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved: {csv_path}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("DIVERSITY SOURCE ABLATION SUMMARY (K=4)")
print(f"{'='*60}")

import numpy as np
for strategy in ["seed_only", "gumbel_only", "combined"]:
    risks = [r[f"risk_{strategy}"] for r in rows if np.isfinite(r[f"risk_{strategy}"])]
    accepts = sum(1 for r in rows if r[f"action_{strategy}"] == "accept")
    print(f"  {strategy:12s}: mean_risk={np.mean(risks):.2f}  accept={accepts}/{len(rows)}")

# per type
TYPES = ["static", "locomotion", "expressive", "whole_body"]
print("\nBy type:")
for mtype in TYPES:
    sub = [r for r in rows if r["type"] == mtype]
    for strategy in ["seed_only", "gumbel_only", "combined"]:
        risks = [r[f"risk_{strategy}"] for r in sub if np.isfinite(r[f"risk_{strategy}"])]
        a = sum(1 for r in sub if r[f"action_{strategy}"] == "accept")
        n = len(sub)
        if risks:
            print(f"  {mtype:15s} {strategy:12s}: {np.mean(risks):.2f} (acc={a}/{n})")
