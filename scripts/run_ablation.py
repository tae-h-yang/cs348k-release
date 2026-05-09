"""Full K-ablation: all 13 motion styles x 3 seeds x K=1,4,8,16."""
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

# Import helpers directly to avoid package path issue
import importlib.util, types
_spec = importlib.util.spec_from_file_location("generate_guided",
    Path(__file__).parent / "generate_guided.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
generate_clip = _mod.generate_clip

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
K_VALUES = [1, 4, 8, 16]
N_SEEDS = 3

rows = []
out_base = Path("data/guided_ablation")
out_base.mkdir(parents=True, exist_ok=True)

for cfg in ALL_CONFIGS:
    mode, n_frames, mtype = cfg["mode"], cfg["n_frames"], cfg["type"]
    for seed_idx in range(N_SEEDS):
        base_seed = seed_idx * 1000
        clip_name = f"{mode}_seed{seed_idx}"
        print(f"\n[{clip_name}]")

        for K in K_VALUES:
            best_risk = float("inf")
            best_qpos = None
            all_risks = []

            for k in range(K):
                stochastic = (k > 0)
                seed_k = base_seed + k * 137
                try:
                    qpos_k = generate_clip(demo_agent, mode, n_frames, seed=seed_k, stochastic=stochastic)
                    risk_k = segment_critic.score_segment(qpos_k)
                except Exception as e:
                    print(f"  k={k} FAILED: {e}")
                    risk_k = float("inf")
                    qpos_k = None
                all_risks.append(risk_k)
                if risk_k < best_risk and qpos_k is not None:
                    best_risk = risk_k
                    best_qpos = qpos_k

            if best_qpos is not None:
                report, _ = full_critic.score(best_qpos, clip_name, mtype)
                full_risk = report.risk_score
                action = report.recommended_action
                np.save(out_base / f"{clip_name}_K{K}.npy", best_qpos)
            else:
                full_risk, action = float("nan"), "failed"

            finite = [r for r in all_risks if np.isfinite(r)]
            k0 = all_risks[0] if np.isfinite(all_risks[0]) else float("nan")
            gain = 100.0 * (k0 - best_risk) / max(abs(k0), 1e-8) if np.isfinite(k0) else float("nan")

            rows.append({
                "clip": clip_name, "type": mtype, "K": K, "full_risk": full_risk,
                "action": action, "best_seg_risk": best_risk, "k0_seg_risk": k0,
                "gain_vs_k0_pct": gain,
                "mean_seg_risk": float(np.mean(finite)) if finite else float("nan"),
            })
            print(f"  K={K:2d}: full_risk={full_risk:.2f}  gain={gain:.1f}%  action={action}")

Path("results").mkdir(exist_ok=True)
with open("results/guided_ablation_full.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nSaved: results/guided_ablation_full.csv")

print("\n=== Summary by K ===")
for K in K_VALUES:
    kr = [r for r in rows if r["K"] == K]
    risks = [r["full_risk"] for r in kr if np.isfinite(r["full_risk"])]
    gains = [r["gain_vs_k0_pct"] for r in kr if np.isfinite(r["gain_vs_k0_pct"])]
    accept = sum(1 for r in kr if r["action"] == "accept")
    reject = sum(1 for r in kr if r["action"] == "reject_or_regenerate")
    print(f"K={K:2d}: mean_risk={np.mean(risks):.2f}  mean_gain={np.mean(gains):.1f}%  accept={accept}  reject={reject}  n={len(kr)}")

print("\n=== By type at K=8 ===")
for mtype in ["static", "locomotion", "expressive", "whole_body"]:
    kr = [r for r in rows if r["type"] == mtype and r["K"] == 8]
    if not kr:
        continue
    risks = [r["full_risk"] for r in kr if np.isfinite(r["full_risk"])]
    k1_risks = [r["full_risk"] for r in rows if r["type"] == mtype and r["K"] == 1 and np.isfinite(r["full_risk"])]
    accept = sum(1 for r in kr if r["action"] == "accept")
    print(f"  {mtype:15s}: K=1 mean={np.mean(k1_risks):.1f}  K=8 mean={np.mean(risks):.1f}  accept_K8={accept}/{len(kr)}")
