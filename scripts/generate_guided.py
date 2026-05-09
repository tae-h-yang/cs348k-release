"""
Physics-critic-guided best-of-K generation for MotionBricks.

For each (motion_style, seed) pair, generates K candidate clips with stochastic
pose-token sampling (Gumbel, different random seeds) and selects the
lowest-risk candidate according to the inverse-dynamics critic.

This is inference-time compute scaling for physics-feasible motion generation:
zero retraining, improves quality by spending more compute at generation time.

Usage:
    # K-ablation (K=1,4,8,16,32) for all motion styles
    python scripts/generate_guided.py --ablation

    # Single K for quick test
    python scripts/generate_guided.py --n_candidates 8 --modes walk idle

    # Baseline (K=1, deterministic, matches original generate_motions.py)
    python scripts/generate_guided.py --n_candidates 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics_eval.online_critic import OnlineSegmentCritic
from physics_eval.physaware import PhysicalAwarenessCritic

MOTIONBRICKS_DIR = Path(__file__).parent.parent.parent / "GR00T-WholeBodyControl" / "motionbricks"
RESULTS_DIR = Path(__file__).parent.parent / "results"

MOTION_CONFIGS = [
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


# ─── MotionBricks stochastic-sampling patch ───────────────────────────────────

def _enable_stochastic_sampling(inferencer):
    """
    Patch inferencer.predict so pose tokens are sampled stochastically (Gumbel)
    instead of argmax. Stores original for restoration.
    """
    original_predict = inferencer.predict

    def stochastic_predict(*args, config=None, **kwargs):
        if config is None:
            config = {}
        config = dict(config)
        config["pose_token_sampling_use_argmax"] = False
        return original_predict(*args, config=config, **kwargs)

    inferencer._original_predict = original_predict
    inferencer.predict = stochastic_predict


def _restore_deterministic_sampling(inferencer):
    if hasattr(inferencer, "_original_predict"):
        inferencer.predict = inferencer._original_predict
        del inferencer._original_predict


# ─── Clip generation ──────────────────────────────────────────────────────────

def generate_clip(demo_agent, mode: str, n_frames: int, seed: int,
                  stochastic: bool = False, generate_dt: float = 2.0) -> np.ndarray:
    """Generate one clip. If stochastic=True, uses Gumbel pose-token sampling."""
    import torch
    import mujoco

    np.random.seed(seed)
    torch.manual_seed(seed)

    inferencer = demo_agent.full_agent._inferencer
    if stochastic:
        _enable_stochastic_sampling(inferencer)

    demo_agent.full_agent.reset()
    qpos_list = []
    controller_dt = demo_agent.controller.get_controller_dt()

    try:
        for step in range(n_frames):
            qpos = demo_agent.full_agent.get_next_frame()
            qpos_list.append(qpos.copy())

            context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
            demo_agent.mj_data.qpos[:] = qpos

            force_idle = step + 100 > n_frames
            control_signals = demo_agent.controller.generate_control_signals(
                None, demo_agent.mj_model, demo_agent.mj_data, visualize=False,
                control_info={"force_idle": force_idle, "allowed_mode": mode}
            )
            control_signals["context_mujoco_qpos"] = context_mujoco_qpos
            control_signals["random_seed"] = torch.tensor([seed])

            with torch.no_grad():
                demo_agent.full_agent.generate_new_frames(
                    control_signals, controller_dt * generate_dt
                )

            mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)
    finally:
        if stochastic:
            _restore_deterministic_sampling(inferencer)

    return np.stack(qpos_list, axis=0).astype(np.float32)


def generate_best_of_k(
    demo_agent, mode: str, n_frames: int, base_seed: int,
    K: int, critic: OnlineSegmentCritic,
    motion_type: str = "unknown",
) -> tuple[np.ndarray, dict]:
    """
    Generate K candidate clips and return the lowest-risk one.

    Diversity sources:
      - Different random seeds → different target reference poses from VQ-VAE codebook.
      - Stochastic Gumbel pose-token sampling (vs. argmax) → variation for same target.

    Returns: (best_qpos, metadata_dict)
    """
    import torch

    candidate_clips = []
    candidate_risks = []

    for k in range(K):
        # Alternate between deterministic (k=0) and stochastic seeds
        stochastic = (k > 0)  # k=0 is the deterministic baseline
        seed_k = base_seed + k * 137  # prime-spaced seeds for good coverage

        try:
            qpos_k = generate_clip(
                demo_agent, mode, n_frames, seed=seed_k, stochastic=stochastic
            )
            risk_k = critic.score_segment(qpos_k)
        except Exception as e:
            print(f"      candidate k={k} failed: {e}")
            qpos_k = None
            risk_k = float("inf")

        candidate_clips.append(qpos_k)
        candidate_risks.append(risk_k)
        print(f"      k={k:2d}  seed={seed_k}  stochastic={stochastic}  risk={risk_k:.2f}")

    # Select best valid candidate
    valid = [(i, r) for i, r in enumerate(candidate_risks)
             if candidate_clips[i] is not None and np.isfinite(r)]
    if not valid:
        raise RuntimeError("All candidates failed")

    best_k, best_risk = min(valid, key=lambda x: x[1])
    best_qpos = candidate_clips[best_k]

    return best_qpos, {
        "best_k": best_k,
        "best_risk": best_risk,
        "K": K,
        "mean_risk": float(np.mean([r for r in candidate_risks if np.isfinite(r)])),
        "worst_risk": float(np.max([r for r in candidate_risks if np.isfinite(r)])),
        "risk_reduction_vs_k0_pct": float(
            100.0 * (candidate_risks[0] - best_risk) / max(abs(candidate_risks[0]), 1e-8)
        ) if np.isfinite(candidate_risks[0]) else float("nan"),
        "all_risks": candidate_risks,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_candidates", type=int, default=8)
    parser.add_argument("--ablation", action="store_true",
                        help="Run K=1,4,8,16,32 ablation")
    parser.add_argument("--modes", nargs="*", default=None,
                        help="Subset of modes (default: all 13)")
    parser.add_argument("--out_dir", type=str, default="data/guided")
    args = parser.parse_args()

    k_values = [1, 4, 8, 16, 32] if args.ablation else [args.n_candidates]

    # Load MotionBricks
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

    import torch
    device = next(demo_agent.full_agent.parameters()).device
    print(f"Model loaded. Device: {device}")

    critic = OnlineSegmentCritic()
    full_critic = PhysicalAwarenessCritic()  # for final clip-level scoring

    configs = MOTION_CONFIGS
    if args.modes:
        configs = [c for c in MOTION_CONFIGS if c["mode"] in args.modes]

    all_rows = []

    for K in k_values:
        out_dir = Path(args.out_dir if not args.ablation else f"data/guided_K{K}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\nK={K} | output: {out_dir}")

        for cfg in configs:
            mode, n_frames, mtype = cfg["mode"], cfg["n_frames"], cfg["type"]
            for seed_idx in range(N_SEEDS):
                clip_name = f"{mode}_seed{seed_idx}"
                base_seed = seed_idx * 1000
                print(f"\n  [{clip_name}]  K={K}")

                try:
                    best_qpos, meta = generate_best_of_k(
                        demo_agent, mode, n_frames, base_seed=base_seed,
                        K=K, critic=critic, motion_type=mtype,
                    )
                except Exception as e:
                    print(f"    FAILED: {e}")
                    import traceback; traceback.print_exc()
                    continue

                # Full-clip score via the standard critic
                report, _ = full_critic.score(best_qpos, clip_name, mtype, variant=f"best_of_{K}")
                np.save(out_dir / f"{clip_name}.npy", best_qpos)

                row = {
                    "clip": clip_name,
                    "type": mtype,
                    "K": K,
                    "n_frames": n_frames,
                    "full_clip_risk": report.risk_score,
                    "p95_torque_limit_ratio": report.p95_torque_limit_ratio,
                    "recommended_action": report.recommended_action,
                    "best_k_idx": meta["best_k"],
                    "segment_best_risk": meta["best_risk"],
                    "segment_mean_risk": meta["mean_risk"],
                    "segment_worst_risk": meta["worst_risk"],
                    "risk_reduction_vs_k0_pct": meta["risk_reduction_vs_k0_pct"],
                }
                all_rows.append(row)
                print(f"    SELECTED k={meta['best_k']} | full_risk={report.risk_score:.2f}"
                      f" | action={report.recommended_action}"
                      f" | reduction_vs_k0={meta['risk_reduction_vs_k0_pct']:.1f}%")

    # Save and summarize
    summary_path = RESULTS_DIR / "guided_generation_summary.csv"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nSaved: {summary_path}")

    print("\n=== Summary by K ===")
    for K in k_values:
        rows_k = [r for r in all_rows if r["K"] == K]
        if not rows_k:
            continue
        risks = [r["full_clip_risk"] for r in rows_k]
        accept = sum(1 for r in rows_k if r["recommended_action"] == "accept")
        reject = sum(1 for r in rows_k if r["recommended_action"] == "reject_or_regenerate")
        print(f"K={K:2d}: mean_risk={np.mean(risks):.2f}  accept={accept}  reject={reject}  n={len(rows_k)}")


if __name__ == "__main__":
    main()
