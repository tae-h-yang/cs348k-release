"""
Compute-matched ablation: Per-Segment-K4 vs Whole-Clip-K4.

Both strategies use approximately the same inference budget:
  - WC-K4: 4 full clips × ~12 segments each = ~48 inference calls
  - PS-K4: ~12 segments × 4 candidates each = ~48 inference calls

The question: does greedy segment-level steering beat whole-clip selection
at the same compute cost? This tests whether the physics critic signal is
informative enough at the segment level to outperform clip-level reranking.

CUDA nondeterminism note: MotionBricks GPU inference is non-deterministic
across calls even with the same torch.manual_seed. Two consecutive calls
with the same seed diverge from the first generation step (~frame 16, max diff
~0.85 rad). This means PS cannot be fairly compared against a "true K=1
deterministic baseline" — hence the compute-matched design.

Results saved to: results/steered_vs_wc_ablation.csv
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
steered_mod = _load("generate_steered", "generate_steered.py")
generate_clip = guided_mod.generate_clip
generate_best_of_k = guided_mod.generate_best_of_k
generate_clip_steered = steered_mod.generate_clip_steered

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
K = 4  # budget: 4 candidates per strategy

out_base = Path("data/steered_ablation")
out_base.mkdir(parents=True, exist_ok=True)
rows = []

for cfg in ALL_CONFIGS:
    mode, n_frames, mtype = cfg["mode"], cfg["n_frames"], cfg["type"]
    for seed_idx in range(N_SEEDS):
        seed = seed_idx * 1000
        clip_name = f"{mode}_seed{seed_idx}"
        print(f"\n[{clip_name}]")

        # ── Whole-Clip K=4 (WC-K4) ──────────────────────────────────────────
        # Generate 4 full clips independently, pick the lowest-risk one.
        try:
            wc_qpos, wc_meta = generate_best_of_k(
                demo_agent, mode, n_frames, base_seed=seed,
                K=K, critic=segment_critic, motion_type=mtype,
            )
            report_wc, _ = full_critic.score(wc_qpos, clip_name, mtype)
            risk_wc = report_wc.risk_score
            action_wc = report_wc.recommended_action
            np.save(out_base / f"{clip_name}_WC{K}.npy", wc_qpos)
        except Exception as e:
            print(f"  WC-{K} FAILED: {e}")
            import traceback; traceback.print_exc()
            risk_wc, action_wc = float("nan"), "failed"
            wc_meta = {"risk_reduction_vs_k0_pct": float("nan")}

        print(f"  WC-{K}: risk={risk_wc:.2f}  action={action_wc}")

        # ── Per-Segment K=4 (PS-K4) ─────────────────────────────────────────
        # At each generation step (~16 frames), try 4 candidates and commit best.
        # The committed segment becomes context for the next step.
        try:
            ps_qpos, ps_meta = generate_clip_steered(
                demo_agent, mode, n_frames, seed=seed,
                K_per_segment=K, critic=segment_critic,
            )
            report_ps, _ = full_critic.score(ps_qpos, clip_name, mtype)
            risk_ps = report_ps.risk_score
            action_ps = report_ps.recommended_action
            np.save(out_base / f"{clip_name}_PS{K}.npy", ps_qpos)
        except Exception as e:
            print(f"  PS-{K} FAILED: {e}")
            import traceback; traceback.print_exc()
            risk_ps, action_ps = float("nan"), "failed"
            ps_meta = {"mean_segment_risk": float("nan"), "n_segments_steered": 0,
                       "n_segments_total": 0}

        print(f"  PS-{K}: risk={risk_ps:.2f}  action={action_ps}"
              f"  seg_switched={ps_meta['n_segments_steered']}/{ps_meta['n_segments_total']}")

        rows.append({
            "clip": clip_name,
            "type": mtype,
            "K": K,
            # Whole-clip strategy
            "risk_wc": risk_wc,
            "action_wc": action_wc,
            # Per-segment strategy
            "risk_ps": risk_ps,
            "action_ps": action_ps,
            # Which is better?
            "wc_wins": (np.isfinite(risk_wc) and np.isfinite(risk_ps) and risk_wc <= risk_ps),
            "ps_wins": (np.isfinite(risk_wc) and np.isfinite(risk_ps) and risk_ps < risk_wc),
            # Per-segment metadata
            "ps_segs_switched": ps_meta.get("n_segments_steered", 0),
            "ps_segs_total":    ps_meta.get("n_segments_total", 0),
            "ps_mean_seg_risk": ps_meta.get("mean_segment_risk", float("nan")),
            # WC metadata
            "wc_gain_vs_k0_pct": wc_meta.get("risk_reduction_vs_k0_pct", float("nan")),
        })

Path("results").mkdir(exist_ok=True)
csv_path = Path("results/steered_vs_wc_ablation.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved: {csv_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"COMPUTE-MATCHED COMPARISON: WC-K{K} vs PS-K{K}")
print(f"(Both strategies: ~K × n_segments inference calls)")
print(f"{'='*60}")

def _risks(key, mtype=None):
    sub = [r for r in rows if mtype is None or r["type"] == mtype]
    return [r[key] for r in sub if np.isfinite(r[key])]

def _accepts(key, mtype=None):
    sub = [r for r in rows if mtype is None or r["type"] == mtype]
    return sum(1 for r in sub if r[key] == "accept")

print(f"\n{'Strategy':15s}  {'mean_risk':>10s}  {'accept':>8s}  {'reject':>8s}")
for label, krisk, kact in [("WC-K4", "risk_wc", "action_wc"),
                             ("PS-K4", "risk_ps", "action_ps")]:
    r = _risks(krisk)
    a = _accepts(kact)
    rej = sum(1 for row in rows if row[kact] == "reject_or_regenerate")
    print(f"  {label:13s}  {np.mean(r):>10.2f}  {a:>8d}  {rej:>8d}")

wc_wins = sum(1 for r in rows if r["wc_wins"])
ps_wins = sum(1 for r in rows if r["ps_wins"])
print(f"\nWC wins: {wc_wins}/{len(rows)}   PS wins: {ps_wins}/{len(rows)}")

print(f"\n{'By type (mean risk WC / PS):':}")
for mtype in ["static", "locomotion", "expressive", "whole_body"]:
    r_wc = _risks("risk_wc", mtype)
    r_ps = _risks("risk_ps", mtype)
    a_wc = _accepts("action_wc", mtype)
    a_ps = _accepts("action_ps", mtype)
    n = len([r for r in rows if r["type"] == mtype])
    if r_wc and r_ps:
        print(f"  {mtype:15s}: WC={np.mean(r_wc):.1f}(acc={a_wc}/{n})  "
              f"PS={np.mean(r_ps):.1f}(acc={a_ps}/{n})")
