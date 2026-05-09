"""
Neural-guided best-of-K ablation: addresses Reviewer R5.

Runs WC-K8 using the neural surrogate critic (1D CNN) instead of the heuristic
mj_inverse critic for candidate selection. Compares:
  - Heuristic-K8 (mj_inverse, existing results in guided_ablation_full.csv)
  - Neural-K8 (1D CNN surrogate, this experiment)

Reports:
  1. Risk distribution: are neural-selected clips as good as heuristic-selected?
  2. Selection agreement: which K=8 clips does neural vs heuristic select?
  3. Speed comparison: heuristic scoring time vs neural scoring time per candidate

Results: results/neural_guided_ablation.csv
"""
import sys, os, csv, time
from pathlib import Path
import numpy as np
import torch

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

# ── Load neural critic ────────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Inline NeuralCritic (avoids circular import)
import torch.nn as nn
class NeuralCritic(nn.Module):
    def __init__(self, in_ch=36, hidden=64, window=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden*2, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden*2), nn.ReLU(),
            nn.Conv1d(hidden*2, hidden*2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden*2), nn.ReLU(),
            nn.Conv1d(hidden*2, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)

neural_model = NeuralCritic().to(DEVICE)
model_path = REPO / "results" / "neural_critic" / "model.pt"
neural_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
neural_model.eval()
print(f"Neural critic loaded from {model_path}")

WINDOW = 32

def score_with_neural(qpos: np.ndarray) -> float:
    """Score a clip with neural critic using overlapping windows."""
    T = len(qpos)
    if T < WINDOW:
        # Pad
        pad = np.zeros((WINDOW - T, 36), dtype=np.float32)
        qpos = np.vstack([qpos, pad]).astype(np.float32)
        T = WINDOW
    windows = []
    for start in range(0, T - WINDOW + 1, WINDOW // 2):
        seg = qpos[start:start + WINDOW].astype(np.float32)
        windows.append(torch.from_numpy(seg).T)  # (36, WINDOW)
    if not windows:
        return float("inf")
    batch = torch.stack(windows).to(DEVICE)  # (N, 36, WINDOW)
    with torch.no_grad():
        log_risks = neural_model(batch).cpu().numpy()
    return float(np.expm1(log_risks).max())  # max window risk as clip risk

heuristic_critic = OnlineSegmentCritic()
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
K = 8
SEED_OFFSETS = [k * 137 for k in range(K)]

rows = []
neural_score_times = []
heuristic_score_times = []

for cfg in ALL_CONFIGS:
    mode, n_frames, mtype = cfg["mode"], cfg["n_frames"], cfg["type"]
    for seed_idx in range(N_SEEDS):
        base_seed = seed_idx * 1000
        clip_name = f"{mode}_seed{seed_idx}"
        print(f"\n[{clip_name}]")

        candidates = []
        for k in range(K):
            seed_k = base_seed + SEED_OFFSETS[k]
            stochastic = (k > 0)
            try:
                qpos_k = generate_clip(demo_agent, mode, n_frames, seed=seed_k, stochastic=stochastic)
                candidates.append(qpos_k)
            except Exception as e:
                print(f"  k={k} FAILED: {e}")
                candidates.append(None)

        # Score with neural critic
        t0 = time.perf_counter()
        neural_risks = []
        for qpos_k in candidates:
            if qpos_k is not None:
                neural_risks.append(score_with_neural(qpos_k))
            else:
                neural_risks.append(float("inf"))
        neural_score_times.append(1000 * (time.perf_counter() - t0) / K)

        # Score with heuristic critic (for comparison)
        t0 = time.perf_counter()
        heuristic_risks = []
        for qpos_k in candidates:
            if qpos_k is not None:
                heuristic_risks.append(heuristic_critic.score_segment(qpos_k))
            else:
                heuristic_risks.append(float("inf"))
        heuristic_score_times.append(1000 * (time.perf_counter() - t0) / K)

        # Best selection
        valid_neural = [(k, r) for k, r in enumerate(neural_risks)
                        if candidates[k] is not None and np.isfinite(r)]
        valid_heuristic = [(k, r) for k, r in enumerate(heuristic_risks)
                           if candidates[k] is not None and np.isfinite(r)]

        if not valid_neural or not valid_heuristic:
            print("  All candidates failed, skipping")
            continue

        best_k_neural = min(valid_neural, key=lambda x: x[1])[0]
        best_k_heuristic = min(valid_heuristic, key=lambda x: x[1])[0]
        selection_agrees = (best_k_neural == best_k_heuristic)

        # Full-clip scores for selected clips
        report_neural, _ = full_critic.score(candidates[best_k_neural], clip_name, mtype)
        report_heuristic, _ = full_critic.score(candidates[best_k_heuristic], clip_name, mtype)

        print(f"  Neural:    k={best_k_neural}  full_risk={report_neural.risk_score:.2f}  "
              f"action={report_neural.recommended_action}")
        print(f"  Heuristic: k={best_k_heuristic}  full_risk={report_heuristic.risk_score:.2f}  "
              f"action={report_heuristic.recommended_action}  agree={selection_agrees}")

        rows.append({
            "clip": clip_name, "type": mtype,
            # Neural-guided selection
            "best_k_neural": best_k_neural,
            "full_risk_neural": report_neural.risk_score,
            "action_neural": report_neural.recommended_action,
            # Heuristic-guided selection
            "best_k_heuristic": best_k_heuristic,
            "full_risk_heuristic": report_heuristic.risk_score,
            "action_heuristic": report_heuristic.recommended_action,
            # Agreement
            "selection_agrees": selection_agrees,
            # Timing
            "neural_ms_per_clip": neural_score_times[-1],
            "heuristic_ms_per_clip": heuristic_score_times[-1],
        })

csv_path = Path("results/neural_guided_ablation.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSaved: {csv_path}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("NEURAL vs HEURISTIC CRITIC FOR WC-K8 SELECTION")
print(f"{'='*60}")

neural_full_risks = [r["full_risk_neural"] for r in rows if np.isfinite(r["full_risk_neural"])]
heuristic_full_risks = [r["full_risk_heuristic"] for r in rows if np.isfinite(r["full_risk_heuristic"])]
n_agree = sum(1 for r in rows if r["selection_agrees"])
n_total = len(rows)

print(f"\nNeural-K8:    mean_risk={np.mean(neural_full_risks):.2f}  "
      f"accept={sum(1 for r in rows if r['action_neural']=='accept')}/{n_total}")
print(f"Heuristic-K8: mean_risk={np.mean(heuristic_full_risks):.2f}  "
      f"accept={sum(1 for r in rows if r['action_heuristic']=='accept')}/{n_total}")
print(f"Selection agreement: {n_agree}/{n_total} ({100*n_agree/n_total:.0f}%)")
print(f"\nScoring speed (K={K} candidates):")
print(f"  Neural:    {np.mean(neural_score_times):.2f} ms/candidate avg")
print(f"  Heuristic: {np.mean(heuristic_score_times):.2f} ms/candidate avg")
print(f"  Speedup:   {np.mean(heuristic_score_times)/np.mean(neural_score_times):.1f}×")
