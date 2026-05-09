"""
Smoothing baseline (Reviewer R2): addresses the lack of external baselines.

We apply Gaussian velocity smoothing and Savitzky-Golay filter to K=1 clips.
This is distinct from the retiming baselines (which slow the motion down) —
smoothing reduces jerk without changing timing.

Applies 3 smoothing variants to all 39 K=1 clips and scores with full critic.
Compares to WC-K=8 to show best-of-K isn't just smoothing with extra steps.

Results: results/smoothing_baseline.csv
"""
import csv, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from physics_eval.physaware import PhysicalAwarenessCritic

REPO = Path(__file__).parent.parent
DATA_K1 = REPO / "data" / "guided_ablation"
RESULTS = REPO / "results"

full_critic = PhysicalAwarenessCritic()

# Load K=1 files and K=8 files
k1_files = sorted(DATA_K1.glob("*_K1.npy"))
k8_files = {p.stem.replace("_K8", ""): p for p in DATA_K1.glob("*_K8.npy")}

TYPE_MAP = {
    "idle": "static",
    "walk": "locomotion", "slow_walk": "locomotion", "stealth_walk": "locomotion",
    "injured_walk": "locomotion", "walk_zombie": "locomotion", "walk_stealth": "locomotion",
    "walk_boxing": "expressive", "walk_happy_dance": "expressive",
    "walk_gun": "expressive", "walk_scared": "expressive",
    "hand_crawling": "whole_body", "elbow_crawling": "whole_body",
}

def get_type(clip_name):
    for style, mtype in TYPE_MAP.items():
        if clip_name.startswith(style):
            return mtype
    return "unknown"

def smooth_gaussian(qpos, sigma=1.5):
    return gaussian_filter1d(qpos, sigma=sigma, axis=0).astype(np.float32)

def smooth_savgol(qpos, window=11, poly=3):
    out = np.zeros_like(qpos)
    for j in range(qpos.shape[1]):
        out[:, j] = savgol_filter(qpos[:, j], window_length=window, polyorder=poly)
    return out.astype(np.float32)

def smooth_heavy_savgol(qpos, window=21, poly=3):
    out = np.zeros_like(qpos)
    for j in range(qpos.shape[1]):
        out[:, j] = savgol_filter(qpos[:, j], window_length=window, polyorder=poly)
    return out.astype(np.float32)

rows = []
print(f"Processing {len(k1_files)} K=1 clips...")

for k1_path in k1_files:
    stem = k1_path.stem.replace("_K1", "")  # e.g., "walk_seed0"
    mtype = get_type(stem)
    qpos_k1 = np.load(k1_path).astype(np.float32)

    # K=1 risk
    report_k1, _ = full_critic.score(qpos_k1, stem, mtype)
    risk_k1 = report_k1.risk_score

    # Smoothing variants
    variants = {
        "gaussian_σ1.5": smooth_gaussian(qpos_k1, sigma=1.5),
        "savgol_w11":    smooth_savgol(qpos_k1, window=11, poly=3),
        "savgol_w21":    smooth_heavy_savgol(qpos_k1, window=21, poly=3),
    }

    row = {"clip": stem, "type": mtype, "risk_k1": risk_k1}

    best_smooth_risk = float("inf")
    best_smooth_name = "none"
    for name, qpos_s in variants.items():
        report_s, _ = full_critic.score(qpos_s, stem, mtype)
        row[f"risk_{name}"] = report_s.risk_score
        row[f"action_{name}"] = report_s.recommended_action
        if report_s.risk_score < best_smooth_risk:
            best_smooth_risk = report_s.risk_score
            best_smooth_name = name

    row["best_smooth_risk"] = best_smooth_risk
    row["best_smooth_name"] = best_smooth_name

    # K=8 risk (from K=8 file)
    k8_key = stem
    if k8_key in k8_files:
        qpos_k8 = np.load(k8_files[k8_key]).astype(np.float32)
        report_k8, _ = full_critic.score(qpos_k8, stem, mtype)
        row["risk_k8"] = report_k8.risk_score
        row["action_k8"] = report_k8.recommended_action
    else:
        row["risk_k8"] = float("nan")
        row["action_k8"] = "missing"

    rows.append(row)
    print(f"  {stem}: K1={risk_k1:.1f}  best_smooth={best_smooth_risk:.1f}  K8={row['risk_k8']:.1f}")

csv_path = RESULTS / "smoothing_baseline.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nSaved: {csv_path}")

print(f"\n{'='*60}")
print("SMOOTHING BASELINE SUMMARY")
print(f"{'='*60}")

k1_risks = [r["risk_k1"] for r in rows if np.isfinite(r["risk_k1"])]
smooth_risks = [r["best_smooth_risk"] for r in rows if np.isfinite(r["best_smooth_risk"])]
k8_risks = [r["risk_k8"] for r in rows if np.isfinite(r["risk_k8"])]

print(f"  K=1 (no post-process):     mean={np.mean(k1_risks):.2f}  accept={sum(1 for r in rows if r['action_k8']!='accept' and r['risk_k1']<10)}")
print(f"  Best smoothing (per clip): mean={np.mean(smooth_risks):.2f}  accept={sum(1 for r in rows if r['best_smooth_risk']<10)}")
print(f"  WC-K8 (this work):         mean={np.mean(k8_risks):.2f}  accept={sum(1 for r in rows if r['action_k8']=='accept')}")
