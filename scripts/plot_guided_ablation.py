"""
Generate the three key figures for the best-of-K ablation:
  1. Risk vs. K curve (main result)
  2. Accept/Reject/Repair counts vs. K (bar chart)
  3. Per-motion-type breakdown at K=1 vs K=8 vs K=16
Run after run_ablation.py has produced results/guided_ablation_full.csv.
"""
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
CSV_PATH = RESULTS_DIR / "guided_ablation_full.csv"

if not CSV_PATH.exists():
    print(f"ERROR: {CSV_PATH} not found. Run scripts/run_ablation.py first.")
    sys.exit(1)

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

K_VALUES = sorted({int(r["K"]) for r in rows})
TYPES = ["static", "locomotion", "expressive", "whole_body"]
TYPE_LABELS = {"static": "Static", "locomotion": "Locomotion",
               "expressive": "Expressive", "whole_body": "Whole-body"}
TYPE_COLORS = {"static": "#66BB6A", "locomotion": "#26A69A",
               "expressive": "#5C6BC0", "whole_body": "#EF5350"}

def get_risks(k, mtype=None):
    subset = [r for r in rows if int(r["K"]) == k]
    if mtype:
        subset = [r for r in subset if r["type"] == mtype]
    return [float(r["full_risk"]) for r in subset if r["full_risk"] not in ("nan", "")]

def get_actions(k):
    subset = [r for r in rows if int(r["K"]) == k]
    return {
        "accept": sum(1 for r in subset if r["action"] == "accept"),
        "repair_or_rerank": sum(1 for r in subset if r["action"] == "repair_or_rerank"),
        "reject_or_regenerate": sum(1 for r in subset if r["action"] == "reject_or_regenerate"),
    }

# ── Figure 1: Mean risk vs. K (overall + by type) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
means = [np.mean(get_risks(k)) for k in K_VALUES]
p25 = [np.percentile(get_risks(k), 25) for k in K_VALUES]
p75 = [np.percentile(get_risks(k), 75) for k in K_VALUES]
ax.plot(K_VALUES, means, "o-", color="#1565C0", linewidth=2.5, markersize=8, label="All clips")
ax.fill_between(K_VALUES, p25, p75, alpha=0.15, color="#1565C0")
ax.set_xlabel("Number of Candidates (K)", fontsize=12)
ax.set_ylabel("Mean Heuristic Risk Score", fontsize=12)
ax.set_title("Best-of-K: Risk vs. Compute Budget", fontsize=13)
ax.set_xscale("log", base=2)
ax.set_xticks(K_VALUES)
ax.set_xticklabels([str(k) for k in K_VALUES])
ax.grid(alpha=0.35)
ax.legend(fontsize=10)

ax = axes[1]
for mtype in TYPES:
    means_t = [np.mean(get_risks(k, mtype)) if get_risks(k, mtype) else float("nan")
               for k in K_VALUES]
    ax.plot(K_VALUES, means_t, "o-", label=TYPE_LABELS[mtype],
            color=TYPE_COLORS[mtype], linewidth=2, markersize=7)
ax.set_xlabel("Number of Candidates (K)", fontsize=12)
ax.set_ylabel("Mean Risk Score", fontsize=12)
ax.set_title("Risk vs. K by Motion Type", fontsize=13)
ax.set_xscale("log", base=2)
ax.set_xticks(K_VALUES)
ax.set_xticklabels([str(k) for k in K_VALUES])
ax.grid(alpha=0.35)
ax.legend(fontsize=9)

fig.tight_layout()
out = RESULTS_DIR / "guided_risk_vs_K.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Figure 2: Action counts vs. K ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(K_VALUES))
width = 0.25
action_colors = {"accept": "#26A69A", "repair_or_rerank": "#FFA726", "reject_or_regenerate": "#EF5350"}
action_labels = {"accept": "Accept", "repair_or_rerank": "Repair/Rerank", "reject_or_regenerate": "Reject"}

for i, (action, color) in enumerate(action_colors.items()):
    counts = [get_actions(k)[action] for k in K_VALUES]
    ax.bar(x + (i - 1) * width, counts, width=width, label=action_labels[action],
           color=color, edgecolor="k", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in K_VALUES])
ax.set_ylabel("Number of Clips")
ax.set_title("Critic Action Distribution vs. Candidate Count K")
ax.legend()
ax.grid(axis="y", alpha=0.35)
fig.tight_layout()
out = RESULTS_DIR / "guided_action_counts_vs_K.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Figure 3: Before/after scatter K=1 vs K=8 ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
clips_k1 = {r["clip"]: float(r["full_risk"]) for r in rows if int(r["K"]) == 1 and r["full_risk"] not in ("nan", "")}
clips_k8 = {r["clip"]: float(r["full_risk"]) for r in rows if int(r["K"]) == 8 and r["full_risk"] not in ("nan", "")}
clips_type = {r["clip"]: r["type"] for r in rows}

for mtype in TYPES:
    clips = [c for c in clips_k1 if c in clips_k8 and clips_type.get(c) == mtype]
    x_vals = [clips_k1[c] for c in clips]
    y_vals = [clips_k8[c] for c in clips]
    ax.scatter(x_vals, y_vals, label=TYPE_LABELS[mtype], color=TYPE_COLORS[mtype],
               s=65, alpha=0.85, edgecolors="k", linewidths=0.5)

max_risk = max(max(clips_k1.values()), max(clips_k8.values())) * 1.05
ax.plot([0, max_risk], [0, max_risk], "k--", lw=1, alpha=0.5, label="No improvement")
ax.set_xlabel("Risk Score  K=1 (deterministic baseline)", fontsize=11)
ax.set_ylabel("Risk Score  K=8 (best-of-8)", fontsize=11)
ax.set_title("Per-Clip Risk: K=1 vs K=8", fontsize=13)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
# Annotation
n_improved = sum(1 for c in clips_k1 if c in clips_k8 and clips_k8[c] < clips_k1[c])
n_total = sum(1 for c in clips_k1 if c in clips_k8)
ax.text(0.04, 0.96, f"{n_improved}/{n_total} clips improved",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
fig.tight_layout()
out = RESULTS_DIR / "guided_k1_vs_k8_scatter.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n=== Summary Table ===")
print(f"{'K':>4}  {'mean_risk':>10}  {'accept':>8}  {'reject':>8}  {'n':>4}")
for k in K_VALUES:
    risks = get_risks(k)
    acts = get_actions(k)
    n = len([r for r in rows if int(r["K"]) == k])
    print(f"{k:>4}  {np.mean(risks):>10.2f}  {acts['accept']:>8}  {acts['reject_or_regenerate']:>8}  {n:>4}")

print("\n=== By Type at K=8 ===")
for mtype in TYPES:
    r_k1 = get_risks(1, mtype)
    r_k8 = get_risks(8, mtype)
    if r_k1 and r_k8:
        reduction = 100.0 * (np.mean(r_k1) - np.mean(r_k8)) / max(np.mean(r_k1), 1e-8)
        acts = get_actions(8)
        print(f"  {mtype:15s}: K=1={np.mean(r_k1):.1f}  K=8={np.mean(r_k8):.1f}  reduction={reduction:.1f}%")
