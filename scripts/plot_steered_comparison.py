"""
Plot WC-K4 vs PS-K4 compute-matched comparison.
Reads results/steered_vs_wc_ablation.csv.
"""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results"
CSV_PATH = RESULTS_DIR / "steered_vs_wc_ablation.csv"

if not CSV_PATH.exists():
    print(f"ERROR: {CSV_PATH} not found.")
    sys.exit(1)

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

TYPES = ["static", "locomotion", "expressive", "whole_body"]
TYPE_LABELS = {"static": "Static", "locomotion": "Locomotion",
               "expressive": "Expressive", "whole_body": "Whole-body"}
TYPE_COLORS = {"static": "#66BB6A", "locomotion": "#26A69A",
               "expressive": "#5C6BC0", "whole_body": "#EF5350"}

def get_risks(strategy, mtype=None):
    sub = rows if mtype is None else [r for r in rows if r["type"] == mtype]
    key = "risk_wc" if strategy == "WC" else "risk_ps"
    return [float(r[key]) for r in sub if r[key] not in ("nan", "")]

def get_accepts(strategy, mtype=None):
    sub = rows if mtype is None else [r for r in rows if r["type"] == mtype]
    key = "action_wc" if strategy == "WC" else "action_ps"
    return sum(1 for r in sub if r[key] == "accept")

# ── Figure 1: per-clip scatter WC vs PS ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: per-clip scatter
ax = axes[0]
for mtype in TYPES:
    sub = [r for r in rows if r["type"] == mtype
           and r["risk_wc"] not in ("nan","") and r["risk_ps"] not in ("nan","")]
    xs = [float(r["risk_wc"]) for r in sub]
    ys = [float(r["risk_ps"]) for r in sub]
    ax.scatter(xs, ys, label=TYPE_LABELS[mtype], color=TYPE_COLORS[mtype],
               s=65, alpha=0.85, edgecolors="k", linewidths=0.5)

all_wc = [float(r["risk_wc"]) for r in rows if r["risk_wc"] not in ("nan","")]
all_ps = [float(r["risk_ps"]) for r in rows if r["risk_ps"] not in ("nan","")]
maxv = max(max(all_wc), max(all_ps)) * 1.05
ax.plot([0, maxv], [0, maxv], "k--", lw=1, alpha=0.5, label="Equal")
ax.set_xlabel("WC-K4 risk (whole-clip selection)", fontsize=11)
ax.set_ylabel("PS-K4 risk (per-segment steering)", fontsize=11)
ax.set_title("Compute-Matched: WC-K4 vs PS-K4", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

wc_wins = sum(1 for r in rows if float(r["risk_wc"] or "inf") < float(r["risk_ps"] or "inf"))
ax.text(0.04, 0.96, f"WC wins: {wc_wins}/{len(rows)} clips\n(points above diagonal = WC better)",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Right: bar chart by type
ax = axes[1]
x = np.arange(len(TYPES))
width = 0.35
wc_means = [np.mean(get_risks("WC", t)) for t in TYPES]
ps_means = [np.mean(get_risks("PS", t)) for t in TYPES]
ax.bar(x - width/2, wc_means, width, label="WC-K4", color="#1565C0", alpha=0.85, edgecolor="k")
ax.bar(x + width/2, ps_means, width, label="PS-K4", color="#E65100", alpha=0.85, edgecolor="k")
ax.set_xticks(x)
ax.set_xticklabels([TYPE_LABELS[t] for t in TYPES])
ax.set_ylabel("Mean Risk Score", fontsize=11)
ax.set_title("Mean Risk by Motion Type (K=4 budget)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.35)

# Annotate accepts
for i, mtype in enumerate(TYPES):
    a_wc = get_accepts("WC", mtype)
    a_ps = get_accepts("PS", mtype)
    n = len([r for r in rows if r["type"] == mtype])
    ax.text(i - width/2, wc_means[i] + 1, f"acc\n{a_wc}/{n}", ha="center", fontsize=7, color="#1565C0")
    ax.text(i + width/2, ps_means[i] + 1, f"acc\n{a_ps}/{n}", ha="center", fontsize=7, color="#E65100")

fig.tight_layout()
out = RESULTS_DIR / "steered_vs_wc_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Compute-matched WC-K4 vs PS-K4 ===")
print(f"{'Strategy':10s}  {'mean_risk':>10s}  {'accept':>8s}  {'reject':>8s}")
for label, strat in [("WC-K4", "WC"), ("PS-K4", "PS")]:
    r = get_risks(strat)
    a = get_accepts(strat)
    rej = sum(1 for row in rows if row[f"action_{strat.lower()}"] == "reject_or_regenerate")
    print(f"  {label:8s}  {np.mean(r):>10.2f}  {a:>8d}  {rej:>8d}")
print(f"\nWC wins: {wc_wins}/{len(rows)}  PS wins: {len(rows)-wc_wins}/{len(rows)}")
