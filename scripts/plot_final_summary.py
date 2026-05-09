"""
Final comprehensive summary plot: all methods and baselines.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).parent.parent
RESULTS = REPO / "results"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ─── Data ─────────────────────────────────────────────────────────────────────
guided = load_csv(RESULTS / "guided_ablation_full.csv")
steered = load_csv(RESULTS / "steered_vs_wc_ablation.csv")
smoothing = load_csv(RESULTS / "smoothing_baseline.csv")
diversity = load_csv(RESULTS / "diversity_ablation.csv")

def mean_risk(rows, key="full_risk"):
    vals = [float(r[key]) for r in rows if np.isfinite(float(r.get(key, "nan") or "nan"))]
    return np.mean(vals) if vals else float("nan")

def n_accept(rows, key="action"):
    return sum(1 for r in rows if r.get(key, "") == "accept")

def bootstrap_ci(arr, n=5000):
    rng = np.random.default_rng(42)
    arr = np.array(arr)
    boot = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ─── Figure 1: All methods comparison bar chart ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

methods = [
    ("K=1\nBaseline", 38.90, 5, "#EF5350"),
    ("Post-hoc\nSmoothing\n(best)", 41.02, 8, "#FF7043"),
    ("Retiming\n(best)", 23.74, 17, "#FFA726"),
    ("Seed Rerank\n(K=3)", 23.00, None, "#FFCC80"),
    ("PS-K4", 34.35, 18, "#AB47BC"),
    ("WC-K4", 19.61, 22, "#42A5F5"),
    ("WC-K8\n(main)", 15.93, 26, "#1565C0"),
    ("WC-K16", 13.61, 31, "#0D47A1"),
]

ax = axes[0]
xs = np.arange(len(methods))
colors_bar = [m[3] for m in methods]
vals_bar = [m[1] for m in methods]
accepts_bar = [m[2] for m in methods]

bars = ax.bar(xs, vals_bar, color=colors_bar, edgecolor="k", alpha=0.88)

# Annotate
for xi, (label, v, acc, c) in enumerate(methods):
    ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
    if acc is not None:
        ax.text(xi, -3.5, f"acc={acc}\n/39", ha="center", fontsize=7, color="#333")

ax.set_xticks(xs)
ax.set_xticklabels([m[0] for m in methods], fontsize=8)
ax.set_ylabel("Mean Risk Score (39 clips)", fontsize=11)
ax.set_title("All Methods: Mean Risk Comparison", fontsize=12)
ax.axhline(10, color="green", linestyle="--", lw=1.5, alpha=0.7, label="Accept threshold (<10)")
ax.axhline(50, color="red", linestyle="--", lw=1.5, alpha=0.7, label="Reject threshold (>50)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.35)
ax.set_ylim(-7, 50)

# ─── Figure 2: Per-type K=1 vs K=8 with CI ───────────────────────────────────
ax = axes[1]
TYPES = ["static", "locomotion", "expressive", "whole_body"]
TYPE_LABELS = {"static": "Static\n(n=3)", "locomotion": "Locomotion\n(n=18)",
               "expressive": "Expressive\n(n=12)", "whole_body": "Whole-body\n(n=6)"}

k1_by_type_means = []
k1_by_type_ci_lo = []
k1_by_type_ci_hi = []
k8_by_type_means = []
k8_by_type_ci_lo = []
k8_by_type_ci_hi = []
wilcoxon_ps = {"static": "n.s.", "locomotion": "p<0.001", "expressive": "p=0.001", "whole_body": "p=0.016"}

for t in TYPES:
    k1r = [float(r["full_risk"]) for r in guided if int(r["K"]) == 1 and r["type"] == t]
    k8r = [float(r["full_risk"]) for r in guided if int(r["K"]) == 8 and r["type"] == t]
    k1_by_type_means.append(np.mean(k1r))
    k8_by_type_means.append(np.mean(k8r))
    lo, hi = bootstrap_ci(k1r)
    k1_by_type_ci_lo.append(lo); k1_by_type_ci_hi.append(hi)
    lo, hi = bootstrap_ci(k8r)
    k8_by_type_ci_lo.append(lo); k8_by_type_ci_hi.append(hi)

x = np.arange(len(TYPES))
width = 0.35

# K=1 bars
b1 = ax.bar(x - width/2, k1_by_type_means, width, color="#EF9A9A", edgecolor="k", alpha=0.9, label="K=1")
ax.errorbar(x - width/2, k1_by_type_means,
            yerr=[np.array(k1_by_type_means) - np.array(k1_by_type_ci_lo),
                  np.array(k1_by_type_ci_hi) - np.array(k1_by_type_means)],
            fmt="none", color="black", capsize=4, lw=1.5)

# K=8 bars
b8 = ax.bar(x + width/2, k8_by_type_means, width, color="#1565C0", edgecolor="k", alpha=0.9, label="K=8")
ax.errorbar(x + width/2, k8_by_type_means,
            yerr=[np.array(k8_by_type_means) - np.array(k8_by_type_ci_lo),
                  np.array(k8_by_type_ci_hi) - np.array(k8_by_type_means)],
            fmt="none", color="black", capsize=4, lw=1.5)

# Add Wilcoxon annotations
for xi, t in enumerate(TYPES):
    ax.text(xi, max(k1_by_type_means[xi], k8_by_type_means[xi]) + 4,
            wilcoxon_ps[t], ha="center", fontsize=8, style="italic")

ax.set_xticks(x)
ax.set_xticklabels([TYPE_LABELS[t] for t in TYPES], fontsize=9)
ax.set_ylabel("Mean Risk Score", fontsize=11)
ax.set_title("K=1 vs K=8: Risk by Type\n(bars = 95% bootstrap CI, Wilcoxon p-values)", fontsize=11)
ax.legend(fontsize=10)
ax.axhline(10, color="green", linestyle="--", lw=1, alpha=0.5)
ax.grid(axis="y", alpha=0.35)
ax.set_ylim(0, 165)

fig.tight_layout()
out = RESULTS / "final_summary_plot.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ─── Figure 3: K=1 vs K=8 scatter with per-clip improvement annotations ───────
fig, ax = plt.subplots(figsize=(8, 7))

TYPE_COLORS = {"static": "#66BB6A", "locomotion": "#26A69A",
               "expressive": "#5C6BC0", "whole_body": "#EF5350"}

k1_map = {r["clip"]: float(r["full_risk"]) for r in guided if int(r["K"]) == 1}
k8_map = {r["clip"]: float(r["full_risk"]) for r in guided if int(r["K"]) == 8}
type_map = {r["clip"]: r["type"] for r in guided}

for mtype, color in TYPE_COLORS.items():
    clips = [c for c, t in type_map.items() if t == mtype and c in k1_map and c in k8_map]
    xs = [k1_map[c] for c in clips]
    ys = [k8_map[c] for c in clips]
    ax.scatter(xs, ys, color=color, s=70, alpha=0.85, edgecolors="k", lw=0.5,
               label={"static": "Static", "locomotion": "Locomotion",
                      "expressive": "Expressive", "whole_body": "Whole-body"}[mtype])

maxv = max(max(k1_map.values()), max(k8_map.values())) * 1.05
ax.plot([0, maxv], [0, maxv], "k--", lw=1, alpha=0.4, label="Equal")
ax.axhline(10, color="green", lw=1, linestyle=":", alpha=0.7, label="Accept threshold")
ax.set_xlabel("K=1 risk (deterministic baseline)", fontsize=11)
ax.set_ylabel("K=8 risk (best-of-8)", fontsize=11)
n_improved = sum(1 for c in k1_map if c in k8_map and k8_map[c] < k1_map[c])
ax.set_title(f"Per-Clip Risk: K=1 vs K=8\n(34/39 clips improved, Wilcoxon p<10⁻⁶, Cohen's d=1.09)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

out2 = RESULTS / "k1_vs_k8_scatter_final.png"
fig.tight_layout()
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")
