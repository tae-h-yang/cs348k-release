"""
Generate new plots for Iteration 2 results:
1. Diversity ablation (seed-only vs Gumbel-only vs combined)
2. Neural vs heuristic K=8 selection comparison
3. Updated risk-vs-K with bootstrap CIs
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path(__file__).parent.parent
RESULTS = REPO / "results"

# ─── Plot 1: Diversity ablation ───────────────────────────────────────────────
diversity_path = RESULTS / "diversity_ablation.csv"
if diversity_path.exists():
    with open(diversity_path) as f:
        div_rows = list(csv.DictReader(f))

    TYPES = ["static", "locomotion", "expressive", "whole_body"]
    TYPE_LABELS = {"static": "Static", "locomotion": "Locomotion",
                   "expressive": "Expressive", "whole_body": "Whole-body"}
    STRATEGIES = ["seed_only", "gumbel_only", "combined"]
    STRAT_LABELS = {"seed_only": "Seed-Only\n(argmax, 4 seeds)",
                    "gumbel_only": "Gumbel-Only\n(stoch, 1 seed)",
                    "combined": "Combined\n(current)"}
    STRAT_COLORS = {"seed_only": "#42A5F5", "gumbel_only": "#FFA726", "combined": "#66BB6A"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall bar
    ax = axes[0]
    overall_means = {}
    for s in STRATEGIES:
        risks = [float(r[f"risk_{s}"]) for r in div_rows if np.isfinite(float(r[f"risk_{s}"]))]
        overall_means[s] = np.mean(risks)

    # Add K=1 baseline
    k1_mean = 38.90
    x = np.arange(len(STRATEGIES) + 1)
    labels = ["K=1\nBaseline"] + [STRAT_LABELS[s] for s in STRATEGIES]
    vals = [k1_mean] + [overall_means[s] for s in STRATEGIES]
    colors = ["#EF5350"] + [STRAT_COLORS[s] for s in STRATEGIES]

    bars = ax.bar(x, vals, color=colors, edgecolor="k", alpha=0.88)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Risk Score (39 clips)", fontsize=11)
    ax.set_title("Diversity Source Ablation (K=4)", fontsize=12)
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(0, 48)

    # Per-type breakdown
    ax = axes[1]
    x2 = np.arange(len(TYPES))
    width = 0.22
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    # K=1 per type
    k1_by_type = {"static": 6.11, "locomotion": 21.91, "expressive": 24.47, "whole_body": 135.12}

    all_strat = ["K=1"] + STRATEGIES
    all_colors = ["#EF5350"] + [STRAT_COLORS[s] for s in STRATEGIES]
    all_labels = ["K=1"] + [STRAT_LABELS[s].split('\n')[0] for s in STRATEGIES]

    for oi, (strat, color, label) in enumerate(zip(all_strat, all_colors, all_labels)):
        means = []
        for tk in TYPES:
            if strat == "K=1":
                means.append(k1_by_type[tk])
            else:
                risks = [float(r[f"risk_{strat}"]) for r in div_rows
                         if r["type"] == tk and np.isfinite(float(r[f"risk_{strat}"]))]
                means.append(np.mean(risks) if risks else 0)
        ax.bar(x2 + offsets[oi], means, width, color=color, edgecolor="k", alpha=0.88, label=label)

    ax.set_xticks(x2)
    ax.set_xticklabels([TYPE_LABELS[t] for t in TYPES])
    ax.set_ylabel("Mean Risk Score", fontsize=11)
    ax.set_title("Diversity Ablation by Motion Type", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(0, 160)

    fig.tight_layout()
    out1 = RESULTS / "diversity_ablation_plot.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

# ─── Plot 2: Neural vs Heuristic K=8 ─────────────────────────────────────────
neural_path = RESULTS / "neural_guided_ablation.csv"
if neural_path.exists():
    with open(neural_path) as f:
        neu_rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Per-clip scatter
    ax = axes[0]
    neural_risks = [float(r["full_risk_neural"]) for r in neu_rows if np.isfinite(float(r["full_risk_neural"]))]
    heuristic_risks = [float(r["full_risk_heuristic"]) for r in neu_rows if np.isfinite(float(r["full_risk_heuristic"]))]

    TYPES_COLORS = {"static": "#66BB6A", "locomotion": "#26A69A",
                    "expressive": "#5C6BC0", "whole_body": "#EF5350"}
    for mtype, color in TYPES_COLORS.items():
        sub = [r for r in neu_rows if r["type"] == mtype]
        xs = [float(r["full_risk_heuristic"]) for r in sub]
        ys = [float(r["full_risk_neural"]) for r in sub]
        ax.scatter(xs, ys, color=color, s=60, alpha=0.85, edgecolors="k", lw=0.5,
                   label=TYPE_LABELS[mtype] if mtype in TYPE_LABELS else mtype)

    maxv = max(max(neural_risks), max(heuristic_risks)) * 1.05
    ax.plot([0, maxv], [0, maxv], "k--", lw=1, alpha=0.5, label="Equal")
    ax.set_xlabel("Heuristic K=8 risk (ground truth)", fontsize=11)
    ax.set_ylabel("Neural K=8 risk (surrogate-selected)", fontsize=11)

    n_agree = sum(1 for r in neu_rows if r["selection_agrees"] == "True")
    rho, _ = spearmanr(heuristic_risks, neural_risks)
    ax.set_title(f"Neural vs Heuristic WC-K8\nSelection agreement: {n_agree}/39 (31%)  ρ={rho:.3f}", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.text(0.04, 0.96, f"Neural K=8 mean: {np.mean(neural_risks):.1f}\nHeuristic K=8 mean: {np.mean(heuristic_risks):.1f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Bar chart: accepts by method
    ax = axes[1]
    methods = ["Heuristic-K8", "Neural-K8"]
    means_m = [np.mean(heuristic_risks), np.mean(neural_risks)]
    accepts = [sum(1 for r in neu_rows if r["action_heuristic"] == "accept"),
               sum(1 for r in neu_rows if r["action_neural"] == "accept")]
    colors_m = ["#1565C0", "#E65100"]
    x = np.arange(len(methods))
    ax.bar(x, means_m, color=colors_m, edgecolor="k", alpha=0.88)
    for xi, (m, a) in enumerate(zip(means_m, accepts)):
        ax.text(xi, m + 1, f"{m:.1f}\nacc={a}/39", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean Risk Score (39 clips)", fontsize=11)
    ax.set_title("Neural vs Heuristic WC-K8: Full-Clip Risk\n(neural selected via max-window aggregation)", fontsize=11)
    ax.grid(axis="y", alpha=0.35)

    fig.tight_layout()
    out2 = RESULTS / "neural_vs_heuristic_k8.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

# ─── Plot 3: Updated K=1 CI plot (from statistical_tests.json) ────────────────
stats_path = RESULTS / "statistical_tests.json"
if stats_path.exists():
    print("CI plots already generated by compute_statistics.py")

print("\nAll plots generated.")
