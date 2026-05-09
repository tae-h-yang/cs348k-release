"""
Statistical analysis of best-of-K ablation results.
Addresses reviewer R3: Wilcoxon signed-rank tests, bootstrap 95% CIs.
Also addresses R7a: K=1 deterministic vs stochastic baseline comparison.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).parent.parent
RESULTS = REPO / "results"
STATS_OUT = RESULTS / "statistical_tests.json"

# ─── Load ablation CSV ────────────────────────────────────────────────────────

def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

rows = load_csv(RESULTS / "guided_ablation_full.csv")
rows_steered = load_csv(RESULTS / "steered_vs_wc_ablation.csv")

TYPES = ["static", "locomotion", "expressive", "whole_body"]
K_VALS = [1, 4, 8, 16]

# ─── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(arr, n_boot=5000, ci=0.95, stat=np.mean):
    rng = np.random.default_rng(42)
    arr = np.array(arr)
    boot_stats = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boot_stats, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_stats, 100 * (1 + ci) / 2)
    return float(lo), float(hi)

# ─── Extract per-clip risks for each K ────────────────────────────────────────

def get_risks_for_k(rows, k, mtype=None):
    sub = [r for r in rows if int(r["K"]) == k]
    if mtype:
        sub = [r for r in sub if r["type"] == mtype]
    return [(r["clip"], float(r["full_risk"])) for r in sub]

# Build per-clip paired K=1 vs K=k dictionaries
def paired_risks(rows, k_target):
    """Return paired (risk_k1, risk_kN) for same clips."""
    k1 = {r["clip"]: float(r["full_risk"]) for r in rows if int(r["K"]) == 1}
    kn = {r["clip"]: float(r["full_risk"]) for r in rows if int(r["K"]) == k_target}
    clips = sorted(set(k1.keys()) & set(kn.keys()))
    return [(k1[c], kn[c]) for c in clips]

results = {}

print("=" * 60)
print("STATISTICAL ANALYSIS — BEST-OF-K ABLATION")
print("=" * 60)

# ─── 1. Overall K=1 vs K=8 Wilcoxon ─────────────────────────────────────────
print("\n--- K=1 vs K=k (all clips, Wilcoxon signed-rank) ---")
for k_target in [4, 8, 16]:
    pairs = paired_risks(rows, k_target)
    k1_risks = np.array([p[0] for p in pairs])
    kn_risks = np.array([p[1] for p in pairs])
    diffs = k1_risks - kn_risks  # positive = improvement
    stat, pval = wilcoxon(diffs, alternative="greater")
    ci_lo, ci_hi = bootstrap_ci(kn_risks)
    mean_k1 = np.mean(k1_risks)
    mean_kn = np.mean(kn_risks)
    pct_reduction = 100 * (mean_k1 - mean_kn) / mean_k1
    n_improved = int(np.sum(kn_risks < k1_risks))
    n_total = len(pairs)
    binom_p = binomtest(n_improved, n_total, 0.5, alternative="greater").pvalue

    print(f"  K=1 vs K={k_target}: mean {mean_k1:.2f} → {mean_kn:.2f} ({pct_reduction:.1f}% red.)"
          f"  Wilcoxon p={pval:.4f}  binom({n_improved}/{n_total}) p={binom_p:.4f}"
          f"  95%CI=[{ci_lo:.2f}, {ci_hi:.2f}]")

    results[f"K1_vs_K{k_target}_wilcoxon_p"] = float(pval)
    results[f"K1_vs_K{k_target}_binom_p"] = float(binom_p)
    results[f"K{k_target}_mean_risk"] = float(mean_kn)
    results[f"K{k_target}_ci_95"] = [float(ci_lo), float(ci_hi)]
    results[f"K1_vs_K{k_target}_n_improved"] = n_improved
    results[f"K1_vs_K{k_target}_n_total"] = n_total

# ─── 2. Per-type K=1 vs K=8 ──────────────────────────────────────────────────
print("\n--- K=1 vs K=8 by motion type ---")
type_stats = {}
for mtype in TYPES:
    pairs = [(k1, kn) for (k1, kn) in paired_risks(rows, 8) if True]
    # Need to filter by type
    k1_m = {r["clip"]: float(r["full_risk"]) for r in rows if int(r["K"]) == 1 and r["type"] == mtype}
    k8_m = {r["clip"]: float(r["full_risk"]) for r in rows if int(r["K"]) == 8 and r["type"] == mtype}
    clips = sorted(set(k1_m.keys()) & set(k8_m.keys()))
    if not clips:
        continue
    k1_arr = np.array([k1_m[c] for c in clips])
    k8_arr = np.array([k8_m[c] for c in clips])

    if len(clips) >= 3:
        try:
            stat, pval = wilcoxon(k1_arr - k8_arr, alternative="greater")
        except Exception:
            pval = float("nan")
    else:
        pval = float("nan")

    ci_lo, ci_hi = bootstrap_ci(k8_arr)
    mean_k1 = np.mean(k1_arr)
    mean_k8 = np.mean(k8_arr)
    pct_red = 100 * (mean_k1 - mean_k8) / mean_k1 if mean_k1 > 0 else 0.0

    print(f"  {mtype:15s}: K=1={mean_k1:.2f}→K=8={mean_k8:.2f} ({pct_red:+.1f}%)"
          f"  Wilcoxon p={pval:.4f}  n={len(clips)}"
          f"  95%CI=[{ci_lo:.2f}, {ci_hi:.2f}]")

    type_stats[mtype] = {
        "k1_mean": float(mean_k1), "k8_mean": float(mean_k8),
        "pct_reduction": float(pct_red),
        "wilcoxon_p": float(pval),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "n": len(clips),
    }

results["by_type"] = type_stats

# ─── 3. WC-K4 vs PS-K4 ───────────────────────────────────────────────────────
print("\n--- Compute-matched WC-K4 vs PS-K4 ---")
wc_risks = []
ps_risks = []
for r in rows_steered:
    try:
        wc = float(r["risk_wc"])
        ps = float(r["risk_ps"])
        if np.isfinite(wc) and np.isfinite(ps):
            wc_risks.append(wc)
            ps_risks.append(ps)
    except (ValueError, KeyError):
        continue

wc_risks = np.array(wc_risks)
ps_risks = np.array(ps_risks)
diffs = ps_risks - wc_risks  # positive = WC better
n_wc_better = int(np.sum(wc_risks < ps_risks))
n_total_paired = len(wc_risks)

try:
    stat_wc, pval_wc = wilcoxon(diffs, alternative="greater")
except Exception:
    pval_wc = float("nan")

binom_wc = binomtest(n_wc_better, n_total_paired, 0.5, alternative="greater").pvalue
ci_wc_lo, ci_wc_hi = bootstrap_ci(wc_risks)
ci_ps_lo, ci_ps_hi = bootstrap_ci(ps_risks)

print(f"  WC-K4: mean={np.mean(wc_risks):.2f}  95%CI=[{ci_wc_lo:.2f},{ci_wc_hi:.2f}]")
print(f"  PS-K4: mean={np.mean(ps_risks):.2f}  95%CI=[{ci_ps_lo:.2f},{ci_ps_hi:.2f}]")
print(f"  WC better on {n_wc_better}/{n_total_paired}  Wilcoxon p={pval_wc:.4f}  binom p={binom_wc:.4f}")

results["wc_vs_ps"] = {
    "wc_mean": float(np.mean(wc_risks)),
    "ps_mean": float(np.mean(ps_risks)),
    "wc_ci_95": [float(ci_wc_lo), float(ci_wc_hi)],
    "ps_ci_95": [float(ci_ps_lo), float(ci_ps_hi)],
    "wilcoxon_p": float(pval_wc),
    "binom_p": float(binom_wc),
    "n_wc_better": n_wc_better,
    "n_total": n_total_paired,
}

# ─── 4. Effect size (Cohen's d for paired differences) ───────────────────────
print("\n--- Effect sizes (Cohen's d) ---")
pairs_k1_k8 = paired_risks(rows, 8)
k1_arr = np.array([p[0] for p in pairs_k1_k8])
k8_arr = np.array([p[1] for p in pairs_k1_k8])
diff_arr = k1_arr - k8_arr
cohens_d = np.mean(diff_arr) / np.std(diff_arr, ddof=1)
print(f"  K=1 vs K=8: Cohen's d = {cohens_d:.3f} (>0.8 = large effect)")
results["cohens_d_k1_vs_k8"] = float(cohens_d)

# WC vs PS
diff_wcps = ps_risks - wc_risks
cohens_d_wcps = np.mean(diff_wcps) / np.std(diff_wcps, ddof=1)
print(f"  WC-K4 vs PS-K4: Cohen's d = {cohens_d_wcps:.3f}")
results["cohens_d_wc_vs_ps"] = float(cohens_d_wcps)

# ─── 5. K=1 stochastic baseline (R7a) ────────────────────────────────────────
# Check if stochastic K=1 data exists in guided_ablation_full.csv
# K=1 with stochastic uses k0_seg_risk = same as k1 (no diversity)
# We can't compute this without running it, but we can note it in results
print("\n--- K=1 baseline breakdown ---")
k1_rows = [r for r in rows if int(r["K"]) == 1]
print(f"  K=1 clips: {len(k1_rows)}")
print(f"  All K=1 are deterministic (argmax + base seed, no Gumbel)")
print(f"  Stochastic K=1 not run — noted as gap in reviewer response")

# ─── 6. Figure: K=8 risk distribution with bootstrap CIs ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
k_vals = [1, 4, 8, 16]
means = []
ci_los = []
ci_his = []
for k in k_vals:
    risks = [float(r["full_risk"]) for r in rows if int(r["K"]) == k]
    means.append(np.mean(risks))
    lo, hi = bootstrap_ci(risks)
    ci_los.append(lo)
    ci_his.append(hi)

x = np.arange(len(k_vals))
ax.bar(x, means, color="#1565C0", alpha=0.8, edgecolor="k")
ax.errorbar(x, means,
            yerr=[np.array(means)-np.array(ci_los), np.array(ci_his)-np.array(means)],
            fmt="none", color="black", capsize=5, lw=2)
for xi, (m, lo, hi) in enumerate(zip(means, ci_los, ci_his)):
    ax.text(xi, m + 1.5, f"{m:.1f}\n[{lo:.1f},{hi:.1f}]", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"K={k}" for k in k_vals])
ax.set_ylabel("Mean Risk Score (39 clips)", fontsize=11)
ax.set_title("Best-of-K: Mean Risk with 95% Bootstrap CI", fontsize=12)
ax.grid(axis="y", alpha=0.35)

# Per-type CI plot
ax = axes[1]
type_labels = ["Static", "Locomotion", "Expressive", "Whole-body"]
type_keys = ["static", "locomotion", "expressive", "whole_body"]
colors_k1 = ["#90CAF9"] * 4
colors_k8 = ["#1565C0"] * 4

x2 = np.arange(len(type_keys))
width = 0.35

for ti, (tk, label) in enumerate(zip(type_keys, type_labels)):
    k1_risks = [float(r["full_risk"]) for r in rows if int(r["K"]) == 1 and r["type"] == tk]
    k8_risks = [float(r["full_risk"]) for r in rows if int(r["K"]) == 8 and r["type"] == tk]
    if not k1_risks:
        continue
    m1 = np.mean(k1_risks)
    m8 = np.mean(k8_risks)
    lo1, hi1 = bootstrap_ci(k1_risks)
    lo8, hi8 = bootstrap_ci(k8_risks)

    b1 = ax.bar(x2[ti] - width/2, m1, width, color="#90CAF9", edgecolor="k", alpha=0.9)
    b8 = ax.bar(x2[ti] + width/2, m8, width, color="#1565C0", edgecolor="k", alpha=0.9)
    ax.errorbar(x2[ti] - width/2, m1, yerr=[[m1-lo1],[hi1-m1]], fmt="none", color="k", capsize=4)
    ax.errorbar(x2[ti] + width/2, m8, yerr=[[m8-lo8],[hi8-m8]], fmt="none", color="k", capsize=4)

ax.set_xticks(x2)
ax.set_xticklabels(type_labels)
ax.set_ylabel("Mean Risk Score", fontsize=11)
ax.set_title("K=1 vs K=8 by Motion Type (95% Bootstrap CI)", fontsize=12)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#90CAF9", label="K=1"), Patch(color="#1565C0", label="K=8")],
          fontsize=10)
ax.grid(axis="y", alpha=0.35)

fig.tight_layout()
out = RESULTS / "statistical_ci_plot.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

# ─── Save JSON ────────────────────────────────────────────────────────────────
with open(STATS_OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {STATS_OUT}")

print("\n" + "=" * 60)
print("SUMMARY FOR REVIEWER RESPONSE")
print("=" * 60)
print(f"K=1 vs K=8: Wilcoxon p={results['K1_vs_K8_wilcoxon_p']:.5f} (<0.05? {results['K1_vs_K8_wilcoxon_p']<0.05})")
print(f"K=1 vs K=8: {results['K1_vs_K8_n_improved']}/{results['K1_vs_K8_n_total']} clips improved")
print(f"K=1 vs K=8: Cohen's d={results['cohens_d_k1_vs_k8']:.3f}")
print(f"WC vs PS:   Wilcoxon p={results['wc_vs_ps']['wilcoxon_p']:.5f} (<0.05? {results['wc_vs_ps']['wilcoxon_p']<0.05})")
print(f"WC better:  {results['wc_vs_ps']['n_wc_better']}/{results['wc_vs_ps']['n_total']} clips")
