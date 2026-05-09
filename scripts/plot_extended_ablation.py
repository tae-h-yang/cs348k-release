"""Plots for the 105-identity extended K=1 vs K=8 ablation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "results" / "guided_ablation_extended.csv"
OUT_RISK = ROOT / "results" / "guided_extended_k1_vs_k8.png"
OUT_ACTIONS = ROOT / "results" / "guided_extended_action_counts.png"

TYPE_ORDER = ["static", "locomotion", "expressive", "whole_body"]
TYPE_COLORS = {
    "static": "#7E57C2",
    "locomotion": "#2E7D32",
    "expressive": "#EF6C00",
    "whole_body": "#C62828",
}
ACTION_ORDER = ["accept", "repair_or_rerank", "reject_or_regenerate"]
ACTION_LABELS = ["Accept", "Review", "Reject"]
ACTION_COLORS = ["#43A047", "#FB8C00", "#E53935"]


def load_rows() -> list[dict]:
    with open(IN_CSV) as f:
        return list(csv.DictReader(f))


def paired_rows(rows: list[dict]) -> list[tuple[dict, dict]]:
    by_clip: dict[str, dict[int, dict]] = {}
    for row in rows:
        by_clip.setdefault(row["clip"], {})[int(row["K"])] = row
    return [(values[1], values[8]) for values in by_clip.values() if 1 in values and 8 in values]


def plot_risk(rows: list[dict]) -> None:
    pairs = paired_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    ax = axes[0]
    max_risk = max(float(r["full_risk"]) for pair in pairs for r in pair)
    ax.plot([0, max_risk], [0, max_risk], color="#777777", lw=1, ls="--")
    for motion_type in TYPE_ORDER:
        xs = [float(k1["full_risk"]) for k1, k8 in pairs if k1["type"] == motion_type]
        ys = [float(k8["full_risk"]) for k1, k8 in pairs if k1["type"] == motion_type]
        if xs:
            ax.scatter(xs, ys, s=42, alpha=0.82, label=motion_type.replace("_", " "),
                       color=TYPE_COLORS[motion_type], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("K=1 heuristic risk")
    ax.set_ylabel("K=8 selected heuristic risk")
    ax.set_title("105 Motion Identities: K=1 vs K=8")
    ax.set_xlim(left=-2)
    ax.set_ylim(bottom=-2)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    x = np.arange(len(TYPE_ORDER))
    width = 0.34
    k1_means = []
    k8_means = []
    for motion_type in TYPE_ORDER:
        k1_vals = [float(k1["full_risk"]) for k1, k8 in pairs if k1["type"] == motion_type]
        k8_vals = [float(k8["full_risk"]) for k1, k8 in pairs if k1["type"] == motion_type]
        k1_means.append(np.mean(k1_vals) if k1_vals else np.nan)
        k8_means.append(np.mean(k8_vals) if k8_vals else np.nan)
    ax.bar(x - width / 2, k1_means, width, label="K=1", color="#B0BEC5", edgecolor="black")
    ax.bar(x + width / 2, k8_means, width, label="K=8", color="#42A5F5", edgecolor="black")
    ax.set_xticks(x, [t.replace("_", "\n") for t in TYPE_ORDER])
    ax.set_ylabel("Mean heuristic risk")
    ax.set_title("Risk by Motion Type")
    ax.legend(frameon=False)

    fig.savefig(OUT_RISK, dpi=180)
    print(f"Saved: {OUT_RISK}")


def plot_actions(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, constrained_layout=True)
    for ax, k in zip(axes, [1, 8]):
        group = [row for row in rows if int(row["K"]) == k]
        bottom = np.zeros(len(TYPE_ORDER))
        x = np.arange(len(TYPE_ORDER))
        for action, label, color in zip(ACTION_ORDER, ACTION_LABELS, ACTION_COLORS):
            vals = [sum(r["type"] == motion_type and r["action"] == action for r in group)
                    for motion_type in TYPE_ORDER]
            ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="black")
            bottom += np.array(vals)
        ax.set_title(f"K={k}")
        ax.set_xticks(x, [t.replace("_", "\n") for t in TYPE_ORDER])
        ax.set_ylabel("Clip count")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Heuristic Action Labels on 105 Motion Identities")
    fig.savefig(OUT_ACTIONS, dpi=180)
    print(f"Saved: {OUT_ACTIONS}")


def main() -> None:
    rows = load_rows()
    plot_risk(rows)
    plot_actions(rows)


if __name__ == "__main__":
    main()
