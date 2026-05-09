"""Summarize and plot SONIC policy-in-MuJoCo tracking results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "results" / "sonic_policy_mujoco_tracking_210.csv"
DEFAULT_GUIDED = ROOT / "results" / "guided_ablation_extended.csv"
DEFAULT_SUMMARY = ROOT / "results" / "sonic_policy_selector_summary.csv"
DEFAULT_BY_CLIP = ROOT / "results" / "sonic_policy_selector_by_clip.csv"
DEFAULT_PLOT = ROOT / "results" / "sonic_policy_tracking_k1_k8_selector.png"


def paired_table(rows: pd.DataFrame) -> pd.DataFrame:
    k1 = rows[rows.K == 1].set_index("clip")
    k8 = rows[rows.K == 8].set_index("clip")
    clips = sorted(set(k1.index) & set(k8.index))
    out = []
    for clip in clips:
        a = k1.loc[clip]
        b = k8.loc[clip]
        # Controller-aware selector: choose the candidate that survives longer,
        # then the lower tracking error if survival ties.
        if (b.track_seconds > a.track_seconds) or (
            np.isclose(b.track_seconds, a.track_seconds) and b.mean_tracking_rmse < a.mean_tracking_rmse
        ):
            chosen = b
            chosen_k = 8
        else:
            chosen = a
            chosen_k = 1
        out.append(
            {
                "clip": clip,
                "mode": clip.rsplit("_seed", 1)[0],
                "chosen_K": chosen_k,
                "k1_fell": bool(a.fell),
                "k8_fell": bool(b.fell),
                "selected_fell": bool(chosen.fell),
                "k1_seconds": a.track_seconds,
                "k8_seconds": b.track_seconds,
                "selected_seconds": chosen.track_seconds,
                "k1_rmse": a.mean_tracking_rmse,
                "k8_rmse": b.mean_tracking_rmse,
                "selected_rmse": chosen.mean_tracking_rmse,
                "k1_torque_sat": a.mean_torque_saturation_frac,
                "k8_torque_sat": b.mean_torque_saturation_frac,
                "selected_torque_sat": chosen.mean_torque_saturation_frac,
            }
        )
    return pd.DataFrame(out)


def summarize(pairs: pd.DataFrame, guided: pd.DataFrame | None) -> list[dict[str, object]]:
    groups = {
        "MotionBricks_K1": ("k1_seconds", "k1_rmse", "k1_torque_sat"),
        "ID_screened_K8": ("k8_seconds", "k8_rmse", "k8_torque_sat"),
        "SONIC_policy_selector": ("selected_seconds", "selected_rmse", "selected_torque_sat"),
    }
    rows: list[dict[str, object]] = []
    for name, (sec_col, rmse_col, sat_col) in groups.items():
        rows.append(
            {
                "method": name,
                "n": len(pairs),
                "mean_track_seconds": float(pairs[sec_col].mean()),
                "median_track_seconds": float(pairs[sec_col].median()),
                "mean_tracking_rmse": float(pairs[rmse_col].mean()),
                "median_tracking_rmse": float(pairs[rmse_col].median()),
                "mean_torque_saturation_frac": float(pairs[sat_col].mean()),
                "fell_count": int(pairs[{"k1_seconds": "k1_fell", "k8_seconds": "k8_fell", "selected_seconds": "selected_fell"}[sec_col]].sum()),
            }
        )

    for label, col in [("K8_minus_K1", "k8_seconds"), ("selector_minus_K1", "selected_seconds")]:
        delta_seconds = pairs[col] - pairs["k1_seconds"]
        delta_rmse = (pairs["k8_rmse"] if label == "K8_minus_K1" else pairs["selected_rmse"]) - pairs["k1_rmse"]
        try:
            p_sec = wilcoxon(delta_seconds, alternative="greater").pvalue
        except ValueError:
            p_sec = np.nan
        try:
            p_rmse = wilcoxon(delta_rmse, alternative="less").pvalue
        except ValueError:
            p_rmse = np.nan
        rows.append(
            {
                "method": label,
                "n": len(pairs),
                "mean_track_seconds": float(delta_seconds.mean()),
                "median_track_seconds": float(delta_seconds.median()),
                "mean_tracking_rmse": float(delta_rmse.mean()),
                "median_tracking_rmse": float(delta_rmse.median()),
                "mean_torque_saturation_frac": np.nan,
                "fell_count": "",
                "wilcoxon_seconds_p_greater": float(p_sec),
                "wilcoxon_rmse_p_less": float(p_rmse),
            }
        )

    if guided is not None:
        g = guided.pivot(index="clip", columns="K", values="full_risk")
        merged = pairs.join(g, on="clip", rsuffix="_risk")
        valid = merged[[1, 8]].dropna()
        if len(valid):
            rows.append(
                {
                    "method": "inverse_dynamics_risk_K8_minus_K1",
                    "n": len(valid),
                    "mean_track_seconds": np.nan,
                    "median_track_seconds": np.nan,
                    "mean_tracking_rmse": np.nan,
                    "median_tracking_rmse": np.nan,
                    "mean_torque_saturation_frac": np.nan,
                    "fell_count": "",
                    "mean_full_risk_delta": float((valid[8] - valid[1]).mean()),
                    "median_full_risk_delta": float((valid[8] - valid[1]).median()),
                }
            )
    return rows


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(pairs: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    labels = ["K=1\nbaseline", "K=8\nID screen", "policy\nselector"]
    sec = [pairs.k1_seconds, pairs.k8_seconds, pairs.selected_seconds]
    rmse = [pairs.k1_rmse, pairs.k8_rmse, pairs.selected_rmse]
    chosen_counts = pairs.chosen_K.value_counts().to_dict()

    axes[0].boxplot(sec, tick_labels=labels, showfliers=False)
    axes[0].scatter(np.repeat(1, len(pairs)), pairs.k1_seconds, s=8, alpha=0.25)
    axes[0].scatter(np.repeat(2, len(pairs)), pairs.k8_seconds, s=8, alpha=0.25)
    axes[0].scatter(np.repeat(3, len(pairs)), pairs.selected_seconds, s=8, alpha=0.25)
    axes[0].set_ylabel("Seconds before fall")
    axes[0].set_title("Closed-loop survival")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(rmse, tick_labels=labels, showfliers=False)
    axes[1].set_ylabel("Joint RMSE (rad)")
    axes[1].set_title("Tracking quality")
    axes[1].grid(axis="y", alpha=0.25)

    by_mode = pairs.groupby("mode")[["k1_seconds", "k8_seconds", "selected_seconds"]].mean().sort_index()
    x = np.arange(len(by_mode))
    axes[2].bar(x - 0.25, by_mode.k1_seconds, width=0.25, label="K=1")
    axes[2].bar(x, by_mode.k8_seconds, width=0.25, label="K=8")
    axes[2].bar(x + 0.25, by_mode.selected_seconds, width=0.25, label="selector")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(by_mode.index, rotation=70, ha="right", fontsize=7)
    axes[2].set_ylabel("Mean seconds")
    axes[2].set_title(f"By mode; selector chose K1={chosen_counts.get(1,0)}, K8={chosen_counts.get(8,0)}")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("SONIC learned-policy MuJoCo screening: inverse-dynamics K=8 does not improve execution")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_csv", type=Path, default=DEFAULT_IN)
    parser.add_argument("--guided_csv", type=Path, default=DEFAULT_GUIDED)
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--by_clip_csv", type=Path, default=DEFAULT_BY_CLIP)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()

    rows = pd.read_csv(args.tracking_csv)
    pairs = paired_table(rows)
    guided = pd.read_csv(args.guided_csv) if args.guided_csv.exists() else None
    summary = summarize(pairs, guided)
    pairs.to_csv(args.by_clip_csv, index=False)
    write_dict_csv(args.summary_csv, summary)
    plot(pairs, args.plot)
    print(f"Wrote {args.by_clip_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.plot}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
