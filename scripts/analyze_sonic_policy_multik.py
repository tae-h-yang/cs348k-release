"""Analyze SONIC policy rollouts for a multi-K MotionBricks candidate set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_policy(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values(
        ["track_seconds", "mean_tracking_rmse"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ordered.iloc[0]


def make_pairs(rows: pd.DataFrame, guided: pd.DataFrame | None) -> pd.DataFrame:
    selected = rows.groupby("clip", group_keys=False).apply(select_policy, include_groups=False)
    selected = selected.reset_index()
    k1 = rows[rows.K == 1].set_index("clip")
    k8 = rows[rows.K == 8].set_index("clip") if 8 in set(rows.K) else None
    k16 = rows[rows.K == 16].set_index("clip") if 16 in set(rows.K) else None
    out = []
    for _, sel in selected.iterrows():
        clip = sel["clip"]
        row = {
            "clip": clip,
            "mode": clip.rsplit("_seed", 1)[0],
            "selected_K": int(sel.K),
            "selected_fell": bool(sel.fell),
            "selected_seconds": float(sel.track_seconds),
            "selected_rmse": float(sel.mean_tracking_rmse),
            "selected_torque_sat": float(sel.mean_torque_saturation_frac),
            "k1_fell": bool(k1.loc[clip].fell),
            "k1_seconds": float(k1.loc[clip].track_seconds),
            "k1_rmse": float(k1.loc[clip].mean_tracking_rmse),
            "k1_torque_sat": float(k1.loc[clip].mean_torque_saturation_frac),
        }
        if k8 is not None and clip in k8.index:
            row.update(
                {
                    "k8_seconds": float(k8.loc[clip].track_seconds),
                    "k8_rmse": float(k8.loc[clip].mean_tracking_rmse),
                    "k8_torque_sat": float(k8.loc[clip].mean_torque_saturation_frac),
                    "k8_fell": bool(k8.loc[clip].fell),
                }
            )
        if k16 is not None and clip in k16.index:
            row.update(
                {
                    "k16_seconds": float(k16.loc[clip].track_seconds),
                    "k16_rmse": float(k16.loc[clip].mean_tracking_rmse),
                    "k16_torque_sat": float(k16.loc[clip].mean_torque_saturation_frac),
                    "k16_fell": bool(k16.loc[clip].fell),
                }
            )
        out.append(row)
    pairs = pd.DataFrame(out)

    if guided is not None:
        risk = guided.pivot(index="clip", columns="K", values="full_risk")
        for k in sorted(set(rows.K)):
            if k in risk.columns:
                pairs[f"k{k}_id_risk"] = pairs["clip"].map(risk[k])
        pairs["selected_id_risk"] = [
            row.get(f"k{int(row.selected_K)}_id_risk", np.nan)
            for _, row in pairs.iterrows()
        ]
    return pairs


def summarize(rows: pd.DataFrame, pairs: pd.DataFrame) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for k in sorted(set(rows.K)):
        g = rows[rows.K == k]
        out.append(
            {
                "method": f"K={k}_ID_screened",
                "n": len(g),
                "fell_count": int(g.fell.sum()),
                "mean_track_seconds": float(g.track_seconds.mean()),
                "median_track_seconds": float(g.track_seconds.median()),
                "mean_tracking_rmse": float(g.mean_tracking_rmse.mean()),
                "median_tracking_rmse": float(g.mean_tracking_rmse.median()),
                "mean_torque_saturation_frac": float(g.mean_torque_saturation_frac.mean()),
            }
        )

    out.append(
        {
            "method": "SONIC_policy_selector_all_available_K",
            "n": len(pairs),
            "fell_count": int(pairs.selected_fell.sum()),
            "mean_track_seconds": float(pairs.selected_seconds.mean()),
            "median_track_seconds": float(pairs.selected_seconds.median()),
            "mean_tracking_rmse": float(pairs.selected_rmse.mean()),
            "median_tracking_rmse": float(pairs.selected_rmse.median()),
            "mean_torque_saturation_frac": float(pairs.selected_torque_sat.mean()),
        }
    )

    comparisons = [("selector_minus_K1", "selected_seconds", "selected_rmse")]
    if "k8_seconds" in pairs:
        comparisons.append(("K8_minus_K1", "k8_seconds", "k8_rmse"))
    if "k16_seconds" in pairs:
        comparisons.append(("K16_minus_K1", "k16_seconds", "k16_rmse"))
    for label, sec_col, rmse_col in comparisons:
        dsec = pairs[sec_col] - pairs.k1_seconds
        drmse = pairs[rmse_col] - pairs.k1_rmse
        try:
            p_sec = wilcoxon(dsec, alternative="greater").pvalue
        except ValueError:
            p_sec = np.nan
        try:
            p_rmse = wilcoxon(drmse, alternative="less").pvalue
        except ValueError:
            p_rmse = np.nan
        out.append(
            {
                "method": label,
                "n": len(pairs),
                "mean_track_seconds_delta": float(dsec.mean()),
                "median_track_seconds_delta": float(dsec.median()),
                "mean_tracking_rmse_delta": float(drmse.mean()),
                "median_tracking_rmse_delta": float(drmse.median()),
                "wilcoxon_seconds_p_greater": float(p_sec),
                "wilcoxon_rmse_p_less": float(p_rmse),
            }
        )

    if "k1_id_risk" in pairs and "k8_id_risk" in pairs:
        out.append(
            {
                "method": "inverse_dynamics_risk_K8_minus_K1",
                "n": int(pairs[["k1_id_risk", "k8_id_risk"]].dropna().shape[0]),
                "mean_full_risk_delta": float((pairs.k8_id_risk - pairs.k1_id_risk).mean()),
                "median_full_risk_delta": float((pairs.k8_id_risk - pairs.k1_id_risk).median()),
            }
        )
    return out


def plot(rows: pd.DataFrame, pairs: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.size": 10})
    ks = sorted(set(rows.K))
    labels = [f"K={k}" for k in ks] + ["policy\nselector"]
    seconds = [rows[rows.K == k].track_seconds for k in ks] + [pairs.selected_seconds]
    rmse = [rows[rows.K == k].mean_tracking_rmse for k in ks] + [pairs.selected_rmse]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    axes[0].boxplot(seconds, tick_labels=labels, showfliers=False)
    for i, vals in enumerate(seconds, start=1):
        axes[0].scatter(np.repeat(i, len(vals)), vals, s=9, alpha=0.25)
    axes[0].set_ylabel("Seconds before fall")
    axes[0].set_title("Closed-loop survival")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(rmse, tick_labels=labels, showfliers=False)
    axes[1].set_ylabel("Joint RMSE (rad)")
    axes[1].set_title("Tracking error")
    axes[1].grid(axis="y", alpha=0.25)

    counts = pairs.selected_K.value_counts().sort_index()
    axes[2].bar([f"K={int(k)}" for k in counts.index], counts.values, color="#4C78A8")
    axes[2].set_ylabel("Selected identities")
    axes[2].set_title("Policy-aware selector choices")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("SONIC policy-in-the-loop audit over K=1/4/8/16 screened variants")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_csv", type=Path, default=ROOT / "results" / "sonic_policy_mujoco_tracking_k_sweep.csv")
    parser.add_argument("--guided_csv", type=Path, default=ROOT / "results" / "guided_ablation_full.csv")
    parser.add_argument("--summary_csv", type=Path, default=ROOT / "results" / "sonic_policy_multik_selector_summary.csv")
    parser.add_argument("--by_clip_csv", type=Path, default=ROOT / "results" / "sonic_policy_multik_selector_by_clip.csv")
    parser.add_argument("--plot", type=Path, default=ROOT / "results" / "sonic_policy_multik_selector.png")
    args = parser.parse_args()

    rows = pd.read_csv(args.tracking_csv)
    guided = pd.read_csv(args.guided_csv) if args.guided_csv.exists() else None
    pairs = make_pairs(rows, guided)
    summary = summarize(rows, pairs)
    pairs.to_csv(args.by_clip_csv, index=False)
    write_dict_csv(args.summary_csv, summary)
    plot(rows, pairs, args.plot)
    print(f"Wrote {args.by_clip_csv}")
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.plot}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
