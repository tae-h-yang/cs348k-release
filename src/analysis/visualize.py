"""
Visualization utilities for kinematic-to-dynamic gap analysis results.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe; must come before pyplot import
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from physics_eval.metrics import ClipMetrics, InverseDynamicsClipMetrics

RESULTS_DIR = Path(__file__).parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Joint names matching the actuator order in g1_29dof.xml
JOINT_NAMES = [
    "L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r",
    "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r",
    "W_yaw", "W_roll", "W_pitch",
    "L_sh_p", "L_sh_r", "L_sh_y", "L_elbow", "L_wr_r", "L_wr_p", "L_wr_y",
    "R_sh_p", "R_sh_r", "R_sh_y", "R_elbow", "R_wr_r", "R_wr_p", "R_wr_y",
]


def _color_by_type(motion_type: str) -> str:
    palette = {
        "static":      "#4CAF50",
        "locomotion":  "#2196F3",
        "expressive":  "#9C27B0",
        "whole_body":  "#FF9800",
        "adversarial": "#F44336",
        "unknown":     "#9E9E9E",
    }
    return palette.get(motion_type, "#9E9E9E")


def plot_tracking_error_by_type(results: List[ClipMetrics], save: bool = True):
    """Bar chart: mean joint tracking RMSE grouped by motion type (physics mode only)."""
    phys = [r for r in results if r.mode == "physics"]
    by_type: Dict[str, List[float]] = {}
    for r in phys:
        by_type.setdefault(r.motion_type, []).append(r.mean_tracking_rmse)
    if not by_type:
        return None

    types  = sorted(by_type.keys())
    means  = [np.mean(by_type[t]) for t in types]
    stds   = [np.std(by_type[t])  for t in types]
    colors = [_color_by_type(t)   for t in types]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(types, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Mean Joint Tracking RMSE (rad)")
    ax.set_title("Kinematic-to-Dynamic Gap: Tracking Error by Motion Type")
    ax.set_xlabel("Motion Type")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if save:
        path = RESULTS_DIR / "tracking_error_by_type.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_fall_rate_by_type(results: List[ClipMetrics], fps: float = 30.0,
                           save: bool = True):
    """
    Bar chart: mean time-to-fall (seconds) by motion type, physics mode only.
    Shows which motion styles survive longest before the gap causes failure.
    """
    phys = [r for r in results if r.mode == "physics"]
    by_type: Dict[str, List[float]] = {}
    fall_rate: Dict[str, float] = {}
    for r in phys:
        ttf = (r.time_to_fall / fps) if r.fell else (r.n_frames / fps)
        by_type.setdefault(r.motion_type, []).append(ttf)
    for t, values in by_type.items():
        clips = [r for r in phys if r.motion_type == t]
        fall_rate[t] = 100.0 * sum(r.fell for r in clips) / len(clips)
    if not by_type:
        return None

    types  = sorted(by_type.keys())
    means  = [np.mean(by_type[t]) for t in types]
    stds   = [np.std(by_type[t])  for t in types]
    colors = [_color_by_type(t)   for t in types]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(types, means, yerr=stds, color=colors, capsize=5,
           edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Mean Time-to-Fall (s)")
    ax.set_title("Physics Execution: Mean Time-to-Fall by Motion Type\n"
                 "(survived clips use full clip duration)")
    ax.set_xlabel("Motion Type")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, t in enumerate(types):
        ax.text(i, means[i], f"{fall_rate[t]:.0f}% fell", ha="center", va="bottom",
                fontsize=8, color="#263238")

    if save:
        path = RESULTS_DIR / "fall_rate_by_type.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_root_error_over_time(results: List[ClipMetrics], save: bool = True):
    """
    Line plot: root position error over time for all physics-mode clips.
    Lines are colored by motion type; legend shows types only (not individual clips).
    Y-axis capped at 15m to keep scale readable despite outlier drift.
    """
    import matplotlib.patches as mpatches
    phys = [r for r in results if r.mode == "physics"]
    if not phys:
        return None

    fig, ax = plt.subplots(figsize=(11, 5))

    seen_types = set()
    for r in phys:
        errors = [min(fm.root_pos_error, 15.0) for fm in r.frame_metrics]
        ax.plot(errors, color=_color_by_type(r.motion_type), alpha=0.55,
                linewidth=1.0)
        seen_types.add(r.motion_type)

    patches = [mpatches.Patch(color=_color_by_type(t), label=t)
               for t in sorted(seen_types)]
    ax.legend(handles=patches, fontsize=9, loc="upper left")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Root Position Error (m, capped at 15)")
    ax.set_title(f"Root Position Drift: Kinematic Target vs Physics Execution\n"
                 f"({len(phys)} clips; colored by motion type)")
    ax.grid(linestyle="--", alpha=0.4)

    if save:
        path = RESULTS_DIR / "root_error_over_time.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_per_joint_error_heatmap(results: List[ClipMetrics], save: bool = True):
    """
    Heatmap: mean per-joint tracking error (rad) for each clip × joint.
    Reveals which joints fail most and whether failure is motion-type-specific.
    Only uses physics mode results.
    """
    phys = [r for r in results if r.mode == "physics"]
    if not phys:
        return None

    data_matrix = np.stack([r.mean_per_joint_error for r in phys], axis=0)  # (N_clips, 29)
    clip_labels  = [f"{r.clip_name}\n[{r.motion_type}]" for r in phys]

    fig, ax = plt.subplots(figsize=(max(10, len(JOINT_NAMES) * 0.45),
                                    max(4, len(phys) * 0.4 + 1.5)))
    im = ax.imshow(data_matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="Mean |error| (rad)")

    ax.set_xticks(range(len(JOINT_NAMES)))
    ax.set_xticklabels(JOINT_NAMES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(phys)))
    ax.set_yticklabels(clip_labels, fontsize=7)
    ax.set_title("Per-Joint Tracking Error Heatmap (physics execution)")
    fig.tight_layout()

    if save:
        path = RESULTS_DIR / "per_joint_error_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_time_to_fall(results: List[ClipMetrics], fps: float = 30.0, save: bool = True):
    """
    Bar chart: time-to-fall (seconds) per clip, grouped by motion type.
    Clips that survive the full sequence show the clip duration as a filled bar.
    Physics mode only.
    """
    phys = [r for r in results if r.mode == "physics"]
    if not phys:
        return None

    clips   = [r.clip_name for r in phys]
    durations = [r.n_frames / fps for r in phys]
    ttf       = [(r.time_to_fall / fps) if r.fell else (r.n_frames / fps) for r in phys]
    colors    = [_color_by_type(r.motion_type) for r in phys]
    hatches   = ['' if r.fell else '///' for r in phys]

    x = np.arange(len(clips))
    fig, ax = plt.subplots(figsize=(max(8, len(clips) * 0.7), 5))

    # Full clip duration (background)
    ax.bar(x, durations, color="#ECEFF1", edgecolor="#90A4AE", linewidth=0.8, label="Clip duration")
    # Time to fall (foreground)
    bars = ax.bar(x, ttf, color=colors, edgecolor="black", linewidth=0.8,
                  label="Time to fall (physics)")
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(clips, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Time (s)")
    ax.set_title("Physical Feasibility: Time-to-Fall Under PD Tracking\n"
                 "(hatched = survived full clip)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if save:
        path = RESULTS_DIR / "time_to_fall.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_gap_comparison(physics_results: List[ClipMetrics],
                        kinematic_results: List[ClipMetrics],
                        save: bool = True):
    """
    Side-by-side: kinematic joint violations vs physics tracking RMSE per motion type.
    Kinematic mode is zero-gap by definition (qpos set directly), so we show
    joint violations as a proxy for kinematic difficulty instead.
    """
    if not kinematic_results:
        return None

    kin_viol:  Dict[str, List[int]]   = {}
    phys_rmse: Dict[str, List[float]] = {}
    for r in kinematic_results:
        kin_viol.setdefault(r.motion_type, []).append(r.max_joint_limit_violations)
    for r in physics_results:
        phys_rmse.setdefault(r.motion_type, []).append(r.mean_tracking_rmse)

    types = sorted(set(kin_viol) | set(phys_rmse))
    x = np.arange(len(types))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = [_color_by_type(t) for t in types]
    ax1.bar(x, [np.mean(kin_viol.get(t, [0])) for t in types],
            color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(types)
    ax1.set_ylabel("Max Joint Limit Violations (kinematic)")
    ax1.set_title("Kinematic Difficulty\n(joint violations in raw sequence)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.bar(x, [np.mean(phys_rmse.get(t, [0])) for t in types],
            color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(types)
    ax2.set_ylabel("Mean Joint Tracking RMSE (rad)")
    ax2.set_title("Dynamic Tracking Gap\n(physics execution with PD control)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Kinematic-to-Dynamic Gap: Difficulty vs Tracking Error", fontweight="bold")
    fig.tight_layout()

    if save:
        path = RESULTS_DIR / "gap_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_summary_table(results: List[ClipMetrics], save: bool = True):
    """Print and save a summary metrics CSV."""
    rows = [r.summary() for r in results]
    if not rows:
        return

    keys   = list(rows[0].keys())
    header = " | ".join(f"{k:>22}" for k in keys)
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(f"{str(row[k]):>22}" for k in keys))
    print(sep)

    if save:
        import csv
        path = RESULTS_DIR / "summary.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {path}")


def plot_inverse_torque_by_type(results: List[InverseDynamicsClipMetrics], save: bool = True):
    """Bar chart: required torque as a fraction of actuator limits."""
    if not results:
        return None
    by_type: Dict[str, List[float]] = {}
    for r in results:
        by_type.setdefault(r.motion_type, []).append(r.p95_torque_limit_ratio)

    types = sorted(by_type)
    means = [np.mean(by_type[t]) for t in types]
    stds = [np.std(by_type[t]) for t in types]
    colors = [_color_by_type(t) for t in types]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(types, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.8)
    ax.axhline(1.0, color="#B71C1C", linestyle="--", linewidth=1.2, label="actuator limit")
    ax.set_ylabel("95th Percentile |Required Torque| / Limit")
    ax.set_xlabel("Motion Type")
    ax.set_title("Inverse Dynamics: Torque Demand by Motion Type")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(fontsize=8)

    if save:
        path = RESULTS_DIR / "inverse_torque_demand_by_type.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_inverse_joint_heatmap(results: List[InverseDynamicsClipMetrics], save: bool = True):
    """Heatmap of per-joint torque-limit ratios from inverse dynamics."""
    if not results:
        return None

    data_matrix = np.stack([r.p95_per_joint_ratio for r in results], axis=0)
    clip_labels = [f"{r.clip_name}\n[{r.motion_type}]" for r in results]

    fig, ax = plt.subplots(figsize=(max(10, len(JOINT_NAMES) * 0.45),
                                    max(4, len(results) * 0.4 + 1.5)))
    im = ax.imshow(data_matrix, aspect="auto", cmap="magma", vmin=0)
    plt.colorbar(im, ax=ax, label="95th percentile |tau| / actuator limit")
    ax.set_xticks(range(len(JOINT_NAMES)))
    ax.set_xticklabels(JOINT_NAMES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(clip_labels, fontsize=7)
    ax.set_title("Inverse Dynamics: Per-Joint Torque-Limit Demand")
    fig.tight_layout()

    if save:
        path = RESULTS_DIR / "inverse_per_joint_torque_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_root_wrench_by_type(results: List[InverseDynamicsClipMetrics], save: bool = True):
    """Bar chart: unactuated root force/torque required by exact kinematic replay."""
    if not results:
        return None
    by_type_force: Dict[str, List[float]] = {}
    by_type_torque: Dict[str, List[float]] = {}
    for r in results:
        by_type_force.setdefault(r.motion_type, []).append(r.p95_root_force_N)
        by_type_torque.setdefault(r.motion_type, []).append(r.p95_root_torque_Nm)

    types = sorted(set(by_type_force) | set(by_type_torque))
    x = np.arange(len(types))
    colors = [_color_by_type(t) for t in types]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(x, [np.mean(by_type_force.get(t, [0])) for t in types],
            color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(types, rotation=15, ha="right")
    ax1.set_ylabel("95th Percentile Root Force (N)")
    ax1.set_title("Unactuated Root Force Demand")
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    ax2.bar(x, [np.mean(by_type_torque.get(t, [0])) for t in types],
            color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(types, rotation=15, ha="right")
    ax2.set_ylabel("95th Percentile Root Torque (Nm)")
    ax2.set_title("Unactuated Root Torque Demand")
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    fig.suptitle("Inverse Dynamics: External Root Wrench Required for Exact Replay",
                 fontweight="bold")
    fig.tight_layout()

    if save:
        path = RESULTS_DIR / "inverse_root_wrench_by_type.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_pd_vs_inverse_demand(
    physics_results: List[ClipMetrics],
    inverse_results: List[InverseDynamicsClipMetrics],
    fps: float = 30.0,
    save: bool = True,
):
    """Scatter plot comparing PD survival time with inverse-dynamics torque demand."""
    if not physics_results or not inverse_results:
        return None
    inv_by_clip = {r.clip_name: r for r in inverse_results}
    rows = [(p, inv_by_clip[p.clip_name]) for p in physics_results if p.clip_name in inv_by_clip]
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for p, inv in rows:
        ttf = (p.time_to_fall / fps) if p.fell else (p.n_frames / fps)
        ax.scatter(inv.p95_torque_limit_ratio, ttf, color=_color_by_type(p.motion_type),
                   edgecolor="black", linewidth=0.4, s=42, alpha=0.85)
    ax.axvline(1.0, color="#B71C1C", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Inverse Dynamics 95th Percentile |Torque| / Limit")
    ax.set_ylabel("PD Time-to-Fall or Clip Duration (s)")
    ax.set_title("Forward PD Outcome vs Exact-Replay Torque Demand")
    ax.grid(linestyle="--", alpha=0.4)

    if save:
        path = RESULTS_DIR / "pd_vs_inverse_demand.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)
    return fig


def plot_inverse_summary_table(results: List[InverseDynamicsClipMetrics], save: bool = True):
    rows = [r.summary() for r in results]
    if not rows:
        return

    keys = list(rows[0].keys())
    header = " | ".join(f"{k:>28}" for k in keys)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(f"{str(row[k]):>28}" for k in keys))
    print(sep)

    if save:
        import csv
        path = RESULTS_DIR / "inverse_dynamics_summary.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {path}")


def plot_all(results: List[ClipMetrics]):
    phys = [r for r in results if r.mode == "physics"]
    kin  = [r for r in results if r.mode == "kinematic"]

    plot_tracking_error_by_type(results)
    plot_fall_rate_by_type(results)
    plot_time_to_fall(results)
    plot_root_error_over_time(results)
    plot_per_joint_error_heatmap(results)
    plot_gap_comparison(phys, kin)
    plot_summary_table(results)


def plot_inverse_all(
    inverse_results: List[InverseDynamicsClipMetrics],
    physics_results: List[ClipMetrics] | None = None,
):
    plot_inverse_torque_by_type(inverse_results)
    plot_inverse_joint_heatmap(inverse_results)
    plot_root_wrench_by_type(inverse_results)
    if physics_results:
        plot_pd_vs_inverse_demand(physics_results, inverse_results)
    plot_inverse_summary_table(inverse_results)
