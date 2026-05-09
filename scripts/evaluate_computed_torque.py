"""Computed-torque tracking validation for K=1 vs K=8 guided clips.

This is not a learned retargeting policy. It is a stronger model-based tracking
baseline than the original weak PD controller: for each reference trajectory we
compute inverse-dynamics joint torques at the reference state, then track the
reference in forward MuJoCo with torque-limited inverse-dynamics feedforward
plus PD feedback.

The purpose is to test whether physics-critic-selected references are easier to
track under the same actuator limits, without overclaiming real robot execution.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from physics_eval.inverse_dynamics import InverseDynamicsEvaluator
from physics_eval.metrics import FALL_HEIGHT_THRESHOLD
from physics_eval.pd_controller import ACTUATOR_FORCE_LIMITS, KD, KP
from physics_eval.simulator import SCENE_XML
from physics_eval.validation import validate_qpos_sequence

DATA_DIR = ROOT / "data" / "guided_ablation"
GUIDED_CSV = ROOT / "results" / "guided_ablation_full.csv"
OUT_CSV = ROOT / "results" / "computed_torque_tracking.csv"
OUT_SUMMARY = ROOT / "results" / "computed_torque_tracking_summary.csv"
OUT_PLOT = ROOT / "results" / "computed_torque_tracking.png"

CTRL_DT = 1.0 / 30.0
MAX_VEL = 50.0


def load_guided_rows() -> list[dict]:
    with open(GUIDED_CSV) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if int(r["K"]) in (1, 8)]


def reference_dynamics(qpos: np.ndarray, evaluator: InverseDynamicsEvaluator):
    qvel = evaluator.compute_qvel(qpos)
    qacc = evaluator.compute_qacc(qvel)
    tau = np.zeros((len(qpos), 29), dtype=np.float64)
    for t in range(len(qpos)):
        evaluator.data.qpos[:] = qpos[t]
        evaluator.data.qvel[:] = qvel[t]
        evaluator.data.qacc[:] = qacc[t]
        mujoco.mj_inverse(evaluator.model, evaluator.data)
        tau[t] = evaluator.data.qfrc_inverse[6:]
    return qvel, tau


def evaluate_reference(
    qpos: np.ndarray,
    clip: str,
    motion_type: str,
    risk: float,
    kp_scale: float = 1.0,
    kd_scale: float = 1.0,
) -> dict:
    qpos = validate_qpos_sequence(qpos, name=clip, min_frames=2)
    evaluator = InverseDynamicsEvaluator()
    qvel_ref, tau_ff = reference_dynamics(qpos, evaluator)

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    substeps_for_frame = [
        max(1, round((t + 1) * CTRL_DT / model.opt.timestep) - round(t * CTRL_DT / model.opt.timestep))
        for t in range(len(qpos))
    ]

    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)

    tracking_errors = []
    root_errors = []
    torque_saturation = []
    time_to_fall = None
    fell = False

    kp = KP * kp_scale
    kd = KD * kd_scale
    limits = ACTUATOR_FORCE_LIMITS

    for t in range(len(qpos)):
        q_target = qpos[t, 7:]
        dq_target = qvel_ref[t, 6:]
        q = data.qpos[7:].copy()
        dq = data.qvel[6:].copy()

        tau = tau_ff[t] + kp * (q_target - q) + kd * (dq_target - dq)
        tau = np.clip(tau, -limits, limits)
        data.ctrl[:] = tau

        for _ in range(substeps_for_frame[t]):
            mujoco.mj_step(model, data)
        np.clip(data.qvel, -MAX_VEL, MAX_VEL, out=data.qvel)

        if not np.isfinite(data.qpos).all():
            fell = True
            time_to_fall = t
            break

        tracking_errors.append(float(np.sqrt(np.mean((data.qpos[7:] - q_target) ** 2))))
        root_errors.append(float(np.linalg.norm(data.qpos[:3] - qpos[t, :3])))
        torque_saturation.append(float(np.mean(np.abs(tau) >= limits * 0.999)))

        if data.qpos[2] < FALL_HEIGHT_THRESHOLD:
            fell = True
            time_to_fall = t
            break

    active_frames = len(tracking_errors)
    if time_to_fall is None:
        time_to_fall = len(qpos)

    return {
        "clip": clip,
        "type": motion_type,
        "K": None,
        "risk": risk,
        "fell": fell,
        "time_to_fall": time_to_fall,
        "track_seconds": time_to_fall / 30.0,
        "tracking_rmse": float(np.mean(tracking_errors)) if tracking_errors else float("nan"),
        "root_rmse": float(np.mean(root_errors)) if root_errors else float("nan"),
        "mean_torque_saturation_frac": float(np.mean(torque_saturation)) if torque_saturation else float("nan"),
        "active_frames": active_frames,
    }


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for k in (1, 8):
        group = [r for r in rows if int(r["K"]) == k]
        out.append({
            "K": k,
            "n": len(group),
            "mean_risk": float(np.mean([r["risk"] for r in group])),
            "mean_time_to_fall": float(np.mean([r["time_to_fall"] for r in group])),
            "mean_tracking_rmse": float(np.nanmean([r["tracking_rmse"] for r in group])),
            "mean_root_rmse": float(np.nanmean([r["root_rmse"] for r in group])),
            "mean_torque_saturation_frac": float(np.nanmean([r["mean_torque_saturation_frac"] for r in group])),
            "fell_count": sum(bool(r["fell"]) for r in group),
        })

    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], {})[int(r["K"])] = r
    paired = [v for v in by_clip.values() if 1 in v and 8 in v]
    if paired:
        risk_delta = np.array([p[8]["risk"] - p[1]["risk"] for p in paired])
        track_delta = np.array([p[8]["tracking_rmse"] - p[1]["tracking_rmse"] for p in paired])
        fall_delta = np.array([p[8]["time_to_fall"] - p[1]["time_to_fall"] for p in paired])
        rho, pval = spearmanr([p[8]["risk"] for p in paired], [p[8]["tracking_rmse"] for p in paired])
        try:
            w_track = wilcoxon(track_delta, alternative="less")
            w_fall = wilcoxon(fall_delta, alternative="greater")
        except ValueError:
            w_track = w_fall = None
        out.append({
            "K": "paired_K8_minus_K1",
            "n": len(paired),
            "mean_risk": float(np.mean(risk_delta)),
            "mean_time_to_fall": float(np.mean(fall_delta)),
            "mean_tracking_rmse": float(np.nanmean(track_delta)),
            "mean_root_rmse": float("nan"),
            "mean_torque_saturation_frac": float("nan"),
            "fell_count": "",
            "spearman_risk_vs_tracking_rho_at_K8": float(rho),
            "spearman_risk_vs_tracking_p_at_K8": float(pval),
            "wilcoxon_tracking_p_K8_less_than_K1": float(w_track.pvalue) if w_track else float("nan"),
            "wilcoxon_falltime_p_K8_greater_than_K1": float(w_fall.pvalue) if w_fall else float("nan"),
        })
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def plot(rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], {})[int(r["K"])] = r
    paired = [v for v in by_clip.values() if 1 in v and 8 in v]
    names = [p[1]["clip"] for p in paired]
    x = np.arange(len(names))
    k1 = np.array([p[1]["tracking_rmse"] for p in paired])
    k8 = np.array([p[8]["tracking_rmse"] for p in paired])

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.32), 5))
    ax.bar(x - 0.18, k1, width=0.36, label="K=1", color="#90A4AE", edgecolor="black")
    ax.bar(x + 0.18, k8, width=0.36, label="K=8", color="#42A5F5", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Computed-Torque Tracking RMSE (rad)")
    ax.set_title("K=8 References Under Torque-Limited Computed-Torque Tracking")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PLOT}")


def main():
    rows = []
    for meta in load_guided_rows():
        clip = meta["clip"]
        k = int(meta["K"])
        path = DATA_DIR / f"{clip}_K{k}.npy"
        if not path.exists():
            print(f"Missing {path}, skipping")
            continue
        qpos = np.load(path)
        out = evaluate_reference(qpos, clip, meta["type"], float(meta["full_risk"]))
        out["K"] = k
        rows.append(out)
        print(
            f"{clip:28s} K={k:<2d} risk={out['risk']:6.2f} "
            f"rmse={out['tracking_rmse']:.3f} fall={out['time_to_fall']}"
        )

    write_csv(OUT_CSV, rows)
    write_csv(OUT_SUMMARY, summarize(rows))
    plot(rows)


if __name__ == "__main__":
    main()
