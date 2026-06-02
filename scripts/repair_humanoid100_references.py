"""Second-stage repair for generated humanoid100 MotionBricks references.

The first-stage experiment selects a low-risk K=8 candidate. This script tests
a stronger, controller-friendly repair family: time scaling plus light joint
smoothing. Retiming is not free semantically because it changes duration, so the
CSV reports the selected repair variant explicitly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_humanoid_100_prompts import render_proxy_video, write_rows  # noqa: E402
from physics_eval.physaware import PhysicalAwarenessCritic  # noqa: E402


IN_CSV = ROOT / "results" / "humanoid100_motionbricks_experiment" / "humanoid100_motionbricks_results.csv"
OUT_DIR = ROOT / "results" / "humanoid100_repaired_retimed"
DATA_DIR = ROOT / "data" / "humanoid100_repaired_retimed"
IMPROVEMENT_EPS = 1e-5


def validate_qpos(qpos: np.ndarray, source: str) -> np.ndarray:
    arr = np.asarray(qpos, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{source}: expected qpos with shape (T, 36), got ndim={arr.ndim}")
    if arr.shape[1] != 36:
        raise ValueError(f"{source}: expected qpos with shape (T, 36), got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"{source}: dynamic repair needs at least two frames, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{source}: qpos contains non-finite values")

    quat_norm = np.linalg.norm(arr[:, 3:7], axis=1)
    if np.any(quat_norm < 1e-6):
        raise ValueError(f"{source}: root quaternion contains near-zero norm")
    arr = arr.copy()
    arr[:, 3:7] /= quat_norm[:, None]
    return arr


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def qpos_slerp(qpos: np.ndarray, factor: float) -> np.ndarray:
    qpos = validate_qpos(qpos, "qpos_slerp")
    if factor <= 0:
        raise ValueError("factor must be positive")
    if abs(factor - 1.0) < 1e-6:
        return qpos.astype(np.float32).copy()

    t_old = np.arange(len(qpos), dtype=np.float64)
    t_new = np.linspace(0, len(qpos) - 1, max(2, int(round(len(qpos) * factor))))
    out = np.empty((len(t_new), qpos.shape[1]), dtype=np.float64)

    for j in range(3):
        out[:, j] = np.interp(t_new, t_old, qpos[:, j])

    quat_xyzw = qpos[:, [4, 5, 6, 3]]
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    slerp = Slerp(t_old, Rotation.from_quat(quat_xyzw))
    interp_quat = slerp(t_new).as_quat()
    out[:, 3:7] = interp_quat[:, [3, 0, 1, 2]]

    for j in range(7, qpos.shape[1]):
        out[:, j] = np.interp(t_new, t_old, qpos[:, j])
    return out.astype(np.float32)


def smooth_joints(qpos: np.ndarray, method: str) -> np.ndarray:
    qpos = validate_qpos(qpos, f"smooth_joints/{method}")
    out = qpos.astype(np.float64).copy()
    if method == "none":
        return qpos.astype(np.float32).copy()
    if method == "gaussian":
        out[:, :3] = gaussian_filter1d(out[:, :3], sigma=1.0, axis=0, mode="nearest")
        out[:, 7:] = gaussian_filter1d(out[:, 7:], sigma=1.0, axis=0, mode="nearest")
    elif method == "savgol":
        window = min(15, len(out) if len(out) % 2 == 1 else len(out) - 1)
        if window >= 5:
            out[:, :3] = savgol_filter(out[:, :3], window_length=window, polyorder=3, axis=0, mode="interp")
            out[:, 7:] = savgol_filter(out[:, 7:], window_length=window, polyorder=3, axis=0, mode="interp")
    else:
        raise ValueError(f"unknown smoothing method: {method}")

    quat = out[:, 3:7]
    out[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return out.astype(np.float32)


def path_length(qpos: np.ndarray) -> float:
    qpos = validate_qpos(qpos, "path_length")
    if len(qpos) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1).sum())


def root_displacement(qpos: np.ndarray) -> float:
    qpos = validate_qpos(qpos, "root_displacement")
    return float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2])) if len(qpos) else 0.0


def repair_variants(qpos: np.ndarray) -> dict[str, np.ndarray]:
    qpos = validate_qpos(qpos, "repair_variants")
    variants = {"identity": qpos.astype(np.float32).copy()}
    for factor in (1.25, 1.5, 2.0):
        retimed = qpos_slerp(qpos, factor)
        tag = str(factor).replace(".", "p")
        variants[f"retime_{tag}x"] = retimed
        variants[f"retime_{tag}x_gaussian"] = smooth_joints(retimed, "gaussian")
        variants[f"retime_{tag}x_savgol"] = smooth_joints(retimed, "savgol")
    return variants


def score_variant(critic: PhysicalAwarenessCritic, qpos: np.ndarray, row: dict[str, str], variant: str):
    report, _ = critic.score(qpos, row["prompt_id"], row["category"], variant=variant)
    return report


def write_report(rows: list[dict[str, object]], out_dir: Path) -> None:
    before = np.array([float(row["before_risk"]) for row in rows])
    selected = np.array([float(row["repaired_risk"]) for row in rows])
    k8 = np.array([float(row["after_risk"]) for row in rows])
    improved_vs_k8 = int(np.sum(selected < k8))
    worsened_vs_k8 = int(np.sum(selected > k8 + IMPROVEMENT_EPS))
    lines = [
        "# Humanoid100 Retiming Repair",
        "",
        "This second-stage baseline repairs each selected K=8 reference using",
        "time-scaling plus light joint smoothing, then picks the lowest inverse-",
        "dynamics risk variant. It is a physical feasibility repair, not a semantic",
        "prompt-following fix.",
        "",
        "## Summary",
        "",
        f"- Prompts repaired: {len(rows)}",
        f"- Mean K=1 risk: {float(before.mean()):.3f}",
        f"- Mean K=8 risk: {float(k8.mean()):.3f}",
        f"- Mean repaired risk: {float(selected.mean()):.3f}",
        f"- Repaired better than K=8: {improved_vs_k8}/{len(rows)}",
        f"- Repaired worse than K=8: {worsened_vs_k8}/{len(rows)}",
        f"- Aggregate reduction vs K=1: {100.0 * (float(before.mean()) - float(selected.mean())) / max(float(before.mean()), 1e-8):.2f}%",
        f"- Aggregate reduction vs K=8: {100.0 * (float(k8.mean()) - float(selected.mean())) / max(float(k8.mean()), 1e-8):.2f}%",
        "",
        "## Caveat",
        "",
        "Retiming changes motion duration. It can make references easier to execute",
        "under dynamics, but it may not preserve the intended tempo or task timing.",
        "Forced proxy rows remain semantically invalid for their natural-language",
        "prompt even after repair.",
        "",
        "## Files",
        "",
        f"- CSV: `{out_dir / 'repair_summary.csv'}`",
        f"- Videos: `{out_dir / 'videos'}`",
        f"- Qpos: `{DATA_DIR}`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, default=IN_CSV)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "videos").mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    critic = PhysicalAwarenessCritic()
    rows: list[dict[str, object]] = []

    for source in read_rows(args.input_csv)[: args.limit]:
        qpos = validate_qpos(np.load(source["after_qpos_path"]), source["after_qpos_path"])
        variants = repair_variants(qpos)
        scored = []
        for name, candidate in variants.items():
            report = score_variant(critic, candidate, source, name)
            scored.append((float(report.risk_score), name, candidate, report))
        risk, name, selected_qpos, report = min(scored, key=lambda item: item[0])
        after_risk = float(source["after_risk"])
        if risk >= after_risk - IMPROVEMENT_EPS:
            name = "identity"
            selected_qpos = qpos
            risk = after_risk
            report = score_variant(critic, selected_qpos, source, name)
        out_qpos = args.data_dir / f"{source['prompt_id']}_{source['subcategory']}_{name}.npy"
        np.save(out_qpos, selected_qpos.astype(np.float32))

        before_risk = float(source["before_risk"])
        row: dict[str, object] = dict(source)
        row.update({
            "repaired_variant": name,
            "repaired_risk": risk,
            "repaired_action": report.recommended_action,
            "repaired_frames": len(selected_qpos),
            "duration_scale": len(selected_qpos) / max(len(qpos), 1),
            "path_length_before": path_length(qpos),
            "path_length_repaired": path_length(selected_qpos),
            "root_displacement_before": root_displacement(qpos),
            "root_displacement_repaired": root_displacement(selected_qpos),
            "risk_reduction_vs_k1_pct": (
                100.0 * (before_risk - risk) / before_risk if before_risk > 1e-6 else float("nan")
            ),
            "risk_reduction_vs_k8_pct": (
                100.0 * (after_risk - risk) / after_risk if after_risk > 1e-6 else float("nan")
            ),
            "repaired_qpos_path": str(out_qpos),
            "video_path": str(args.out_dir / "videos" / f"{source['prompt_id']}_{source['subcategory']}_repaired.mp4"),
        })
        rows.append(row)
        write_rows(args.out_dir / "repair_summary.csv", rows)
        write_report(rows, args.out_dir)
        print(
            f"{source['prompt_id']} {source['subcategory']}: "
            f"K8={after_risk:.2f} repaired={risk:.2f} variant={name}"
        )

        if args.render:
            render_row = dict(row)
            render_row["after_qpos_path"] = str(out_qpos)
            render_row["after_risk"] = risk
            render_proxy_video(render_row, Path(str(row["video_path"])))

    write_rows(args.out_dir / "repair_summary.csv", rows)
    write_report(rows, args.out_dir)
    print(f"Wrote {len(rows)} rows to {args.out_dir / 'repair_summary.csv'}")


if __name__ == "__main__":
    main()
