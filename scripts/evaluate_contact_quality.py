"""Contact-quality and self-collision metrics for kinematic qpos clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physics_eval.simulator import SCENE_XML
from physics_eval.validation import validate_qpos_sequence


DATA_DIR = ROOT / "data" / "guided_ablation_extended"
RISK_CSV = ROOT / "results" / "guided_ablation_extended.csv"
OUT_CSV = ROOT / "results" / "contact_quality.csv"
SUMMARY_CSV = ROOT / "results" / "contact_quality_summary.csv"
PLOT_CONTACT = ROOT / "results" / "contact_quality_by_category.png"
PLOT_SCATTER = ROOT / "results" / "risk_vs_contact_artifacts.png"
FPS = 30.0


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""


def geom_body_name(model: mujoco.MjModel, geom_id: int) -> str:
    return body_name(model, int(model.geom_bodyid[geom_id]))


def is_ancestor(model: mujoco.MjModel, a: int, b: int) -> bool:
    parent = int(model.body_parentid[b])
    while parent >= 0:
        if parent == a:
            return True
        if parent == int(model.body_parentid[parent]):
            break
        parent = int(model.body_parentid[parent])
    return False


def is_adjacent_body_pair(model: mujoco.MjModel, a: int, b: int) -> bool:
    return is_ancestor(model, a, b) or is_ancestor(model, b, a)


def foot_body_ids(model: mujoco.MjModel) -> set[int]:
    out = set()
    for i in range(model.nbody):
        name = body_name(model, i)
        if name in {"left_ankle_roll_link", "right_ankle_roll_link"}:
            out.add(i)
    return out


def side_body_ids(model: mujoco.MjModel, side: str) -> set[int]:
    return {
        i for i in range(model.nbody)
        if body_name(model, i).startswith(f"{side}_")
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def body_xy(data: mujoco.MjData, ids: set[int]) -> np.ndarray:
    if not ids:
        return np.zeros(2, dtype=np.float64)
    pts = np.stack([data.xpos[i, :2] for i in sorted(ids)], axis=0)
    return np.mean(pts, axis=0)


def evaluate_clip(model: mujoco.MjModel, qpos: np.ndarray, clip: str) -> dict[str, object]:
    qpos = validate_qpos_sequence(qpos, name=clip, min_frames=2)
    data = mujoco.MjData(model)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = foot_body_ids(model)
    left_ids = side_body_ids(model, "left")
    right_ids = side_body_ids(model, "right")

    self_contact_frames = 0
    nonfoot_floor_frames = 0
    foot_contact_frames = 0
    double_support_frames = 0
    support_outside_frames = 0
    max_self_penetration = 0.0
    max_floor_penetration = 0.0
    mean_contact_count = []
    mean_self_count = []
    left_contact = []
    right_contact = []
    left_xy = []
    right_xy = []
    com_xy = []

    for frame in qpos:
        data.qpos[:] = frame
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        frame_self = 0
        frame_floor_nonfoot = 0
        frame_floor_foot = 0
        frame_left_foot = False
        frame_right_foot = False
        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            dist = float(contact.dist)

            if g1 == floor_gid or g2 == floor_gid:
                other_body = b2 if g1 == floor_gid else b1
                if other_body in foot_ids:
                    frame_floor_foot += 1
                    if "left_" in body_name(model, other_body):
                        frame_left_foot = True
                    if "right_" in body_name(model, other_body):
                        frame_right_foot = True
                else:
                    frame_floor_nonfoot += 1
                max_floor_penetration = max(max_floor_penetration, max(0.0, -dist))
            elif b1 != b2 and not is_adjacent_body_pair(model, b1, b2):
                frame_self += 1
                max_self_penetration = max(max_self_penetration, max(0.0, -dist))

        if frame_self > 0:
            self_contact_frames += 1
        if frame_floor_nonfoot > 0:
            nonfoot_floor_frames += 1
        if frame_floor_foot > 0:
            foot_contact_frames += 1
        if frame_left_foot and frame_right_foot:
            double_support_frames += 1

        left_pos = body_xy(data, left_ids & foot_ids)
        right_pos = body_xy(data, right_ids & foot_ids)
        center = np.asarray(data.subtree_com[0, :2], dtype=np.float64)
        left_xy.append(left_pos)
        right_xy.append(right_pos)
        com_xy.append(center)
        left_contact.append(frame_left_foot)
        right_contact.append(frame_right_foot)

        if frame_left_foot and frame_right_foot:
            lo = np.minimum(left_pos, right_pos) - 0.08
            hi = np.maximum(left_pos, right_pos) + 0.08
            if np.any(center < lo) or np.any(center > hi):
                support_outside_frames += 1

        mean_contact_count.append(float(data.ncon))
        mean_self_count.append(float(frame_self))

    left_xy_arr = np.asarray(left_xy)
    right_xy_arr = np.asarray(right_xy)
    left_speed = np.linalg.norm(np.diff(left_xy_arr, axis=0), axis=1) * FPS
    right_speed = np.linalg.norm(np.diff(right_xy_arr, axis=0), axis=1) * FPS
    # Foot skate should only measure motion while a foot remains in contact,
    # not the swing-to-stance transition as contact appears or disappears.
    left_mask = np.asarray(left_contact[:-1], dtype=bool) & np.asarray(left_contact[1:], dtype=bool)
    right_mask = np.asarray(right_contact[:-1], dtype=bool) & np.asarray(right_contact[1:], dtype=bool)
    contact_speeds = []
    if np.any(left_mask):
        contact_speeds.extend(left_speed[left_mask].tolist())
    if np.any(right_mask):
        contact_speeds.extend(right_speed[right_mask].tolist())

    n = len(qpos)
    return {
        "frames": n,
        "self_contact_frames_pct": 100.0 * self_contact_frames / n,
        "mean_self_contacts_per_frame": float(np.mean(mean_self_count)),
        "max_self_penetration_m": max_self_penetration,
        "nonfoot_floor_contact_frames_pct": 100.0 * nonfoot_floor_frames / n,
        "foot_contact_frames_pct": 100.0 * foot_contact_frames / n,
        "double_support_frames_pct": 100.0 * double_support_frames / n,
        "support_proxy_outside_pct": (
            100.0 * support_outside_frames / max(double_support_frames, 1)
        ),
        "max_floor_penetration_m": max_floor_penetration,
        "mean_contact_count": float(np.mean(mean_contact_count)),
        "mean_contact_foot_speed_mps": float(np.mean(contact_speeds)) if contact_speeds else 0.0,
        "p95_contact_foot_speed_mps": float(np.percentile(contact_speeds, 95)) if contact_speeds else 0.0,
    }


def artifact_score(row: dict[str, object]) -> float:
    return float(
        0.35 * min(float(row["self_contact_frames_pct"]) / 20.0, 1.0)
        + 0.25 * min(float(row["nonfoot_floor_contact_frames_pct"]) / 20.0, 1.0)
        + 0.20 * min(float(row["p95_contact_foot_speed_mps"]) / 0.50, 1.0)
        + 0.20 * min(float(row["support_proxy_outside_pct"]) / 50.0, 1.0)
    )


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for k in sorted({int(r["K"]) for r in rows}):
        group = [r for r in rows if int(r["K"]) == k]
        out.append({
            "group": f"K={k}",
            "n": len(group),
            "mean_artifact_score": float(np.mean([r["contact_artifact_score"] for r in group])),
            "mean_self_contact_frames_pct": float(np.mean([r["self_contact_frames_pct"] for r in group])),
            "mean_nonfoot_floor_contact_pct": float(np.mean([r["nonfoot_floor_contact_frames_pct"] for r in group])),
            "mean_p95_contact_foot_speed_mps": float(np.mean([r["p95_contact_foot_speed_mps"] for r in group])),
        })
    for category in sorted({r["category"] for r in rows}):
        for k in sorted({int(r["K"]) for r in rows}):
            group = [r for r in rows if r["category"] == category and int(r["K"]) == k]
            if not group:
                continue
            out.append({
                "group": f"{category}_K={k}",
                "n": len(group),
                "mean_artifact_score": float(np.mean([r["contact_artifact_score"] for r in group])),
                "mean_self_contact_frames_pct": float(np.mean([r["self_contact_frames_pct"] for r in group])),
                "mean_nonfoot_floor_contact_pct": float(np.mean([r["nonfoot_floor_contact_frames_pct"] for r in group])),
                "mean_p95_contact_foot_speed_mps": float(np.mean([r["p95_contact_foot_speed_mps"] for r in group])),
            })
    return out


def plot_by_category(rows: list[dict[str, object]], out_path: Path) -> None:
    cats = sorted({r["category"] for r in rows})
    x = np.arange(len(cats))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for offset, k, label, color in [
        (-width / 2, 1, "MotionBricks K=1", "#555555"),
        (width / 2, 8, "Screened K=8", "#d62728"),
    ]:
        vals = [
            np.mean([r["contact_artifact_score"] for r in rows if r["category"] == c and int(r["K"]) == k])
            for c in cats
        ]
        ax.bar(x + offset, vals, width, label=label, color=color)
    ax.set_ylabel("Contact artifact score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_title("Contact and support artifacts across the 105-task suite")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_risk_scatter(rows: list[dict[str, object]], out_path: Path) -> None:
    xs = [float(r["full_risk"]) for r in rows if "full_risk" in r]
    ys = [float(r["contact_artifact_score"]) for r in rows if "full_risk" in r]
    colors = ["#555555" if int(r["K"]) == 1 else "#d62728" for r in rows if "full_risk" in r]
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(xs, ys, c=colors, s=28, alpha=0.75, edgecolor="none")
    ax.set_xlabel("Inverse-dynamics heuristic risk")
    ax.set_ylabel("Contact artifact score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Inverse dynamics and contact artifacts capture different risks")
    ax.text(0.98, 0.05, "gray=K1, red=K8", transform=ax.transAxes, ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--risk_csv", type=Path, default=RISK_CSV)
    parser.add_argument("--out_csv", type=Path, default=OUT_CSV)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    risk_rows = load_rows(args.risk_csv)
    rows = []
    for i, r in enumerate(risk_rows):
        if args.limit is not None and i >= args.limit:
            break
        path = Path(r["path"])
        if not path.exists():
            path = args.data_dir / f"{r['clip']}_K{r['K']}.npy"
        if not path.exists():
            continue
        qpos = np.load(path)
        metrics = evaluate_clip(model, qpos, r["clip"])
        row = {
            "clip": r["clip"],
            "mode": r["mode"],
            "category": r["type"],
            "seed_idx": int(r["seed_idx"]),
            "K": int(r["K"]),
            "full_risk": float(r["full_risk"]),
            **metrics,
            "path": str(path),
        }
        row["contact_artifact_score"] = artifact_score(row)
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"Evaluated {len(rows)} clips...")

    write_csv(args.out_csv, rows)
    write_csv(SUMMARY_CSV, summarize(rows))
    plot_by_category(rows, PLOT_CONTACT)
    plot_risk_scatter(rows, PLOT_SCATTER)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(f"Wrote summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
