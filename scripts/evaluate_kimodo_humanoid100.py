#!/usr/bin/env python3
"""Evaluate generated Kimodo-G1 Humanoid100 qpos clips.

This mirrors the MotionBricks physical screening: inverse dynamics,
contact artifacts, and optional SONIC reference export. Text-alignment remains
separate; Kimodo benchmark/TMR evaluation should be used when the full Kimodo
generation outputs are available.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_contact_quality import artifact_score, evaluate_clip  # noqa: E402
from evaluate_humanoid100_final import physical_pass  # noqa: E402
from export_sonic_references import SONIC_FPS, export_clip  # noqa: E402
from physics_eval.physaware import PhysicalAwarenessCritic  # noqa: E402
from physics_eval.simulator import SCENE_XML  # noqa: E402


DEFAULT_MANIFEST = ROOT / "results" / "kimodo_humanoid100_g1" / "manifest.csv"
DEFAULT_OUT = ROOT / "results" / "kimodo_humanoid100_eval"
DEFAULT_SONIC_SENTINEL = Path("__AUTO__")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = read_rows(args.manifest)
    successes = [r for r in rows if r.get("status") in {"success", "skipped_existing"}]
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    critic = PhysicalAwarenessCritic()
    out_rows: list[dict[str, object]] = []

    sonic_manifest: list[dict[str, object]] = []
    if args.export_sonic_refs:
        args.sonic_ref_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(successes, start=1):
        qpos_path = Path(row["qpos_npy_path"])
        qpos = np.load(qpos_path)
        report, _ = critic.score(qpos, row["prompt_id"], row["category"], variant="kimodo_g1")
        contact = evaluate_clip(model, qpos, row["prompt_id"])
        contact["contact_artifact_score"] = artifact_score(contact)
        out = {
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "prompt_text": row["prompt_text"],
            "success_criteria": row.get("success_criteria", ""),
            "method": "Kimodo_G1",
            "model": row["model"],
            "duration_s": row["duration_s"],
            "diffusion_steps": row["diffusion_steps"],
            "seed": row["seed"],
            "qpos_path": str(qpos_path),
            "risk_score": report.risk_score,
            "recommended_action": report.recommended_action,
            "p95_torque_limit_ratio": report.p95_torque_limit_ratio,
            "exceeded_joint_pct": report.exceeded_joint_pct,
            "p95_root_force_N": report.p95_root_force_N,
            "p95_root_torque_Nm": report.p95_root_torque_Nm,
            "p95_joint_vel_rad_s": report.p95_joint_vel_rad_s,
            "p95_joint_acc_rad_s2": report.p95_joint_acc_rad_s2,
            **contact,
        }
        out["physical_pass"] = "__YES__" if physical_pass(out) else "__NO__"
        out_rows.append(out)

        if args.export_sonic_refs:
            name = f"{row['prompt_id']}_{row['subcategory']}_Kimodo"
            exported = export_clip(qpos_path, args.sonic_ref_dir / name, args.source_fps, SONIC_FPS)
            exported.update({
                "prompt_id": row["prompt_id"],
                "subcategory": row["subcategory"],
                "category": row["category"],
                "method": "Kimodo_G1",
                "physical_pass": out["physical_pass"],
                "risk_score": out["risk_score"],
                "contact_artifact_score": out["contact_artifact_score"],
            })
            sonic_manifest.append(exported)
        if idx % 20 == 0:
            print(f"Evaluated {idx}/{len(successes)} Kimodo clips")

    if args.export_sonic_refs:
        write_csv(args.sonic_ref_dir / "manifest.csv", sonic_manifest)
    return out_rows, summarize(out_rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups["method=Kimodo_G1"].append(row)
        groups[f"category={row['category']}|method=Kimodo_G1"].append(row)
    out: list[dict[str, object]] = []
    for group, rs in sorted(groups.items()):
        out.append({
            "group": group,
            "n": len(rs),
            "mean_risk": float(np.mean([float(r["risk_score"]) for r in rs])),
            "median_risk": float(np.median([float(r["risk_score"]) for r in rs])),
            "mean_contact_artifact_score": float(np.mean([float(r["contact_artifact_score"]) for r in rs])),
            "mean_p95_torque_limit_ratio": float(np.mean([float(r["p95_torque_limit_ratio"]) for r in rs])),
            "physical_pass_count": sum(r["physical_pass"] == "__YES__" for r in rs),
            "accept_action_count": sum(r["recommended_action"] == "accept" for r in rs),
            "repair_action_count": sum(r["recommended_action"] == "repair_or_rerank" for r in rs),
            "reject_action_count": sum(r["recommended_action"] == "reject_or_regenerate" for r in rs),
        })
    return out


def plot_summary(rows: list[dict[str, object]], out_dir: Path) -> None:
    if not rows:
        return
    cats = sorted({str(r["category"]) for r in rows})
    risks = [np.mean([float(r["risk_score"]) for r in rows if r["category"] == cat]) for cat in cats]
    passes = [sum(r["physical_pass"] == "__YES__" for r in rows if r["category"] == cat) for cat in cats]
    counts = [sum(1 for r in rows if r["category"] == cat) for cat in cats]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    axes[0].bar(cats, risks, color="#4f7cac")
    axes[0].set_ylabel("Mean inverse-dynamics/contact risk")
    axes[0].set_title("Kimodo-G1 physical risk by category")
    axes[1].bar(cats, passes, color="#2d8f67")
    axes[1].plot(cats, counts, color="#222222", marker="o", lw=1.0, label="category n")
    axes[1].set_ylabel("Physical pass count")
    axes[1].set_title("Kimodo-G1 verifier pass count")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_dir / "kimodo_physical_summary.png", dpi=190)
    plt.close(fig)


def write_readme(out_dir: Path, rows: list[dict[str, object]], summary: list[dict[str, object]], args: argparse.Namespace) -> None:
    agg = next((r for r in summary if r["group"] == "method=Kimodo_G1"), None)
    lines = [
        "# Kimodo Humanoid100 Physical Evaluation",
        "",
        "This evaluates Kimodo-G1 qpos exports with the same physical verifier used",
        "for MotionBricks: inverse-dynamics demand, contact artifacts, and optional",
        "SONIC reference export.",
        "",
        "## Summary",
        "",
        f"- Clips evaluated: {len(rows)}",
    ]
    if agg:
        lines.extend([
            f"- Physical pass: {agg['physical_pass_count']}/{agg['n']}",
            f"- Mean risk: {float(agg['mean_risk']):.3f}",
            f"- Mean p95 torque-limit ratio: {float(agg['mean_p95_torque_limit_ratio']):.3f}",
        ])
    lines.extend([
        "",
        "## Files",
        "",
        f"- Metrics: `{out_dir / 'final_metrics.csv'}`",
        f"- Summary: `{out_dir / 'summary.csv'}`",
        f"- SONIC refs: `{args.sonic_ref_dir}`" if args.export_sonic_refs else "- SONIC refs: not exported in this run",
        "",
    ])
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--export_sonic_refs", action="store_true")
    parser.add_argument(
        "--sonic_ref_dir",
        type=Path,
        default=DEFAULT_SONIC_SENTINEL,
        help="Defaults to <out_dir>/sonic_references.",
    )
    parser.add_argument("--source_fps", type=float, default=30.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.sonic_ref_dir == DEFAULT_SONIC_SENTINEL:
        args.sonic_ref_dir = args.out_dir / "sonic_references"
    rows, summary = evaluate(args)
    write_csv(args.out_dir / "final_metrics.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary)
    plot_summary(rows, args.out_dir)
    write_readme(args.out_dir, rows, summary, args)
    print(f"Wrote {len(rows)} Kimodo eval rows to {args.out_dir}")


if __name__ == "__main__":
    main()
