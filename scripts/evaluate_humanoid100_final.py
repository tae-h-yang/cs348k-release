"""Final 100-row physical evaluation for MotionBricks proxy references.

This joins the generated K=1 baseline, K=8 best-of-K selection, and repaired
retiming/smoothing references into one table. It deliberately keeps semantic
support separate from physical feasibility: forced nearest-mode proxies can be
physically cleaner without satisfying the original natural-language prompt.
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
from physics_eval.physaware import PhysicalAwarenessCritic  # noqa: E402
from physics_eval.simulator import SCENE_XML  # noqa: E402


DEFAULT_MOTIONBRICKS_CSV = ROOT / "results" / "humanoid100_motionbricks_experiment" / "humanoid100_motionbricks_results.csv"
DEFAULT_REPAIR_CSV = ROOT / "results" / "humanoid100_repaired_retimed" / "repair_summary.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "humanoid100_final_eval"

METHODS = ("K1_first", "K8_best_of_8", "repaired_retime")


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


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def method_specs(source: dict[str, str], repair: dict[str, str]) -> list[dict[str, object]]:
    return [
        {
            "method": "K1_first",
            "qpos_path": source["before_qpos_path"],
            "risk_from_csv": f(source, "before_risk"),
            "action_from_csv": source["before_action"],
            "variant": "first_candidate",
        },
        {
            "method": "K8_best_of_8",
            "qpos_path": source["after_qpos_path"],
            "risk_from_csv": f(source, "after_risk"),
            "action_from_csv": source["after_action"],
            "variant": "best_of_8",
        },
        {
            "method": "repaired_retime",
            "qpos_path": repair["repaired_qpos_path"],
            "risk_from_csv": f(repair, "repaired_risk"),
            "action_from_csv": repair["repaired_action"],
            "variant": repair["repaired_variant"],
        },
    ]


def physical_pass(row: dict[str, object]) -> bool:
    category = str(row["category"])
    nonfoot_limit = 45.0 if category == "floor_low_posture" else 8.0
    foot_contact_min = 5.0 if category == "floor_low_posture" else 20.0
    return bool(
        float(row["risk_score"]) <= 25.0
        and float(row["contact_artifact_score"]) <= 0.45
        and float(row["max_floor_penetration_m"]) <= 0.08
        and float(row["self_contact_frames_pct"]) <= 8.0
        and float(row["nonfoot_floor_contact_frames_pct"]) <= nonfoot_limit
        and float(row["foot_contact_frames_pct"]) >= foot_contact_min
    )


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    critic = PhysicalAwarenessCritic()
    motion_rows = read_rows(args.motionbricks_csv)
    repair_rows = {row["prompt_id"]: row for row in read_rows(args.repair_csv)}
    out: list[dict[str, object]] = []

    for idx, source in enumerate(motion_rows, start=1):
        repair = repair_rows[source["prompt_id"]]
        for spec in method_specs(source, repair):
            qpos_path = Path(str(spec["qpos_path"]))
            qpos = np.load(qpos_path)
            report, inv = critic.score(
                qpos,
                source["prompt_id"],
                source["category"],
                variant=str(spec["variant"]),
            )
            contact = evaluate_clip(model, qpos, source["prompt_id"])
            contact["contact_artifact_score"] = artifact_score(contact)
            row: dict[str, object] = {
                "prompt_id": source["prompt_id"],
                "category": source["category"],
                "subcategory": source["subcategory"],
                "prompt_text": source["prompt_text"],
                "semantic_validity": source["semantic_validity"],
                "current_motionbricks_support": source["current_motionbricks_support"],
                "proxy_mode": source["proxy_mode"],
                "method": spec["method"],
                "variant": spec["variant"],
                "qpos_path": str(qpos_path),
                "risk_score": report.risk_score,
                "risk_from_csv": spec["risk_from_csv"],
                "recommended_action": report.recommended_action,
                "action_from_csv": spec["action_from_csv"],
                "p95_torque_limit_ratio": report.p95_torque_limit_ratio,
                "exceeded_joint_pct": report.exceeded_joint_pct,
                "p95_root_force_N": report.p95_root_force_N,
                "p95_root_torque_Nm": report.p95_root_torque_Nm,
                "p95_joint_vel_rad_s": report.p95_joint_vel_rad_s,
                "p95_joint_acc_rad_s2": report.p95_joint_acc_rad_s2,
                **contact,
            }
            row["physical_pass"] = "__YES__" if physical_pass(row) else "__NO__"
            row["semantic_supported"] = "__YES__" if source["semantic_validity"] == "supported_proxy" else "__NO__"
            row["presentation_pass"] = (
                "__YES__" if row["physical_pass"] == "__YES__" and row["semantic_supported"] == "__YES__" else "__NO__"
            )
            out.append(row)
        if idx % 20 == 0:
            print(f"Evaluated {idx}/100 prompts across {len(METHODS)} methods...")

    return out, summarize(out)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[f"method={row['method']}"].append(row)
        groups[f"category={row['category']}|method={row['method']}"].append(row)
        groups[f"semantic={row['semantic_validity']}|method={row['method']}"].append(row)
    summary = []
    for group, rs in sorted(groups.items()):
        summary.append({
            "group": group,
            "n": len(rs),
            "mean_risk": float(np.mean([float(r["risk_score"]) for r in rs])),
            "median_risk": float(np.median([float(r["risk_score"]) for r in rs])),
            "mean_contact_artifact_score": float(np.mean([float(r["contact_artifact_score"]) for r in rs])),
            "mean_p95_torque_limit_ratio": float(np.mean([float(r["p95_torque_limit_ratio"]) for r in rs])),
            "mean_nonfoot_floor_contact_pct": float(np.mean([float(r["nonfoot_floor_contact_frames_pct"]) for r in rs])),
            "physical_pass_count": sum(r["physical_pass"] == "__YES__" for r in rs),
            "semantic_supported_count": sum(r["semantic_supported"] == "__YES__" for r in rs),
            "presentation_pass_count": sum(r["presentation_pass"] == "__YES__" for r in rs),
            "accept_action_count": sum(r["recommended_action"] == "accept" for r in rs),
            "repair_action_count": sum(r["recommended_action"] == "repair_or_rerank" for r in rs),
            "reject_action_count": sum(r["recommended_action"] == "reject_or_regenerate" for r in rs),
        })
    return summary


def plot_method_bars(rows: list[dict[str, object]], out_dir: Path) -> None:
    labels = list(METHODS)
    mean_risk = [np.mean([float(r["risk_score"]) for r in rows if r["method"] == m]) for m in labels]
    mean_contact = [np.mean([float(r["contact_artifact_score"]) for r in rows if r["method"] == m]) for m in labels]
    passes = [sum(r["physical_pass"] == "__YES__" for r in rows if r["method"] == m) for m in labels]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    colors = ["#707070", "#3267a8", "#2d8f67"]
    axes[0].bar(labels, mean_risk, color=colors)
    axes[0].set_ylabel("Mean inverse-dynamics risk")
    axes[0].set_title("Dynamics Demand")
    axes[1].bar(labels, mean_contact, color=colors)
    axes[1].set_ylabel("Mean contact artifact score")
    axes[1].set_title("Contact Artifacts")
    axes[2].bar(labels, passes, color=colors)
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("Physical-pass rows / 100")
    axes[2].set_title("Verifier Pass Count")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "method_summary_bars.png", dpi=190)
    plt.close(fig)


def plot_category_risk(rows: list[dict[str, object]], out_dir: Path) -> None:
    cats = sorted({str(r["category"]) for r in rows})
    x = np.arange(len(cats))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    colors = {"K1_first": "#707070", "K8_best_of_8": "#3267a8", "repaired_retime": "#2d8f67"}
    for i, method in enumerate(METHODS):
        vals = [
            np.mean([float(r["risk_score"]) for r in rows if r["category"] == cat and r["method"] == method])
            for cat in cats
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=method, color=colors[method])
    ax.set_ylabel("Mean inverse-dynamics risk")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=24, ha="right")
    ax.set_title("100-row MotionBricks proxy benchmark by behavior category")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_by_category.png", dpi=190)
    plt.close(fig)


def plot_paired_improvement(rows: list[dict[str, object]], out_dir: Path) -> None:
    by_prompt: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_prompt[str(row["prompt_id"])][str(row["method"])] = row
    xs, ys, colors = [], [], []
    for prompt, methods in by_prompt.items():
        if "K1_first" not in methods or "repaired_retime" not in methods:
            continue
        xs.append(float(methods["K1_first"]["risk_score"]))
        ys.append(float(methods["repaired_retime"]["risk_score"]))
        colors.append("#2d8f67" if methods["repaired_retime"]["semantic_supported"] == "__YES__" else "#b05c3b")
    lim = max(max(xs), max(ys), 1.0) * 1.05
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(xs, ys, c=colors, s=30, alpha=0.82, edgecolor="none")
    ax.plot([0, lim], [0, lim], color="#222222", lw=1.0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("K=1 baseline risk")
    ax.set_ylabel("Repaired reference risk")
    ax.set_title("Paired risk reduction over 100 prompt identities")
    ax.text(0.03, 0.97, "below diagonal = improved", transform=ax.transAxes, va="top")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_risk_scatter.png", dpi=190)
    plt.close(fig)


def write_readme(out_dir: Path, summary: list[dict[str, object]]) -> None:
    by_method = {row["group"].split("=", 1)[1]: row for row in summary if row["group"].startswith("method=")}
    lines = [
        "# Humanoid100 Final Physical Evaluation",
        "",
        "This folder compares three references for every row in the 100-prompt",
        "benchmark: first MotionBricks proxy candidate, best-of-8 selection, and",
        "the repaired retiming/smoothing reference.",
        "",
        "## Headline Table",
        "",
        "| Method | Mean risk | Mean contact artifact | Physical pass | Presentation pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = by_method[method]
        lines.append(
            f"| {method} | {float(row['mean_risk']):.3f} | "
            f"{float(row['mean_contact_artifact_score']):.3f} | "
            f"{int(row['physical_pass_count'])}/100 | "
            f"{int(row['presentation_pass_count'])}/100 |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Physical pass ignores prompt semantics and only asks whether the qpos",
        "looks dynamically/contact feasible according to the verifier. Presentation",
        "pass also requires `semantic_validity=supported_proxy`. This distinction is",
        "essential because 78/100 benchmark rows are forced nearest-mode proxies in",
        "the current local MotionBricks preview.",
        "",
        "## Files",
        "",
        f"- `final_metrics.csv`",
        f"- `summary.csv`",
        f"- `method_summary_bars.png`",
        f"- `risk_by_category.png`",
        f"- `paired_risk_scatter.png`",
        "",
    ])
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motionbricks_csv", type=Path, default=DEFAULT_MOTIONBRICKS_CSV)
    parser.add_argument("--repair_csv", type=Path, default=DEFAULT_REPAIR_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate(args)
    write_csv(args.out_dir / "final_metrics.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary)
    plot_method_bars(rows, args.out_dir)
    plot_category_risk(rows, args.out_dir)
    plot_paired_improvement(rows, args.out_dir)
    write_readme(args.out_dir, summary)
    print(f"Wrote {len(rows)} metric rows to {args.out_dir / 'final_metrics.csv'}")
    print(f"Wrote summary and plots to {args.out_dir}")


if __name__ == "__main__":
    main()
