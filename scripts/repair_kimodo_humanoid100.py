"""Apply deterministic physical repair variants to KIMODO Humanoid100 clips.

This mirrors the earlier Humanoid100 retiming/smoothing repair, but uses the
KIMODO manifest/eval schema directly. It is a test-time curation baseline:
no KIMODO weights are changed, and the repair can change timing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_contact_quality import artifact_score, evaluate_clip  # noqa: E402
from evaluate_humanoid100_final import physical_pass  # noqa: E402
from export_sonic_references import SONIC_FPS, SOURCE_FPS, export_clip  # noqa: E402
from physics_eval.physaware import PhysicalAwarenessCritic  # noqa: E402
from physics_eval.simulator import SCENE_XML  # noqa: E402
from repair_humanoid100_references import repair_variants, validate_qpos  # noqa: E402


DEFAULT_EVAL = (
    ROOT
    / "results"
    / "kimodo_humanoid100_full_kimodo100_full_20260530_200838"
    / "eval"
    / "final_metrics.csv"
)
DEFAULT_OUT = ROOT / "results" / "kimodo_humanoid100_repaired_retimed"
DEFAULT_DATA = ROOT / "data" / "kimodo_humanoid100_repaired_retimed"


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


def score_candidate(
    critic: PhysicalAwarenessCritic,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    source: dict[str, str],
    variant: str,
) -> dict[str, object]:
    report, _ = critic.score(qpos, source["prompt_id"], source["category"], variant=variant)
    contact = evaluate_clip(model, qpos, f"{source['prompt_id']}_{variant}")
    contact["contact_artifact_score"] = artifact_score(contact)
    row: dict[str, object] = {
        "prompt_id": source["prompt_id"],
        "category": source["category"],
        "subcategory": source["subcategory"],
        "prompt_text": source["prompt_text"],
        "success_criteria": source.get("success_criteria", ""),
        "method": "Kimodo_G1_repaired",
        "variant": variant,
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
    row["physical_pass"] = "__YES__" if physical_pass(row) else "__NO__"
    return row


def selection_key(row: dict[str, object], duration_scale: float) -> tuple[int, float, float, float]:
    """Prefer physical-pass variants, then low risk, then smaller retiming."""

    return (
        0 if row["physical_pass"] == "__YES__" else 1,
        float(row["risk_score"]),
        float(row["contact_artifact_score"]),
        abs(duration_scale - 1.0),
    )


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def mean(key: str, subset: list[dict[str, object]]) -> float:
        return float(np.mean([float(r[key]) for r in subset])) if subset else float("nan")

    out: list[dict[str, object]] = []
    for group_name, subset in [
        ("original", rows),
        ("repaired", rows),
    ]:
        prefix = "original_" if group_name == "original" else "repaired_"
        out.append(
            {
                "group": group_name,
                "n": len(subset),
                "physical_pass_count": sum(r[f"{prefix}physical_pass"] == "__YES__" for r in subset),
                "accept_action_count": sum(r[f"{prefix}recommended_action"] == "accept" for r in subset),
                "repair_action_count": sum(r[f"{prefix}recommended_action"] == "repair_or_rerank" for r in subset),
                "reject_action_count": sum(r[f"{prefix}recommended_action"] == "reject_or_regenerate" for r in subset),
                "mean_risk": mean(f"{prefix}risk_score", subset),
                "mean_p95_torque_limit_ratio": mean(f"{prefix}p95_torque_limit_ratio", subset),
                "mean_contact_artifact_score": mean(f"{prefix}contact_artifact_score", subset),
            }
        )
    out.append(
        {
            "group": "paired_delta_repaired_minus_original",
            "n": len(rows),
            "physical_pass_count": (
                sum(r["repaired_physical_pass"] == "__YES__" for r in rows)
                - sum(r["original_physical_pass"] == "__YES__" for r in rows)
            ),
            "accept_action_count": (
                sum(r["repaired_recommended_action"] == "accept" for r in rows)
                - sum(r["original_recommended_action"] == "accept" for r in rows)
            ),
            "repair_action_count": (
                sum(r["repaired_recommended_action"] == "repair_or_rerank" for r in rows)
                - sum(r["original_recommended_action"] == "repair_or_rerank" for r in rows)
            ),
            "reject_action_count": (
                sum(r["repaired_recommended_action"] == "reject_or_regenerate" for r in rows)
                - sum(r["original_recommended_action"] == "reject_or_regenerate" for r in rows)
            ),
            "mean_risk": mean("delta_risk", rows),
            "mean_p95_torque_limit_ratio": mean("delta_p95_torque_limit_ratio", rows),
            "mean_contact_artifact_score": mean("delta_contact_artifact_score", rows),
        }
    )
    return out


def plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    before = np.array([float(r["original_risk_score"]) for r in rows])
    after = np.array([float(r["repaired_risk_score"]) for r in rows])
    before_pass = sum(r["original_physical_pass"] == "__YES__" for r in rows)
    after_pass = sum(r["repaired_physical_pass"] == "__YES__" for r in rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(["original", "repaired"], [before_pass, after_pass], color=["#777777", "#2d8f67"])
    axes[0].set_ylim(0, len(rows))
    axes[0].set_ylabel("physical-pass count")
    axes[0].set_title("KIMODO repair pass count")
    for i, v in enumerate([before_pass, after_pass]):
        axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

    axes[1].scatter(before, after, s=22, alpha=0.75)
    hi = max(float(np.nanmax(before)), float(np.nanmax(after)), 1.0)
    axes[1].plot([0, hi], [0, hi], color="#333333", lw=1)
    axes[1].set_xlabel("original risk")
    axes[1].set_ylabel("repaired risk")
    axes[1].set_title("Below diagonal = lower risk")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    fig.savefig(out_dir / "kimodo_repair_summary.png", dpi=190)
    plt.close(fig)


def write_readme(
    out_dir: Path,
    data_dir: Path,
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
) -> None:
    original = next(r for r in summary if r["group"] == "original")
    repaired = next(r for r in summary if r["group"] == "repaired")
    improved = sum(float(r["repaired_risk_score"]) < float(r["original_risk_score"]) for r in rows)
    rescued = sum(r["original_physical_pass"] != "__YES__" and r["repaired_physical_pass"] == "__YES__" for r in rows)
    harmed = sum(r["original_physical_pass"] == "__YES__" and r["repaired_physical_pass"] != "__YES__" for r in rows)
    lines = [
        "# KIMODO Humanoid100 Retiming/Smoothing Repair",
        "",
        "This applies deterministic retiming and smoothing variants to each KIMODO",
        "G1 qpos clip, then selects the variant with the best physical-screen key.",
        "It is a test-time reference repair baseline; it does not fine-tune KIMODO.",
        "",
        "## Summary",
        "",
        f"- Clips: {len(rows)}",
        f"- Physical pass original: {original['physical_pass_count']}/{len(rows)}",
        f"- Physical pass repaired: {repaired['physical_pass_count']}/{len(rows)}",
        f"- Newly rescued physical-pass clips: {rescued}",
        f"- Previously passing clips harmed: {harmed}",
        f"- Lower risk after repair: {improved}/{len(rows)}",
        f"- Mean risk original: {float(original['mean_risk']):.3f}",
        f"- Mean risk repaired: {float(repaired['mean_risk']):.3f}",
        "",
        "## Files",
        "",
        f"- Selected metrics: `{out_dir / 'repair_summary.csv'}`",
        f"- Candidate metrics: `{out_dir / 'candidate_metrics.csv'}`",
        f"- Summary: `{out_dir / 'summary.csv'}`",
        f"- Plot: `{out_dir / 'kimodo_repair_summary.png'}`",
        f"- Repaired qpos: `{data_dir}`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--export_sonic_refs", action="store_true")
    parser.add_argument("--source_fps", type=float, default=SOURCE_FPS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.export_sonic_refs:
        (args.out_dir / "sonic_references").mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    critic = PhysicalAwarenessCritic()
    rows = read_rows(args.eval_csv)
    if args.limit:
        rows = rows[: args.limit]

    selected_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    sonic_manifest: list[dict[str, object]] = []

    for idx, source in enumerate(rows, start=1):
        qpos = validate_qpos(np.load(source["qpos_path"]), source["qpos_path"])
        original = score_candidate(critic, model, qpos, source, "original")
        candidates: list[tuple[tuple[int, float, float, float], str, np.ndarray, dict[str, object], float]] = []
        for variant, candidate in repair_variants(qpos).items():
            duration_scale = len(candidate) / max(len(qpos), 1)
            metrics = score_candidate(critic, model, candidate, source, variant)
            candidate_rows.append(
                {
                    **metrics,
                    "qpos_path": source["qpos_path"],
                    "duration_scale": duration_scale,
                }
            )
            candidates.append((selection_key(metrics, duration_scale), variant, candidate, metrics, duration_scale))

        _, variant, selected_qpos, repaired, duration_scale = min(candidates, key=lambda item: item[0])
        out_qpos = args.data_dir / f"{source['prompt_id']}_{source['subcategory']}_{variant}.npy"
        np.save(out_qpos, selected_qpos.astype(np.float32))

        selected: dict[str, object] = {
            "prompt_id": source["prompt_id"],
            "category": source["category"],
            "subcategory": source["subcategory"],
            "prompt_text": source["prompt_text"],
            "success_criteria": source.get("success_criteria", ""),
            "original_qpos_path": source["qpos_path"],
            "repaired_qpos_path": str(out_qpos),
            "selected_variant": variant,
            "duration_scale": duration_scale,
        }
        for prefix, metrics in [("original", original), ("repaired", repaired)]:
            for key, value in metrics.items():
                if key in {"prompt_id", "category", "subcategory", "prompt_text", "success_criteria", "method", "variant"}:
                    continue
                selected[f"{prefix}_{key}"] = value
        selected["delta_risk"] = float(selected["repaired_risk_score"]) - float(selected["original_risk_score"])
        selected["delta_p95_torque_limit_ratio"] = (
            float(selected["repaired_p95_torque_limit_ratio"]) - float(selected["original_p95_torque_limit_ratio"])
        )
        selected["delta_contact_artifact_score"] = (
            float(selected["repaired_contact_artifact_score"]) - float(selected["original_contact_artifact_score"])
        )
        selected_rows.append(selected)

        if args.export_sonic_refs:
            name = f"{source['prompt_id']}_{source['subcategory']}_KimodoRepaired"
            exported = export_clip(out_qpos, args.out_dir / "sonic_references" / name, args.source_fps, SONIC_FPS)
            exported.update(
                {
                    "prompt_id": source["prompt_id"],
                    "subcategory": source["subcategory"],
                    "category": source["category"],
                    "method": "Kimodo_G1_repaired",
                    "selected_variant": variant,
                    "physical_pass": selected["repaired_physical_pass"],
                    "risk_score": selected["repaired_risk_score"],
                    "contact_artifact_score": selected["repaired_contact_artifact_score"],
                }
            )
            sonic_manifest.append(exported)

        if idx % 10 == 0 or idx == len(rows):
            write_csv(args.out_dir / "repair_summary.csv", selected_rows)
            write_csv(args.out_dir / "candidate_metrics.csv", candidate_rows)
            print(f"[{idx:03d}/{len(rows):03d}] wrote partial repair summary")

    summary = summarize(selected_rows)
    write_csv(args.out_dir / "repair_summary.csv", selected_rows)
    write_csv(args.out_dir / "candidate_metrics.csv", candidate_rows)
    write_csv(args.out_dir / "summary.csv", summary)
    if sonic_manifest:
        write_csv(args.out_dir / "sonic_references" / "manifest.csv", sonic_manifest)
    plot(selected_rows, args.out_dir)
    write_readme(args.out_dir, args.data_dir, selected_rows, summary)
    print(f"Wrote {len(selected_rows)} selected repairs to {args.out_dir / 'repair_summary.csv'}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
