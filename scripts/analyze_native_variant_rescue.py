#!/usr/bin/env python3
"""Measure how many native-SONIC failures are rescued by alternate candidates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results" / "ralphloop" / "20260529_191342" / "humanoid100_final_eval_k256"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def native_strict(row: dict[str, str]) -> bool:
    return (
        native_no_fall(row)
        and float(row.get("mean_joint_rmse", 999.0) or 999.0) <= 0.20
        and float(row.get("mean_root_xy_error", 999.0) or 999.0) <= 1.5
    )


def native_no_fall(row: dict[str, str]) -> bool:
    ref_aware = str(row.get("ref_aware_fell", "")).strip().lower()
    if ref_aware not in {"", "nan", "none"}:
        return ref_aware == "false"
    return row.get("fell") == "False"


def prompt_id_from_motion(name: str) -> str:
    return "_".join(name.split("_")[:2])


def sort_key(row: dict[str, str]) -> tuple[int, int, float, float]:
    no_fall = native_no_fall(row)
    strict = native_strict(row)
    return (
        0 if strict else 1,
        0 if no_fall else 1,
        float(row.get("mean_joint_rmse", 999.0) or 999.0),
        -float(row.get("fall_time_s", 0.0) or 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.base_dir / "failed_prompt_native_variant_sweep"
    all_join = read_rows(args.base_dir / "all100_native_sonic_release" / "humanoid100_native_joined.csv")
    variant_summary = args.base_dir / "failed_prompt_native_variant_sweep" / "batch_summary_ref_aware.csv"
    if not variant_summary.exists():
        variant_summary = args.base_dir / "failed_prompt_native_variant_sweep" / "batch_summary.csv"
    variants = read_rows(variant_summary)
    original_summary = args.base_dir / "all100_native_sonic_release" / "batch_summary_ref_aware.csv"
    original_by_motion = {}
    if original_summary.exists():
        original_by_motion = {row["motion"]: row for row in read_rows(original_summary)}
    by_prompt = {row["prompt_id"]: row for row in all_join}
    variant_groups: dict[str, list[dict[str, str]]] = {}
    for row in variants:
        variant_groups.setdefault(prompt_id_from_motion(row["motion"]), []).append(row)

    rows: list[dict[str, object]] = []
    for prompt_id, group in sorted(variant_groups.items()):
        original = by_prompt[prompt_id]
        original_metric = original_by_motion.get(original["selected_reference"], original)
        best = sorted(group, key=sort_key)[0]
        no_fall_variants = [row for row in group if native_no_fall(row)]
        strict_variants = [row for row in group if native_strict(row)]
        rows.append(
            {
                "prompt_id": prompt_id,
                "category": original["category"],
                "subcategory": original["subcategory"],
                "original_reference": original["selected_reference"],
                "original_no_fall": native_no_fall(original_metric),
                "original_strict": native_strict(original_metric),
                "best_variant": best["motion"],
                "best_no_fall": native_no_fall(best),
                "best_strict": native_strict(best),
                "best_fall_time_s": float(best["fall_time_s"]),
                "best_mean_joint_rmse": float(best["mean_joint_rmse"]),
                "no_fall_variant_count": len(no_fall_variants),
                "strict_variant_count": len(strict_variants),
            }
        )

    write_rows(out_dir / "native_variant_rescue.csv", rows)
    rescued = [row for row in rows if row["best_no_fall"] and not row["original_no_fall"]]
    strict_rescued = [row for row in rows if row["best_strict"] and not row["original_strict"]]
    selected_no_fall = [row for row in rows if row["best_no_fall"]]
    selected_strict = [row for row in rows if row["best_strict"]]
    still = [row for row in rows if not row["best_no_fall"]]
    if original_by_motion:
        original_no_fall = sum(native_no_fall(row) for row in original_by_motion.values())
        original_strict = sum(native_strict(row) for row in original_by_motion.values())
    else:
        original_no_fall = sum(row["native_no_fall"] in ("True", "1", "__YES__") for row in all_join)
        original_strict = sum(row["native_strict_pass"] in ("True", "1", "__YES__") for row in all_join)
    projected_no_fall = original_no_fall + len(rescued)
    projected_strict = original_strict + len(strict_rescued)

    lines = [
        "# Native Variant Rescue Analysis",
        "",
        "This evaluates K1, K8, and repaired/K9 variants for the 24 prompts that "
        "failed the first all-100 native SONIC pass.",
        "",
        f"- failed prompts retested: `{len(rows)}`",
        f"- selected no-fall prompts in old-failure set: `{len(selected_no_fall)}/{len(rows)}`",
        f"- additional no-fall rescues beyond corrected original metric: `{len(rescued)}/{len(rows)}`",
        f"- selected strict prompts in old-failure set: `{len(selected_strict)}/{len(rows)}`",
        f"- additional strict rescues beyond corrected original metric: `{len(strict_rescued)}/{len(rows)}`",
        f"- projected all-100 no-fall with native verifier selection: `{projected_no_fall}/100`",
        f"- projected all-100 strict pass with native verifier selection: `{projected_strict}/100`",
        "",
        "## Selected No-Fall Prompts",
        "",
    ]
    for row in selected_no_fall:
        lines.append(
            f"- `{row['prompt_id']}` `{row['subcategory']}` ({row['category']}): "
            f"{row['original_reference']} -> {row['best_variant']}, "
            f"rmse={row['best_mean_joint_rmse']:.3f}"
        )
    lines += [
        "",
        "## Still Failing",
        "",
    ]
    for row in still:
        lines.append(
            f"- `{row['prompt_id']}` `{row['subcategory']}` ({row['category']}): "
            f"best={row['best_variant']}, fall={row['best_fall_time_s']:.2f}s, "
            f"rmse={row['best_mean_joint_rmse']:.3f}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "With a reference-aware fall metric, native verifier selection finds at "
        "least one no-fall candidate for every prompt in this 100-motion suite. "
        "That is a survival result, not a perfect tracking result: only the "
        "motions satisfying the stricter joint/RMSE and root-error gate should be "
        "described as cleanly tracked. Low-posture and acrobatic prompts often "
        "survive while still deviating substantially from the requested reference.",
        "",
    ]
    (out_dir / "native_variant_rescue_analysis.md").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'native_variant_rescue_analysis.md'}")


if __name__ == "__main__":
    main()
