#!/usr/bin/env python3
"""Join Humanoid100 selector metadata with native SONIC rollout results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = (
    ROOT
    / "results"
    / "ralphloop"
    / "20260529_191342"
    / "humanoid100_final_eval_k256"
    / "all100_native_sonic_release"
)
DEFAULT_SELECTED = (
    ROOT
    / "results"
    / "ralphloop"
    / "20260529_191342"
    / "humanoid100_final_eval_k256"
    / "final_selector_initref"
    / "selected_methods.csv"
)


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


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def strict_pass(row: dict[str, str]) -> bool:
    return (
        row.get("fell") == "False"
        and float(row.get("mean_joint_rmse", 999.0) or 999.0) <= 0.20
        and float(row.get("mean_root_xy_error", 999.0) or 999.0) <= 1.5
    )


def summarize(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    out: list[dict[str, object]] = []
    for group, group_rows in sorted(groups.items()):
        n = len(group_rows)
        no_fall = sum(bool(r["native_no_fall"]) for r in group_rows)
        strict = sum(bool(r["native_strict_pass"]) for r in group_rows)
        semantic = sum(str(r["semantic_supported"]) == "__YES__" for r in group_rows)
        physical = sum(str(r["physical_pass"]) == "__YES__" for r in group_rows)
        out.append(
            {
                group_key: group,
                "n": n,
                "semantic_supported": semantic,
                "physical_pass": physical,
                "native_no_fall": no_fall,
                "native_no_fall_rate": no_fall / n,
                "native_strict_pass": strict,
                "native_strict_pass_rate": strict / n,
                "mean_native_rmse": mean([float(r["mean_joint_rmse"]) for r in group_rows]),
                "mean_risk_score": mean([float(r["risk_score"]) for r in group_rows]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--selected_csv", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--selector", default="sonic_verified_best")
    args = parser.parse_args()

    batch_rows = {row["motion"]: row for row in read_rows(args.batch_dir / "batch_summary.csv")}
    selected_rows = [row for row in read_rows(args.selected_csv) if row["selector"] == args.selector]
    if len(selected_rows) != 100:
        raise ValueError(f"Expected 100 selected rows for {args.selector}, got {len(selected_rows)}")

    joined: list[dict[str, object]] = []
    for row in selected_rows:
        motion = row["selected_reference"]
        native = batch_rows.get(motion)
        if native is None:
            raise ValueError(f"Missing native rollout for {motion}")
        no_fall = native["fell"] == "False"
        joined.append(
            {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "prompt_text": row["prompt_text"],
                "semantic_validity": row["semantic_validity"],
                "semantic_supported": row["semantic_supported"],
                "selected_reference": motion,
                "selected_method": row["selected_method"],
                "physical_pass": row["physical_pass"],
                "risk_score": float(row["risk_score"]),
                "native_no_fall": no_fall,
                "native_strict_pass": strict_pass(native),
                "fall_time_s": float(native["fall_time_s"]),
                "mean_joint_rmse": float(native["mean_joint_rmse"]),
                "mean_root_xy_error": float(native["mean_root_xy_error"]),
                "min_root_z": float(native["min_root_z"]),
                "video": native["video"],
            }
        )

    out_dir = args.batch_dir
    write_rows(out_dir / "humanoid100_native_joined.csv", joined)
    category_rows = summarize(joined, "category")
    semantic_rows = summarize(joined, "semantic_validity")
    method_rows = summarize(joined, "selected_method")
    write_rows(out_dir / "humanoid100_native_by_category.csv", category_rows)
    write_rows(out_dir / "humanoid100_native_by_semantic_validity.csv", semantic_rows)
    write_rows(out_dir / "humanoid100_native_by_selected_method.csv", method_rows)

    failures = [row for row in joined if not row["native_no_fall"]]
    failures = sorted(failures, key=lambda r: (str(r["category"]), str(r["subcategory"])))
    write_rows(out_dir / "humanoid100_native_failures.csv", failures)

    no_fall = sum(bool(row["native_no_fall"]) for row in joined)
    strict = sum(bool(row["native_strict_pass"]) for row in joined)
    supported = [row for row in joined if row["semantic_supported"] == "__YES__"]
    proxy = [row for row in joined if row["semantic_supported"] != "__YES__"]
    supported_pass = sum(bool(row["native_no_fall"]) for row in supported)
    proxy_pass = sum(bool(row["native_no_fall"]) for row in proxy)

    lines = [
        "# Humanoid100 Native SONIC Join",
        "",
        f"- selector: `{args.selector}`",
        f"- prompts: `{len(joined)}`",
        f"- native no-fall: `{no_fall}/{len(joined)}`",
        f"- native strict pass: `{strict}/{len(joined)}`",
        f"- semantic-supported subset no-fall: `{supported_pass}/{len(supported)}`",
        f"- proxy-only subset no-fall: `{proxy_pass}/{len(proxy)}`",
        f"- mean native RMSE: `{mean([float(row['mean_joint_rmse']) for row in joined]):.3f}`",
        "",
        "## By Prompt Category",
        "",
        "| category | n | no-fall | strict | semantic supported | physical pass | mean RMSE | mean risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in category_rows:
        lines.append(
            f"| {row['category']} | {row['n']} | {row['native_no_fall']} "
            f"({100 * row['native_no_fall_rate']:.1f}%) | {row['native_strict_pass']} "
            f"({100 * row['native_strict_pass_rate']:.1f}%) | {row['semantic_supported']} | "
            f"{row['physical_pass']} | {row['mean_native_rmse']:.3f} | {row['mean_risk_score']:.2f} |"
        )
    lines += [
        "",
        "## Failure Motions",
        "",
    ]
    for row in failures:
        lines.append(
            f"- `{row['prompt_id']}` `{row['subcategory']}` ({row['category']}): "
            f"{row['selected_reference']}, fall={row['fall_time_s']:.2f}s, "
            f"rmse={row['mean_joint_rmse']:.3f}, min_z={row['min_root_z']:.3f}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Native SONIC is substantially more favorable than the approximate Python "
        "bridge, but this is still not arbitrary prompt-conditioned generation. "
        "Most successes are executable proxy motions selected from MotionBricks' "
        "available behavior modes. The failure list remains concentrated in "
        "floor transitions, crawling, rolls, and acrobatic stress tests.",
        "",
    ]
    (out_dir / "humanoid100_native_analysis.md").write_text("\n".join(lines))
    print(f"Wrote {out_dir / 'humanoid100_native_analysis.md'}")


if __name__ == "__main__":
    main()
