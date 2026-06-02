#!/usr/bin/env python3
"""Build the final 100-prompt native SONIC selection table."""

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
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "__yes__"}


def as_float(row: dict[str, str], key: str, default: float = 999.0) -> float:
    try:
        text = row.get(key, "")
        return float(text) if text not in ("", None) else default
    except ValueError:
        return default


def native_no_fall(row: dict[str, str]) -> bool:
    ref_aware = str(row.get("ref_aware_fell", "")).strip().lower()
    if ref_aware not in {"", "nan", "none"}:
        return not as_bool(ref_aware)
    return not as_bool(row.get("fell"))


def native_strict(row: dict[str, str]) -> bool:
    return native_no_fall(row) and as_float(row, "mean_joint_rmse") <= 0.20 and as_float(row, "mean_root_xy_error") <= 1.5


def prompt_id_from_motion(name: str) -> str:
    return "_".join(name.split("_")[:2])


def sort_key(row: dict[str, str]) -> tuple[int, int, float, float, float]:
    return (
        0 if native_strict(row) else 1,
        0 if native_no_fall(row) else 1,
        as_float(row, "mean_joint_rmse"),
        as_float(row, "mean_root_xy_error"),
        -as_float(row, "fall_time_s", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    parser.add_argument("--include_deep", action="store_true")
    parser.add_argument(
        "--extra_batch_csv",
        action="append",
        type=Path,
        default=[],
        help="Additional native SONIC batch_summary.csv files to include as candidate sources.",
    )
    args = parser.parse_args()

    base = args.base_dir
    joined = read_rows(base / "all100_native_sonic_release" / "humanoid100_native_joined.csv")
    original = {row["motion"]: row for row in read_rows(base / "all100_native_sonic_release" / "batch_summary_ref_aware.csv")}
    variant_path = base / "failed_prompt_native_variant_sweep" / "batch_summary_ref_aware.csv"
    variants = read_rows(variant_path) if variant_path.exists() else []
    variant_groups: dict[str, list[dict[str, str]]] = {}
    for row in variants:
        variant_groups.setdefault(prompt_id_from_motion(row["motion"]), []).append(row)
    deep_path = base / "deep_failure_native_sonic" / "batch_summary.csv"
    deep = read_rows(deep_path) if args.include_deep and deep_path.exists() else []
    deep_groups: dict[str, list[dict[str, str]]] = {}
    for row in deep:
        if row.get("status") and row["status"] != "completed":
            continue
        deep_groups.setdefault(prompt_id_from_motion(row["motion"]), []).append(row)
    extra_groups: dict[str, list[dict[str, str]]] = {}
    for extra_csv in args.extra_batch_csv:
        for row in read_rows(extra_csv):
            if row.get("status") and row["status"] != "completed":
                continue
            extra_groups.setdefault(prompt_id_from_motion(row["motion"]), []).append(row)

    rows: list[dict[str, object]] = []
    for meta in sorted(joined, key=lambda r: r["prompt_id"]):
        prompt_id = meta["prompt_id"]
        original_row = original[meta["selected_reference"]]
        candidates = (
            [original_row]
            + variant_groups.get(prompt_id, [])
            + deep_groups.get(prompt_id, [])
            + extra_groups.get(prompt_id, [])
        )
        best = sorted(candidates, key=sort_key)[0]
        rows.append(
            {
                "prompt_id": prompt_id,
                "category": meta["category"],
                "subcategory": meta["subcategory"],
                "prompt_text": meta.get("prompt_text", ""),
                "original_reference": meta["selected_reference"],
                "selected_reference": best["motion"],
                "selection_changed": best["motion"] != meta["selected_reference"],
                "ref_aware_no_fall": native_no_fall(best),
                "strict_tracking_pass": native_strict(best),
                "fixed_height_fell": as_bool(best.get("fell")),
                "fall_time_s": as_float(best, "fall_time_s", 0.0),
                "ref_aware_fall_time_s": as_float(best, "ref_aware_fall_time_s", as_float(best, "fall_time_s", 0.0)),
                "mean_joint_rmse": as_float(best, "mean_joint_rmse"),
                "mean_root_xy_error": as_float(best, "mean_root_xy_error"),
                "min_root_z": as_float(best, "min_root_z"),
                "ref_aware_root_z_threshold": as_float(best, "ref_aware_root_z_threshold", 0.55),
                "reference_root": best.get("reference_root", ""),
                "video": best.get("video", ""),
            }
        )

    out_csv = args.out_csv or base / "final_100_native_selection_ref_aware.csv"
    out_md = args.out_md or base / "final_100_native_selection_ref_aware.md"
    write_rows(out_csv, rows)

    no_fall = sum(as_bool(row["ref_aware_no_fall"]) for row in rows)
    strict = sum(as_bool(row["strict_tracking_pass"]) for row in rows)
    changed = sum(as_bool(row["selection_changed"]) for row in rows)
    by_category: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(row)

    lines = [
        "# Final 100 Native SONIC Selection",
        "",
        f"- prompts: `{len(rows)}`",
        f"- selected references changed by verifier: `{changed}/{len(rows)}`",
        f"- reference-aware no-fall: `{no_fall}/{len(rows)}`",
        f"- strict tracking pass: `{strict}/{len(rows)}`",
        "",
        "## By Category",
        "",
        "| category | prompts | no-fall | strict | mean RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, group in sorted(by_category.items()):
        group_no_fall = sum(as_bool(row["ref_aware_no_fall"]) for row in group)
        group_strict = sum(as_bool(row["strict_tracking_pass"]) for row in group)
        mean_rmse = sum(float(row["mean_joint_rmse"]) for row in group) / len(group)
        lines.append(f"| {category} | {len(group)} | {group_no_fall} | {group_strict} | {mean_rmse:.3f} |")

    lines += [
        "",
        "## Non-Strict Selections",
        "",
    ]
    for row in rows:
        if as_bool(row["strict_tracking_pass"]):
            continue
        lines.append(
            f"- `{row['prompt_id']}` `{row['subcategory']}` ({row['category']}): "
            f"selected={row['selected_reference']}, rmse={row['mean_joint_rmse']:.3f}, "
            f"root_xy={row['mean_root_xy_error']:.3f}, no_fall={row['ref_aware_no_fall']}"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"ref-aware no-fall {no_fall}/{len(rows)}; strict {strict}/{len(rows)}")


if __name__ == "__main__":
    main()
