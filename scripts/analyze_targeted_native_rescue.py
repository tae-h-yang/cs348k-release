#!/usr/bin/env python3
"""Analyze targeted native SONIC rescue candidates against a final selection table."""

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
    parser.add_argument("--base_selection", type=Path, default=DEFAULT_BASE / "final_100_native_selection_ref_aware.csv")
    parser.add_argument("--batch_dir", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--out_md", type=Path, default=None)
    args = parser.parse_args()

    base = read_rows(args.base_selection)
    base_by_prompt = {row["prompt_id"]: row for row in base}
    batch = read_rows(args.batch_dir / "batch_summary.csv")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in batch:
        if row.get("status") and row["status"] != "completed":
            continue
        groups.setdefault(prompt_id_from_motion(row["motion"]), []).append(row)

    rows: list[dict[str, object]] = []
    for prompt_id, group in sorted(groups.items()):
        base_row = base_by_prompt[prompt_id]
        best = sorted(group, key=sort_key)[0]
        rows.append(
            {
                "prompt_id": prompt_id,
                "category": base_row["category"],
                "subcategory": base_row["subcategory"],
                "base_reference": base_row["selected_reference"],
                "base_strict": as_bool(base_row["strict_tracking_pass"]),
                "base_rmse": as_float(base_row, "mean_joint_rmse"),
                "base_root_xy": as_float(base_row, "mean_root_xy_error"),
                "best_candidate": best["motion"],
                "best_no_fall": native_no_fall(best),
                "best_strict": native_strict(best),
                "best_rmse": as_float(best, "mean_joint_rmse"),
                "best_root_xy": as_float(best, "mean_root_xy_error"),
                "best_fall_time_s": as_float(best, "fall_time_s", 0.0),
                "candidate_count": len(group),
                "strict_candidate_count": sum(native_strict(row) for row in group),
                "video": best.get("video", ""),
            }
        )

    out_csv = args.out_csv or args.batch_dir / "targeted_native_rescue.csv"
    out_md = args.out_md or args.batch_dir / "targeted_native_rescue.md"
    write_rows(out_csv, rows)
    base_strict = sum(as_bool(row["strict_tracking_pass"]) for row in base)
    strict_rescues = [row for row in rows if row["best_strict"] and not row["base_strict"]]
    projected = min(100, base_strict + len(strict_rescues))
    lines = [
        "# Targeted Native Rescue Analysis",
        "",
        f"- completed prompt groups: `{len(rows)}`",
        f"- completed native candidates: `{sum(int(row['candidate_count']) for row in rows)}`",
        f"- new strict rescues: `{len(strict_rescues)}/{len(rows)}`",
        f"- base strict: `{base_strict}/100`",
        f"- projected strict if repeat-stable: `{projected}/100`",
        "",
        "## New Strict Rescues",
        "",
    ]
    if strict_rescues:
        for row in strict_rescues:
            lines.append(
                f"- `{row['prompt_id']}` `{row['subcategory']}`: "
                f"{row['base_reference']} -> {row['best_candidate']}, "
                f"rmse={row['best_rmse']:.3f}, root_xy={row['best_root_xy']:.3f}"
            )
    else:
        lines.append("- None yet.")
    lines += ["", "## Best Candidates Per Completed Prompt", ""]
    for row in rows:
        lines.append(
            f"- `{row['prompt_id']}` `{row['subcategory']}`: best={row['best_candidate']}, "
            f"strict={row['best_strict']}, rmse={row['best_rmse']:.3f}, "
            f"root_xy={row['best_root_xy']:.3f}"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"new strict rescues {len(strict_rescues)}/{len(rows)}; projected {projected}/100")


if __name__ == "__main__":
    main()
