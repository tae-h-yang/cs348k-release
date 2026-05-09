"""Build a research-facing task suite for the local MotionBricks G1 demo.

The public preview used in this repository exposes discrete G1 demo modes
rather than a full free-form text interface. This script therefore creates
natural-language task prompts that map onto the available mode/seed controls.
The suite is meant for evaluation breadth and presentation, not for claiming
general text-to-motion coverage.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "configs" / "prompt_suite_105.csv"
OUT_MD = ROOT / "docs" / "prompt_suite.md"


MODE_SPECS = [
    {
        "mode": "idle",
        "category": "static",
        "frames": 150,
        "speed": 0.0,
        "direction": 0.0,
        "template": "Stand upright in a relaxed idle pose without traveling.",
        "tags": "idle;standing;low_displacement",
    },
    {
        "mode": "walk",
        "category": "locomotion",
        "frames": 200,
        "speed": 0.65,
        "direction": 0.0,
        "template": "Walk forward at a natural pace with balanced arm swing.",
        "tags": "walk;forward;upright",
    },
    {
        "mode": "slow_walk",
        "category": "locomotion",
        "frames": 200,
        "speed": 0.35,
        "direction": 0.0,
        "template": "Walk forward slowly and carefully.",
        "tags": "slow_walk;forward;upright",
    },
    {
        "mode": "stealth_walk",
        "category": "locomotion",
        "frames": 200,
        "speed": 0.35,
        "direction": 0.0,
        "template": "Move forward in a stealthy low-profile walk.",
        "tags": "stealth;forward;low_posture",
    },
    {
        "mode": "injured_walk",
        "category": "locomotion",
        "frames": 200,
        "speed": 0.35,
        "direction": 0.0,
        "template": "Walk forward with an injured uneven gait while staying upright.",
        "tags": "injured;forward;asymmetric",
    },
    {
        "mode": "walk_zombie",
        "category": "style_locomotion",
        "frames": 200,
        "speed": 0.45,
        "direction": 0.0,
        "template": "Walk forward in a zombie-like style with expressive upper body.",
        "tags": "zombie;forward;expressive",
    },
    {
        "mode": "walk_stealth",
        "category": "style_locomotion",
        "frames": 180,
        "speed": 0.35,
        "direction": 0.0,
        "template": "Walk forward with a stealthy style and controlled torso motion.",
        "tags": "stealth;forward;style",
    },
    {
        "mode": "walk_left",
        "category": "directional_locomotion",
        "frames": 180,
        "speed": 0.45,
        "direction": 90.0,
        "template": "Travel to the robot's left while maintaining a walking gait.",
        "tags": "walk;left;directional",
    },
    {
        "mode": "walk_right",
        "category": "directional_locomotion",
        "frames": 180,
        "speed": 0.45,
        "direction": -90.0,
        "template": "Travel to the robot's right while maintaining a walking gait.",
        "tags": "walk;right;directional",
    },
    {
        "mode": "walk_boxing",
        "category": "expressive_locomotion",
        "frames": 180,
        "speed": 0.45,
        "direction": 0.0,
        "template": "Walk forward while throwing boxing-style arm motions.",
        "tags": "walk;boxing;arm_motion",
    },
    {
        "mode": "walk_happy_dance",
        "category": "expressive_locomotion",
        "frames": 180,
        "speed": 0.45,
        "direction": 0.0,
        "template": "Walk forward with a happy dance style and energetic arms.",
        "tags": "walk;happy_dance;arm_motion",
    },
    {
        "mode": "walk_gun",
        "category": "expressive_locomotion",
        "frames": 180,
        "speed": 0.45,
        "direction": 0.0,
        "template": "Walk forward while holding the arms in a pretend aiming pose.",
        "tags": "walk;aiming;arm_pose",
    },
    {
        "mode": "walk_scared",
        "category": "expressive_locomotion",
        "frames": 180,
        "speed": 0.35,
        "direction": 0.0,
        "template": "Walk forward in a scared style with guarded upper-body motion.",
        "tags": "walk;scared;expressive",
    },
    {
        "mode": "hand_crawling",
        "category": "whole_body_low",
        "frames": 150,
        "speed": 0.20,
        "direction": 0.0,
        "template": "Crawl forward on hands with a very low whole-body posture.",
        "tags": "crawl;hands;low_posture",
    },
    {
        "mode": "elbow_crawling",
        "category": "whole_body_low",
        "frames": 150,
        "speed": 0.18,
        "direction": 0.0,
        "template": "Crawl forward on elbows with a very low whole-body posture.",
        "tags": "crawl;elbows;low_posture",
    },
]


def build_rows(seeds: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in MODE_SPECS:
        for seed in range(seeds):
            pid = f"{spec['mode']}_seed{seed:02d}"
            rows.append({
                "prompt_id": pid,
                "category": spec["category"],
                "motionbricks_mode": spec["mode"],
                "seed_idx": seed,
                "n_frames": spec["frames"],
                "target_speed_mps": spec["speed"],
                "target_direction_deg": spec["direction"],
                "prompt_text": spec["template"],
                "expected_tags": spec["tags"],
                "available_in_current_results": "__YES__",
                "notes": (
                    "Mode-control prompt for local G1 preview; evaluate with "
                    "trajectory/task proxies rather than language retrieval."
                ),
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1

    lines = [
        "# Prompt/Task Suite",
        "",
        "This suite contains 105 natural-language task descriptions mapped onto",
        "the 15 exposed Unitree G1 MotionBricks preview modes and 7 seeds per",
        "mode. The current public preview is a mode/control interface, not a",
        "fully free-form text-to-motion generator, so prompt following is",
        "evaluated with trajectory and style proxies.",
        "",
        "## Category Counts",
        "",
    ]
    for category, n in sorted(counts.items()):
        lines.append(f"- `{category}`: {n}")
    lines.extend([
        "",
        "## Evaluation Proxies",
        "",
        "- Direction/progress: root displacement along the requested direction.",
        "- Speed: path-length speed compared with the nominal task speed.",
        "- Static stability: low root displacement and low joint motion.",
        "- Low-posture tasks: low mean root height.",
        "- Expressive tasks: upper-body motion relative to leg motion.",
        "- Preservation: compare K=1 and K=8 on displacement, speed, and proxy score.",
        "",
        "These proxies are weaker than HumanML3D-style R-Precision/FID with a",
        "learned text-motion evaluator; that limitation is explicit in the report.",
        "",
        "## First 20 Tasks",
        "",
        "| ID | Category | Prompt |",
        "|---|---|---|",
    ])
    for row in rows[:20]:
        lines.append(f"| `{row['prompt_id']}` | `{row['category']}` | {row['prompt_text']} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=7)
    parser.add_argument("--out_csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out_md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    rows = build_rows(args.seeds)
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows)
    print(f"Wrote {len(rows)} tasks to {args.out_csv}")
    print(f"Wrote notes to {args.out_md}")


if __name__ == "__main__":
    main()
