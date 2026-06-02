"""Build a sports/acrobatic stress-test prompt suite.

The current local MotionBricks G1 preview does not expose true sport or
acrobatic controls. This suite defines target prompts and evaluation criteria
for future generator/retargeting experiments while making current support
status explicit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "configs" / "sports_acrobatics_stress_prompts.csv"
OUT_MD = ROOT / "docs" / "autonomous_loop" / "sports_acrobatics_stress_prompts.md"


Prompt = dict[str, str]


def p(domain: str, subcategory: str, prompt: str, criteria: str, contacts: str, root_motion: str, arm_role: str) -> Prompt:
    return {
        "domain": domain,
        "subcategory": subcategory,
        "prompt_text": prompt,
        "success_criteria": criteria,
        "expected_primary_contacts": contacts,
        "expected_root_motion": root_motion,
        "expected_arm_role": arm_role,
    }


PROMPTS: list[Prompt] = [
    # Acrobatics and floor transitions.
    p("acrobatics", "cartwheel_left", "Perform a slow cartwheel to the left and recover to standing.", "hands contact floor; body inverts laterally; stable final stance", "feet;hands", "lateral_inversion", "support_or_recovery"),
    p("acrobatics", "cartwheel_right", "Perform a slow cartwheel to the right and recover to standing.", "hands contact floor; body inverts laterally; stable final stance", "feet;hands", "lateral_inversion", "support_or_recovery"),
    p("acrobatics", "roundoff", "Perform a cautious roundoff-like half cartwheel and land facing backward.", "hand support; yaw change; two-foot landing", "feet;hands", "lateral_inversion", "support_or_recovery"),
    p("acrobatics", "forward_roll", "Perform a controlled forward roll and recover to kneeling.", "body-floor roll; low impact; controlled kneel", "body_floor;hands;knees", "floor_roll", "support_or_recovery"),
    p("acrobatics", "side_roll", "Roll sideways on the floor and recover to a crouched stance.", "side roll; low root; crouch recovery", "body_floor;hands;feet", "floor_roll", "support_or_recovery"),
    p("acrobatics", "handstand_kickup", "Kick up toward a brief handstand and return both feet to the floor.", "hand support; inverted hips; controlled return", "hands;feet", "vertical_inversion", "support_or_recovery"),
    p("acrobatics", "backbend_bridge", "Lower into a backbend bridge and return to standing.", "hands and feet support; arched torso; stable recovery", "hands;feet", "deep_extension", "support_or_recovery"),
    p("acrobatics", "burpee", "Squat down, kick the legs back to a plank, return the feet under the body, and stand.", "squat; plank; stand recovery", "feet;hands", "low_transition", "support_or_recovery"),
    p("acrobatics", "split_squat_jump", "Perform a small split-squat jump switching which foot is forward.", "brief flight; leg switch; controlled landing", "feet", "aerial_or_impulsive", "natural_or_unspecified"),
    p("acrobatics", "knee_slide_recovery", "Slide forward briefly on both knees and recover to kneeling.", "knee contact; forward slide; controlled stop", "knees;feet", "low_transition", "support_or_recovery"),

    # Soccer.
    p("soccer", "soccer_inside_pass_right", "Step forward and pass an imaginary soccer ball with the inside of the right foot.", "right-foot swing; planted left support; controlled recovery", "feet", "mostly_stationary", "balance"),
    p("soccer", "soccer_inside_pass_left", "Step forward and pass an imaginary soccer ball with the inside of the left foot.", "left-foot swing; planted right support; controlled recovery", "feet", "mostly_stationary", "balance"),
    p("soccer", "soccer_power_shot", "Take two approach steps and perform a powerful right-foot soccer shot.", "approach; large right-leg swing; stable landing", "feet", "forward_or_task_dependent", "balance"),
    p("soccer", "soccer_dribble", "Dribble an imaginary soccer ball forward with alternating light taps.", "forward progress; alternating foot taps; no fall", "feet", "forward_or_task_dependent", "natural_or_unspecified"),
    p("soccer", "soccer_goalkeeper_dive", "Dive sideways like a goalkeeper and recover to the ground safely.", "lateral launch; body-floor contact; controlled landing", "body_floor;hands", "lateral_fall_recovery", "support_or_recovery"),
    p("soccer", "soccer_trap_ball", "Lift the right foot to trap an imaginary soccer ball and set it down softly.", "single-leg support; foot lift; controlled set-down", "feet", "mostly_stationary", "balance"),

    # Baseball and softball.
    p("baseball", "baseball_pitch_right", "Step forward and throw an overhand baseball pitch with the right arm.", "right-arm windup; forward step; follow-through", "feet", "forward_or_task_dependent", "task_constrained"),
    p("baseball", "baseball_bat_swing_right", "Swing an imaginary baseball bat from the right-handed stance.", "two-hand swing; torso rotation; feet stable", "feet", "mostly_stationary_turn", "task_constrained"),
    p("baseball", "baseball_bat_swing_left", "Swing an imaginary baseball bat from the left-handed stance.", "two-hand swing; torso rotation; feet stable", "feet", "mostly_stationary_turn", "task_constrained"),
    p("baseball", "baseball_field_grounder", "Shuffle right and bend down to field a rolling ground ball.", "lateral steps; low reach; recover upright", "feet", "rightward_low_transition", "task_constrained"),
    p("baseball", "baseball_catch_fly_ball", "Backpedal and raise both hands to catch an imaginary fly ball.", "backward steps; hands overhead; stable final pose", "feet", "backward", "task_constrained"),
    p("baseball", "baseball_slide_base", "Perform a feet-first slide into an imaginary base and stop safely.", "low transition; floor contact; controlled stop", "body_floor;feet", "low_transition", "support_or_recovery"),

    # Basketball.
    p("basketball", "basketball_free_throw", "Bend the knees and shoot an imaginary basketball free throw.", "knee bend; both arms extend upward; stable feet", "feet", "mostly_stationary", "task_constrained"),
    p("basketball", "basketball_jump_shot", "Perform a small jump shot with both arms extending overhead.", "brief flight; overhead release; controlled landing", "feet", "aerial_or_impulsive", "task_constrained"),
    p("basketball", "basketball_defensive_slide", "Defensive-slide two steps to the left with knees bent and arms out.", "left lateral motion; low stance; arms wide", "feet", "leftward", "task_constrained"),
    p("basketball", "basketball_dribble_low", "Crouch slightly and mime a low right-hand basketball dribble while stepping forward.", "right hand cyclic low motion; forward steps; stable trunk", "feet", "forward_or_task_dependent", "task_constrained"),

    # Other sports and combat-like stressors.
    p("racket_sports", "tennis_forehand", "Step and swing an imaginary tennis forehand across the body.", "one-step weight shift; arm swing; torso rotation", "feet", "mostly_stationary_turn", "task_constrained"),
    p("racket_sports", "tennis_backhand", "Step and swing an imaginary two-handed tennis backhand.", "two-hand swing; torso rotation; stable stance", "feet", "mostly_stationary_turn", "task_constrained"),
    p("racket_sports", "tennis_serve", "Toss an imaginary ball and perform an overhead tennis serve.", "overhead arm swing; trunk extension; stable recovery", "feet", "mostly_stationary", "task_constrained"),
    p("martial_arts", "front_kick_right", "Lift the right knee and perform a controlled forward front kick.", "single-leg support; right kick; stable recovery", "feet", "mostly_stationary", "balance"),
    p("martial_arts", "roundhouse_kick_left", "Pivot and perform a controlled left roundhouse kick.", "support-foot pivot; left leg swing; recovery", "feet", "mostly_stationary_turn", "balance"),
    p("track_field", "sprinter_start", "Start from a crouched sprint stance and take two explosive steps forward.", "low start; fast forward acceleration; no hand collapse", "feet;hands", "forward_or_task_dependent", "support_or_recovery"),
]


PARTIAL_MODE_HINTS = {
    "soccer_dribble": "walk",
    "basketball_defensive_slide": "walk_left",
    "baseball_field_grounder": "walk_right",
    "baseball_catch_fly_ball": "injured_walk",
    "front_kick_right": "walk_gun",
    "roundhouse_kick_left": "walk_happy_dance",
}


def support_label(row: Prompt) -> tuple[str, str, str]:
    subcategory = row["subcategory"]
    domain = row["domain"]
    if subcategory in PARTIAL_MODE_HINTS:
        return "__PARTIAL__", PARTIAL_MODE_HINTS[subcategory], "Available mode may test coarse locomotion or style only; sport-specific limb/object semantics are not supported."
    if domain == "acrobatics":
        return "__NO__", "negative_control_high_risk", "Unsupported by the local MotionBricks preview; evaluate only as future high-risk target behavior."
    if domain in {"soccer", "baseball", "basketball", "racket_sports", "martial_arts", "track_field"}:
        return "__NO__", "requires_task_conditioned_generator_or_retargeter", "Requires sport/task-conditioned generation or retargeting before native SONIC evaluation."
    return "__NO__", "requires_new_generator_control", "Unsupported by current local generator."


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, row in enumerate(PROMPTS, start=1):
        text = row["prompt_text"].strip()
        key = text.lower()
        if key in seen:
            raise ValueError(f"Duplicate prompt text: {text}")
        seen.add(key)
        support, hint, notes = support_label(row)
        rows.append({
            "prompt_id": f"sport_{idx:03d}",
            "domain": row["domain"],
            "subcategory": row["subcategory"],
            "prompt_text": text,
            "success_criteria": row["success_criteria"],
            "expected_primary_contacts": row["expected_primary_contacts"],
            "expected_root_motion": row["expected_root_motion"],
            "expected_arm_role": row["expected_arm_role"],
            "current_motionbricks_support": support,
            "motionbricks_mode_hint": hint,
            "evaluation_notes": notes,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    support_counts: dict[str, int] = {}
    for row in rows:
        counts[row["domain"]] = counts.get(row["domain"], 0) + 1
        support_counts[row["current_motionbricks_support"]] = support_counts.get(row["current_motionbricks_support"], 0) + 1

    lines = [
        "# Sports and Acrobatics Stress Prompt Suite",
        "",
        "This suite adds high-risk acrobatic and sports motions requested after the",
        "broad13 native SONIC study. It is a target benchmark layer, not a claim",
        "that the current local MotionBricks preview can generate every behavior.",
        "",
        "## Domain Counts",
        "",
    ]
    for domain, count in sorted(counts.items()):
        lines.append(f"- `{domain}`: {count}")
    lines.extend(["", "## Current MotionBricks Support", ""])
    for support, count in sorted(support_counts.items()):
        lines.append(f"- `{support}`: {count}")
    lines.extend([
        "",
        "## Evaluation Plan",
        "",
        "- Use current partial mode hints only as coarse baselines.",
        "- Mark acrobatic inversions, dives, slides, and sport-object interactions as unsupported until a task-conditioned generator or retargeter exists.",
        "- For future executable clips, require native SONIC rollout, contact/root-height checks, sport-specific event predicates, and frame-level visual audit.",
        "",
        "## Prompts",
        "",
        "| ID | Domain | Subcategory | Prompt | Current support |",
        "|---|---|---|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| `{row['prompt_id']}` | `{row['domain']}` | `{row['subcategory']}` | "
            f"{row['prompt_text']} | `{row['current_motionbricks_support']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out_md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    rows = build_rows()
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows)
    print(f"Wrote {len(rows)} sports/acrobatic prompts to {args.out_csv}")
    print(f"Wrote summary to {args.out_md}")


if __name__ == "__main__":
    main()
