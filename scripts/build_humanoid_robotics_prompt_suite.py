"""Build a diverse 100-prompt humanoid robotics motion benchmark.

This target suite is intentionally different from ``prompt_suite_105.csv``.
The older file expands currently exposed MotionBricks G1 modes by seed; this
file defines 100 distinct humanoid motion intents that a mature generator
should support. The suite deliberately includes athletic, dance, low-posture,
floor-contact, manipulation, balance, and safety motions so the project cannot
hide behind many walking-style variants.

The current local MotionBricks preview cannot execute all prompts directly. The
``current_motionbricks_support`` and ``motionbricks_mode_hint`` columns make
that limitation explicit instead of pretending unsupported behaviors were
validated.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "configs" / "humanoid_robotics_100_prompts.csv"
OUT_MD = ROOT / "docs" / "autonomous_loop" / "humanoid_robotics_100_prompts.md"


Prompt = dict[str, str]


def p(category: str, subcategory: str, prompt: str, criteria: str) -> Prompt:
    return {
        "category": category,
        "subcategory": subcategory,
        "prompt": prompt,
        "criteria": criteria,
    }


PROMPTS: list[Prompt] = [
    # Dynamic locomotion, hops, and transitions, 14
    p("dynamic_locomotion", "forward_walk", "Walk forward at a comfortable indoor pace with symmetric arm swing.", "forward progress; upright torso; alternating foot contacts"),
    p("dynamic_locomotion", "backward_walk", "Walk backward for several steps while keeping the chest facing forward.", "backward displacement; upright torso; no spin"),
    p("dynamic_locomotion", "side_shuffle_left", "Shuffle two meters to the robot's left with quick lateral steps.", "leftward displacement; lateral cadence; feet remain under support"),
    p("dynamic_locomotion", "side_shuffle_right", "Shuffle two meters to the robot's right with quick lateral steps.", "rightward displacement; lateral cadence; feet remain under support"),
    p("dynamic_locomotion", "turn_in_place", "Turn in place one hundred eighty degrees and stop facing the opposite direction.", "large yaw change; low translation; balanced final stance"),
    p("dynamic_locomotion", "march_high_knees", "March in place with clear alternating high-knee lifts.", "low translation; alternating knee lift; regular rhythm"),
    p("dynamic_locomotion", "vertical_jump", "Perform a small vertical jump and land with both feet at the starting spot.", "flight phase; two-foot takeoff and landing; stable recovery"),
    p("dynamic_locomotion", "broad_jump", "Jump forward a short distance and land in a balanced two-foot stance.", "forward aerial displacement; controlled landing; no hand contact"),
    p("dynamic_locomotion", "one_leg_hop_right", "Hop in place three times on the right leg while the left knee stays lifted.", "right foot support; repeated flight; stable trunk"),
    p("dynamic_locomotion", "one_leg_hop_left", "Hop in place three times on the left leg while the right knee stays lifted.", "left foot support; repeated flight; stable trunk"),
    p("dynamic_locomotion", "skip_forward", "Skip forward with alternating light hops and relaxed arm swing.", "alternating hop gait; forward progress; rhythmic contacts"),
    p("dynamic_locomotion", "lateral_bound", "Make two lateral bounding steps to the left and recover to a stable stance.", "leftward bounds; brief flight; controlled stop"),
    p("dynamic_locomotion", "forward_lunge", "Step into a deep forward lunge and push back to standing.", "one long step; knee bend; return upright"),
    p("dynamic_locomotion", "quick_stop", "Jog two short steps forward and stop abruptly without overbalancing.", "brief fast progress; rapid deceleration; stable double support"),

    # Dance and expressive whole-body motion, 12
    p("dance_expressive", "hip_hop_heel_toe", "Perform C-walk-inspired hip-hop heel-toe footwork in place without traveling far.", "alternating heel-toe pivots; low displacement; upright rhythm"),
    p("dance_expressive", "moonwalk", "Perform a short moonwalk illusion moving backward while facing forward.", "backward glide; feet slide in pattern; torso faces forward"),
    p("dance_expressive", "robot_dance", "Stand in place and perform stiff robot-dance arm and torso isolations.", "segmented arm motion; planted feet; low root drift"),
    p("dance_expressive", "happy_dance", "Do an energetic happy dance with bouncing knees and wide arm gestures.", "rhythmic bounce; expressive arms; no collapse"),
    p("dance_expressive", "boxing_shadow", "Shadowbox in place with alternating jabs and guarded footwork.", "alternating punches; small stance shifts; upright guard"),
    p("dance_expressive", "tai_chi_sweep", "Perform a slow tai-chi-style weight shift with both arms sweeping outward.", "slow COM shift; smooth arms; planted or small steps"),
    p("dance_expressive", "celebration_pump", "Pump both fists overhead twice while bouncing lightly on the feet.", "overhead arm motion; rhythmic knee bend; stable stance"),
    p("dance_expressive", "disco_point", "Step side to side while pointing one hand diagonally upward in a disco pose.", "side steps; arm point alternates; torso remains upright"),
    p("dance_expressive", "zombie_walk", "Walk forward in a zombie-like style with extended arms and stiff knees.", "forward progress; arms extended; stylized stiff gait"),
    p("dance_expressive", "scared_sneak", "Sneak forward nervously with guarded arms and short cautious steps.", "short steps; guarded upper body; upright recovery"),
    p("dance_expressive", "air_guitar", "Stand with a wide stance and mime playing an air guitar.", "wide support; cyclic arm motion; low root drift"),
    p("dance_expressive", "salute_step", "Take one step forward and perform a crisp right-hand salute.", "single step; right hand to head; final pause"),

    # Low posture and floor-contact behavior, 12
    p("floor_low_posture", "hand_crawl", "Crawl forward using hands and feet in a controlled low posture.", "hand-floor contact; low root; forward progress"),
    p("floor_low_posture", "elbow_crawl", "Crawl forward on elbows with the torso close to the ground.", "elbow/forearm contact; very low root; forward progress"),
    p("floor_low_posture", "bear_crawl", "Bear crawl forward with hips high and hands and feet on the floor.", "hands and feet contact; alternating limbs; forward progress"),
    p("floor_low_posture", "crab_walk", "Crab-walk backward with the chest facing upward and hands behind the body.", "hands and feet contact; backward progress; torso supine-ish"),
    p("floor_low_posture", "duck_walk", "Move forward in a deep squat duck-walk for several steps.", "very low root; alternating feet; no hand-floor support"),
    p("floor_low_posture", "kneel_to_stand", "Start kneeling and rise to a stable standing posture.", "root rises; knee contact early; final upright"),
    p("floor_low_posture", "stand_to_kneel", "Lower from standing into a kneeling posture without falling.", "root lowers; controlled knee contact; final kneel"),
    p("floor_low_posture", "pushup_pose", "Lower into a push-up plank and return to standing.", "hands contact floor; straight-body plank; root recovers"),
    p("floor_low_posture", "sit_to_stand", "Rise from a seated floor posture into a stable stand.", "low initial root; hands or feet assist; final upright"),
    p("floor_low_posture", "roll_to_kneel", "Perform a safe floor roll onto one side and recover to kneeling.", "side roll; low root; controlled kneel recovery"),
    p("floor_low_posture", "low_side_step", "Perform two low sideways squat steps to the right.", "low root; rightward displacement; stable knees"),
    p("floor_low_posture", "inspect_floor", "Drop into a low squat, reach toward the floor, and stand again.", "squat; hand near floor; full recovery"),

    # Manipulation while standing, 12
    p("manipulation_stance", "pick_floor_right", "Bend at the knees to pick up a small object from the floor with the right hand.", "root lowers; right hand reaches floor; recover upright"),
    p("manipulation_stance", "pick_floor_left", "Bend at the knees to pick up a small object from the floor with the left hand.", "root lowers; left hand reaches floor; recover upright"),
    p("manipulation_stance", "reach_overhead", "Reach overhead with both hands as if placing an item on a high shelf.", "both hands high; heels mostly planted; controlled torso"),
    p("manipulation_stance", "reach_low_forward", "Reach down and forward with both hands without moving the feet.", "stationary feet; arms forward-low; stable knees"),
    p("manipulation_stance", "wipe_table", "Stand in place and wipe a horizontal table surface with the right hand.", "feet stationary; cyclic right arm sweep; stable trunk"),
    p("manipulation_stance", "hammer_down", "Stand with feet planted and swing the right hand downward as if using a hammer.", "right arm downstroke; torso controlled; feet stable"),
    p("manipulation_stance", "screwdriver_twist", "Hold both hands near the chest and make a two-handed twisting screwdriver motion.", "hands near midline; cyclic twist; low root drift"),
    p("manipulation_stance", "press_panel_sequence", "Press three imaginary buttons from left to right on a waist-high panel.", "three distinct reaches; left-to-right order; feet stable"),
    p("manipulation_stance", "sort_bins", "Shift weight left and right while moving imaginary objects between waist-high bins.", "lateral weight shifts; alternating hand motions; stable feet"),
    p("manipulation_stance", "lean_left_reach", "Lean left and reach the left hand outward while keeping both feet planted.", "left reach; COM remains supported; no step"),
    p("manipulation_stance", "lean_right_reach", "Lean right and reach the right hand outward while keeping both feet planted.", "right reach; COM remains supported; no step"),
    p("manipulation_stance", "scan_package", "Stand at a table and move both hands around a package as if scanning it.", "two-hand motions; hands near table height; stable feet"),

    # Loco-manipulation and workspace tasks, 14
    p("loco_manipulation", "carry_box_front", "Walk forward while carrying a small box with both hands at chest height.", "hands held near chest; forward gait; reduced arm swing"),
    p("loco_manipulation", "carry_bag_right", "Walk forward while carrying a bag low in the right hand.", "right hand low and steady; forward gait; asymmetric arm motion"),
    p("loco_manipulation", "tray_walk", "Walk carefully while keeping both hands level as if holding a tray.", "hands level; smooth trunk; slow forward progress"),
    p("loco_manipulation", "push_cart", "Walk forward with both hands extended as if pushing a cart.", "hands forward; forward gait; stable torso"),
    p("loco_manipulation", "pull_cart_backward", "Walk backward with both hands forward as if pulling a cart toward the robot.", "backward displacement; arms forward; no spin"),
    p("loco_manipulation", "open_door", "Step forward, reach with the right hand as if opening a door, then pass through.", "right arm reach; step-through; forward continuation"),
    p("loco_manipulation", "close_door", "Turn slightly, pull the right hand back as if closing a door, then stand balanced.", "right arm pull; torso turn; final stable stance"),
    p("loco_manipulation", "handoff_give", "Step forward and extend both hands as if handing over an object.", "forward step; both hands extend; stable end pose"),
    p("loco_manipulation", "handoff_receive", "Step forward and present both hands to receive an object.", "forward step; both hands open forward; final pause"),
    p("loco_manipulation", "drag_object", "Lean back slightly and pull an imaginary heavy object with both hands.", "hands pull backward; backward lean; feet remain stable"),
    p("loco_manipulation", "carry_turn", "Carry an imaginary box, turn left, and continue walking.", "hands held; left yaw; forward continuation"),
    p("loco_manipulation", "inspect_machine", "Walk to a machine, lean in slightly, and inspect it with hands behind the back.", "approach; forward lean; arms held back"),
    p("loco_manipulation", "open_drawer", "Step forward, reach low with both hands, and pull as if opening a drawer.", "approach; hands low-forward; backward pull"),
    p("loco_manipulation", "loaded_recovery_step", "Walk forward carrying a load and take a corrective step after a small imbalance.", "arms held load; perturbation recovery; final upright"),

    # Balance and recovery, 12
    p("balance_recovery", "single_leg_balance_right", "Balance briefly on the right leg while lifting the left knee.", "right support; left knee lift; no fall"),
    p("balance_recovery", "single_leg_balance_left", "Balance briefly on the left leg while lifting the right knee.", "left support; right knee lift; no fall"),
    p("balance_recovery", "stumble_forward", "Recover from a small forward stumble with one quick corrective step.", "forward pitch; recovery step; final upright"),
    p("balance_recovery", "stumble_left", "Recover from a small leftward stumble with a side step.", "left lean; lateral recovery step; final upright"),
    p("balance_recovery", "stumble_right", "Recover from a small rightward stumble with a side step.", "right lean; lateral recovery step; final upright"),
    p("balance_recovery", "backward_recovery", "Recover from a slight backward lean by stepping back.", "backward lean; backward step; final upright"),
    p("balance_recovery", "ankle_sway", "Stand still and sway gently forward and backward without stepping.", "small root sway; feet planted; no arm flail"),
    p("balance_recovery", "hip_strategy", "Stand still and make a larger torso balance correction without moving the feet.", "torso sway; feet planted; recover center"),
    p("balance_recovery", "toe_stand", "Rise briefly onto the toes and return to flat feet.", "heel lift; low translation; controlled return"),
    p("balance_recovery", "heel_rock", "Rock back briefly onto the heels and return to normal standing.", "toe lift; low translation; controlled return"),
    p("balance_recovery", "catch_balance_arms", "Lift both arms outward to regain balance after a small perturbation.", "arms abduct; trunk returns upright; feet stable or one step"),
    p("balance_recovery", "narrow_stance_hold", "Hold a narrow stance while making small balance corrections with the arms.", "narrow support; bounded sway; no stepping"),

    # Safety, communication, and social cues, 8
    p("communication_safety", "wave", "Stand still and wave the right hand at shoulder height.", "feet stationary; right arm wave; upright torso"),
    p("communication_safety", "stop_signal", "Take one step forward and raise the right palm in a stop signal.", "one step; right hand forward-high; final pause"),
    p("communication_safety", "point_left", "Point clearly to the robot's left with the left arm while standing.", "left arm lateral extension; feet stable; low torso drift"),
    p("communication_safety", "point_right", "Point clearly to the robot's right with the right arm while standing.", "right arm lateral extension; feet stable; low torso drift"),
    p("communication_safety", "beckon", "Stand and make a beckoning motion with the right hand.", "right hand cyclic pull; feet stable; upright"),
    p("communication_safety", "yield_step", "Step backward and raise both hands slightly as a yielding gesture.", "backward step; hands lift; final stable stance"),
    p("communication_safety", "look_around", "Stand still and look left, right, then forward.", "yaw sequence; feet planted; no arm requirement"),
    p("communication_safety", "direct_traffic", "Stand and sweep the right arm sideways as if directing someone to pass.", "right arm lateral sweep; feet stable; torso controlled"),

    # Obstacles and terrain proxies, 8
    p("terrain_obstacle", "step_over_right", "Step over a low imaginary obstacle with the right foot first.", "right leg leads; elevated swing foot; no foot drag"),
    p("terrain_obstacle", "step_over_left", "Step over a low imaginary obstacle with the left foot first.", "left leg leads; elevated swing foot; no foot drag"),
    p("terrain_obstacle", "high_step_cables", "Walk forward using high steps as if crossing scattered cables.", "repeated foot clearance; forward progress; no stumble"),
    p("terrain_obstacle", "duck_under_bar", "Crouch slightly while walking under a low bar, then return upright.", "temporary lower root; forward progress; recovery upright"),
    p("terrain_obstacle", "slope_up", "Walk forward as if climbing a shallow ramp with a slight forward lean.", "forward progress; mild torso pitch; stable cadence"),
    p("terrain_obstacle", "slope_down", "Walk forward as if descending a shallow ramp with cautious short steps.", "forward progress; lower speed; stable trunk"),
    p("terrain_obstacle", "swerve_left", "Walk forward and swerve gently left around an obstacle.", "forward then lateral path change; no abrupt jump"),
    p("terrain_obstacle", "tight_turn_back", "Walk forward, make a compact one-hundred-eighty-degree turn, and walk back.", "forward progress; 180 yaw change; return path"),

    # Acrobatics and high-risk stress tests, 8
    p("athletic_stress", "cartwheel_attempt", "Attempt a slow cartwheel-like lateral inversion and recover to standing.", "hands may contact floor; large roll; recover upright"),
    p("athletic_stress", "forward_roll", "Perform a controlled forward roll and return to a kneeling posture.", "floor roll; low root; no uncontrolled fling"),
    p("athletic_stress", "burpee", "Squat down, kick the legs back to a plank, return the feet under the body, and stand.", "squat; plank; stand recovery"),
    p("athletic_stress", "sprawl_recovery", "Drop quickly into a sprawl and recover to an athletic stance.", "rapid root drop; hands/feet contact; stable recovery"),
    p("athletic_stress", "split_squat_jump", "Perform a small split-squat jump switching which foot is forward.", "brief flight; leg switch; controlled landing"),
    p("athletic_stress", "knee_slide", "Slide forward briefly on both knees and recover to kneeling.", "knee contact; forward slide; controlled stop"),
    p("athletic_stress", "side_roll_recovery", "Roll sideways on the floor and recover to a crouched stance.", "side roll; low posture; crouch recovery"),
    p("athletic_stress", "handstand_kickup", "Kick up toward a brief handstand and return the feet to the floor.", "hand contact; inverted posture; controlled return"),
]


SUPPORTED_STYLE_MODES = {
    "boxing_shadow": "walk_boxing",
    "happy_dance": "walk_happy_dance",
    "zombie_walk": "walk_zombie",
    "scared_sneak": "walk_scared",
    "hip_hop_heel_toe": "walk_happy_dance",
}


def has_words(text: str, words: list[str]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def support_label(row: Prompt) -> tuple[str, str]:
    category = row["category"]
    subcategory = row["subcategory"]
    text = row["prompt"].lower()
    if subcategory in SUPPORTED_STYLE_MODES:
        return "__YES_MODE_PROXY__", SUPPORTED_STYLE_MODES[subcategory]
    if category == "floor_low_posture" and "crawl" in text:
        return "__PARTIAL__", "hand_crawling_or_elbow_crawling"
    if category == "dynamic_locomotion" and "left" in text and ("shuffle" in text or "lateral" in text):
        return "__YES_MODE_PROXY__", "walk_left"
    if category == "dynamic_locomotion" and "right" in text and "shuffle" in text:
        return "__YES_MODE_PROXY__", "walk_right"
    if category in {"dynamic_locomotion", "communication_safety"} and not any(
        token in subcategory for token in ["jump", "hop", "skip", "bound", "lunge", "stop"]
    ):
        return "__PARTIAL__", "walk_or_idle_mode"
    if category in {"manipulation_stance", "loco_manipulation", "terrain_obstacle"}:
        return "__NO__", "requires_task_conditioned_generator_or_retargeter"
    if category == "athletic_stress":
        return "__NO__", "negative_control_high_risk"
    return "__NO__", "requires_new_generator_control"


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, row in enumerate(PROMPTS, start=1):
        prompt = row["prompt"].strip()
        key = prompt.lower()
        if key in seen:
            raise ValueError(f"Duplicate prompt text: {prompt}")
        seen.add(key)
        support, mode_hint = support_label(row)
        rows.append({
            "prompt_id": f"hrb_{idx:03d}",
            "category": row["category"],
            "subcategory": row["subcategory"],
            "prompt_text": prompt,
            "success_criteria": row["criteria"],
            "expected_primary_contacts": infer_contacts(row),
            "expected_root_motion": infer_root_motion(row),
            "expected_arm_role": infer_arm_role(row),
            "hardness": infer_hardness(row, support),
            "current_motionbricks_support": support,
            "motionbricks_mode_hint": mode_hint,
            "evaluation_notes": evaluation_notes(row, support),
        })
    if len(rows) != 100:
        raise ValueError(f"Expected 100 prompts, got {len(rows)}")
    return rows


def infer_contacts(row: Prompt) -> str:
    text = row["prompt"].lower()
    if "one_leg" in row["subcategory"] or "single_leg" in row["subcategory"]:
        return "one_foot"
    if has_words(text, ["crawl", "plank", "push-up", "pushup", "handstand", "cartwheel"]):
        return "feet;hands_or_elbows"
    if has_words(text, ["kneel", "kneeling", "knee", "knees"]):
        return "feet;knee"
    if "roll" in text:
        return "body_floor"
    return "feet"


def infer_root_motion(row: Prompt) -> str:
    text = row["prompt"].lower()
    if has_words(text, ["jump", "hop", "bound", "flight"]):
        return "aerial_or_impulsive"
    if has_words(text, ["kneel", "crawl", "squat", "floor", "plank", "sprawl"]):
        return "low_transition"
    if any(phrase in text for phrase in ["in place", "without moving the feet", "feet planted"]) or has_words(text, ["stand", "standing"]):
        return "mostly_stationary"
    if has_words(text, ["backward", "moonwalk"]) or "step backward" in text:
        return "backward"
    if "left" in text and has_words(text, ["shuffle", "swerve", "lateral", "lean"]):
        return "leftward"
    if "right" in text and has_words(text, ["shuffle", "swerve", "lateral", "lean"]):
        return "rightward"
    if has_words(text, ["turn", "cartwheel", "roll"]):
        return "turning_or_floor_transition"
    return "forward_or_task_dependent"


def infer_arm_role(row: Prompt) -> str:
    text = row["prompt"].lower()
    if has_words(text, ["carry", "hands", "reach", "press", "push", "pull", "wipe", "point", "wave", "button", "object", "box", "tray", "panel", "table"]):
        return "task_constrained"
    if has_words(text, ["dance", "boxing", "jab", "zombie", "celebration", "guitar", "disco", "salute"]):
        return "expressive"
    if has_words(text, ["cartwheel", "handstand", "roll", "plank"]):
        return "support_or_recovery"
    return "natural_or_unspecified"


def infer_hardness(row: Prompt, support: str) -> str:
    if row["category"] == "athletic_stress":
        return "extreme_negative_control"
    if support == "__NO__":
        return "high"
    if row["category"] in {"floor_low_posture", "dynamic_locomotion", "terrain_obstacle", "loco_manipulation"}:
        return "medium_high"
    return "medium"


def evaluation_notes(row: Prompt, support: str) -> str:
    if row["category"] == "athletic_stress":
        return "Stress/negative-control prompt: evaluate as unsupported unless a controller can execute it without unsafe impacts."
    if support == "__NO__":
        return "Target prompt for future methods; current MotionBricks preview needs task-conditioned generation or retargeting."
    return "Use MotionSpec predicates plus physics/contact/controller/visual audit."


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    support_counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
        support_counts[row["current_motionbricks_support"]] = support_counts.get(row["current_motionbricks_support"], 0) + 1

    lines = [
        "# Humanoid Robotics 100-Prompt Benchmark",
        "",
        "This is the target benchmark for the pivoted project. It contains 100",
        "distinct humanoid robotics motion intents, not seed duplicates. The suite",
        "was refactored to avoid a walking-only benchmark: it now includes jumps,",
        "one-leg hops, hip-hop-style footwork, crawling, floor transitions,",
        "manipulation, balance recovery, obstacle proxies, and high-risk athletic",
        "negative controls.",
        "",
        "The current local MotionBricks preview only supports a subset through",
        "discrete G1 modes. Unsupported rows are still useful: they define the",
        "target behavior space for evaluating future prompt steering, retargeting,",
        "or learned physically-aware generation.",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in sorted(counts.items()):
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Current Generator Support", ""])
    for support, count in sorted(support_counts.items()):
        lines.append(f"- `{support}`: {count}")
    lines.extend([
        "",
        "## Required Evaluation Layers",
        "",
        "- Prompt/task checks: direction, speed, posture, contacts, arm role, and event order from `success_criteria`.",
        "- Kinematic checks: finite qpos, joint limits, root height, foot skate, self-contact, non-foot floor contact.",
        "- Dynamics checks: inverse dynamics torque demand, unactuated root wrench, velocity, acceleration, and jerk.",
        "- Controller checks: SONIC or another learned G1 tracker for survival time, tracking RMSE, falls, and effort.",
        "- Visual audit: rendered clips/contact sheets for semantic match and obvious artifacts.",
        "",
        "## Full Prompt List",
        "",
        "| ID | Category | Subcategory | Prompt | Support |",
        "|---|---|---|---|---|",
    ])
    for row in rows:
        lines.append(
            f"| `{row['prompt_id']}` | `{row['category']}` | `{row['subcategory']}` | "
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
    print(f"Wrote {len(rows)} distinct prompts to {args.out_csv}")
    print(f"Wrote benchmark notes to {args.out_md}")


if __name__ == "__main__":
    main()
