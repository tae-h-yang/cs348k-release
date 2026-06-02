"""Evaluate the 100-prompt target suite against the local MotionBricks preview.

This is deliberately an audit, not a claim that all 100 prompts are executable.
For prompts with a current proxy mode, the script links K=1 and K=8 clips and
can render side-by-side before/after videos. With ``--force_proxy_all``, every
prompt gets a before/after proxy video using the nearest exposed MotionBricks
mode, while semantic validity is marked explicitly.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_CSV = ROOT / "configs" / "humanoid_robotics_100_prompts.csv"
GUIDED_CSV = ROOT / "results" / "guided_ablation_extended.csv"
DATA_DIR = ROOT / "data" / "guided_ablation_extended"
MODEL_XML = ROOT / "assets" / "g1" / "scene_29dof.xml"
OUT_DIR = ROOT / "results" / "humanoid100_eval"

WIDTH = 480
HEIGHT = 360
FPS = 30


MODE_TYPES = {
    "idle": "static",
    "walk": "locomotion",
    "slow_walk": "locomotion",
    "stealth_walk": "locomotion",
    "injured_walk": "locomotion",
    "walk_left": "directional_locomotion",
    "walk_right": "directional_locomotion",
    "walk_stealth": "style_locomotion",
    "walk_zombie": "style_locomotion",
    "walk_boxing": "expressive_locomotion",
    "walk_happy_dance": "expressive_locomotion",
    "walk_gun": "expressive_locomotion",
    "walk_scared": "expressive_locomotion",
    "hand_crawling": "whole_body_low",
    "elbow_crawling": "whole_body_low",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_metrics(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    rows = read_rows(path)
    return {
        (row["mode"], int(row["seed_idx"]), int(row["K"])): row
        for row in rows
        if row.get("mode") in MODE_TYPES and row.get("seed_idx") and row.get("K")
    }


def proxy_mode_for(row: dict[str, str]) -> tuple[str, str, str]:
    """Return mode, executable_status, and explanation."""
    support = row["current_motionbricks_support"]
    hint = row["motionbricks_mode_hint"]
    sub = row["subcategory"]
    category = row["category"]

    if support == "__NO__":
        return "", "unsupported", "No executable local MotionBricks G1 mode."
    if hint in MODE_TYPES:
        return hint, "mode_proxy", f"Uses exposed local mode `{hint}` as proxy."
    if hint == "hand_crawling_or_elbow_crawling":
        mode = "elbow_crawling" if "elbow" in sub else "hand_crawling"
        return mode, "partial_proxy", f"Uses `{mode}` for low-floor crawl proxy."
    if hint == "walk_or_idle_mode":
        mode = "idle" if category == "communication_safety" else "walk"
        return mode, "partial_proxy", f"Uses `{mode}` as a coarse proxy only."
    return "", "unsupported", f"No local mapping for hint `{hint}`."


def forced_proxy_mode_for(row: dict[str, str]) -> tuple[str, str]:
    """Map every target prompt to an exposed local mode for full-suite stress tests.

    These mappings are intentionally coarse and should not be interpreted as
    semantic prompt-following success. They exist to run a complete 100-row
    before/after physics-screening experiment with the current limited preview.
    """
    category = row["category"]
    sub = row["subcategory"]

    table = {
        "vertical_jump": "walk_happy_dance",
        "broad_jump": "walk",
        "one_leg_hop_right": "walk_right",
        "one_leg_hop_left": "walk_left",
        "skip_forward": "walk_happy_dance",
        "forward_lunge": "walk",
        "quick_stop": "injured_walk",
        "moonwalk": "walk_stealth",
        "robot_dance": "idle",
        "tai_chi_sweep": "slow_walk",
        "celebration_pump": "walk_happy_dance",
        "disco_point": "walk_happy_dance",
        "air_guitar": "walk_gun",
        "salute_step": "walk_gun",
        "crab_walk": "hand_crawling",
        "duck_walk": "walk_stealth",
        "kneel_to_stand": "hand_crawling",
        "stand_to_kneel": "hand_crawling",
        "pushup_pose": "elbow_crawling",
        "sit_to_stand": "hand_crawling",
        "roll_to_kneel": "elbow_crawling",
        "low_side_step": "walk_right",
        "inspect_floor": "walk_stealth",
        "single_leg_balance_right": "walk_right",
        "single_leg_balance_left": "walk_left",
        "stumble_forward": "injured_walk",
        "stumble_left": "walk_left",
        "stumble_right": "walk_right",
        "backward_recovery": "walk_stealth",
        "ankle_sway": "idle",
        "hip_strategy": "idle",
        "toe_stand": "idle",
        "heel_rock": "idle",
        "catch_balance_arms": "idle",
        "narrow_stance_hold": "idle",
        "stop_signal": "walk_gun",
        "step_over_right": "walk_right",
        "step_over_left": "walk_left",
        "high_step_cables": "walk",
        "duck_under_bar": "walk_stealth",
        "slope_up": "slow_walk",
        "slope_down": "slow_walk",
        "swerve_left": "walk_left",
        "tight_turn_back": "walk_stealth",
        "cartwheel_attempt": "hand_crawling",
        "forward_roll": "elbow_crawling",
        "burpee": "hand_crawling",
        "sprawl_recovery": "elbow_crawling",
        "split_squat_jump": "walk_happy_dance",
        "knee_slide": "elbow_crawling",
        "side_roll_recovery": "elbow_crawling",
        "handstand_kickup": "hand_crawling",
    }
    if sub in table:
        return table[sub], "forced nearest-mode proxy; semantic prompt-following is not satisfied"
    if category == "manipulation_stance":
        return "walk_gun", "forced upper-body/action proxy; no object semantics"
    if category == "loco_manipulation":
        return "walk_gun", "forced locomotion-plus-arm proxy; no object semantics"
    if category == "terrain_obstacle":
        return "walk", "forced locomotion proxy; no terrain/object geometry"
    return "walk", "forced generic locomotion proxy"


def camera_for(qpos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = np.array([qpos[0], qpos[1], 0.76], dtype=np.float64)
    cam.azimuth = 90.0
    cam.elevation = -15.0
    cam.distance = 3.5
    return cam


def put(frame: np.ndarray, text: str, x: int, y: int, scale: float = 0.44,
        color: tuple[int, int, int] = (245, 245, 245)) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if len(trial) > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def render_qpos_frame(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    title: str,
    prompt_id: str,
    risk: str,
    t: int,
) -> np.ndarray:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_for(qpos))
    frame = renderer.render().copy()
    cv2.rectangle(frame, (0, 0), (WIDTH, 58), (0, 0, 0), -1)
    put(frame, title, 10, 20, 0.48)
    put(frame, f"{prompt_id} | risk {risk} | t={t / FPS:.2f}s", 10, 44, 0.40, (210, 230, 255))
    return frame


def render_proxy_video(row: dict[str, object], out_path: Path) -> None:
    q1 = np.load(str(row["before_qpos_path"]))
    q8 = np.load(str(row["after_qpos_path"]))
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data1 = mujoco.MjData(model)
    data8 = mujoco.MjData(model)
    r1 = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    r8 = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    divider = np.full((HEIGHT, 6, 3), 32, dtype=np.uint8)

    frames: list[np.ndarray] = []
    T = min(len(q1), len(q8))
    for t in range(T):
        left = render_qpos_frame(
            r1, model, data1, q1[t], "BEFORE: K=1 baseline",
            str(row["prompt_id"]), f"{float(row['before_risk']):.1f}", t,
        )
        right = render_qpos_frame(
            r8, model, data8, q8[t], "AFTER: selected K=8",
            str(row["prompt_id"]), f"{float(row['after_risk']):.1f}", t,
        )
        frame = np.concatenate([left, divider, right], axis=1)
        cv2.rectangle(frame, (0, HEIGHT - 72), (frame.shape[1], HEIGHT), (0, 0, 0), -1)
        prompt_lines = wrap_text(str(row["prompt_text"]), 100)[:2]
        put(
            frame,
            f"{row['executable_status']} | semantic: {row['semantic_validity']} | proxy: {row['proxy_mode']}",
            10,
            HEIGHT - 46,
            0.40,
            (220, 255, 220),
        )
        put(frame, " ".join(prompt_lines), 10, HEIGHT - 20, 0.40, (245, 245, 245))
        frames.append(frame)

    r1.close()
    r8.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=FPS, macro_block_size=1)


def render_unsupported_video(row: dict[str, object], out_path: Path) -> None:
    frames: list[np.ndarray] = []
    w, h = WIDTH * 2 + 6, HEIGHT
    for _ in range(FPS * 3):
        frame = np.full((h, w, 3), 24, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (w, 76), (60, 20, 20), -1)
        put(frame, f"{row['prompt_id']} | unsupported by local MotionBricks preview", 18, 28, 0.62, (255, 245, 245))
        put(frame, "No before/after motion rendered; this is a target benchmark prompt.", 18, 58, 0.48, (230, 210, 210))
        y = 118
        for line in wrap_text(str(row["prompt_text"]), 92)[:4]:
            put(frame, line, 28, y, 0.50, (245, 245, 245))
            y += 30
        y += 12
        put(frame, f"Category: {row['category']} | Subcategory: {row['subcategory']}", 28, y, 0.46, (220, 230, 255))
        put(frame, f"Success criteria: {row['success_criteria']}", 28, y + 30, 0.42, (210, 210, 210))
        put(frame, "Required next step: richer generator/retargeter, then physics + visual audit.", 28, h - 38, 0.46, (255, 220, 150))
        frames.append(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=FPS, macro_block_size=1)


def build_eval_rows(
    seed: int,
    metrics: dict[tuple[str, int, int], dict[str, str]],
    force_proxy_all: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prompt in read_rows(PROMPTS_CSV):
        mode, status, explanation = proxy_mode_for(prompt)
        semantic_validity = "supported_proxy" if status != "unsupported" else "unsupported"
        if force_proxy_all and not mode:
            mode, forced_explanation = forced_proxy_mode_for(prompt)
            status = "forced_proxy"
            explanation = forced_explanation
            semantic_validity = "forced_proxy_not_prompt_following"
        before = metrics.get((mode, seed, 1)) if mode else None
        after = metrics.get((mode, seed, 8)) if mode else None
        render_status = "unsupported_placeholder"
        before_path = ""
        after_path = ""
        before_risk = ""
        after_risk = ""
        improvement_pct = ""

        if before and after:
            before_path = before["path"]
            after_path = after["path"]
            before_risk = float(before["full_risk"])
            after_risk = float(after["full_risk"])
            if before_risk:
                improvement_pct = 100.0 * (before_risk - after_risk) / abs(before_risk)
            render_status = "proxy_before_after"
        elif status != "unsupported":
            render_status = "missing_proxy_data"

        video = f"videos/{prompt['prompt_id']}_{prompt['subcategory']}.mp4"
        rows.append({
            "prompt_id": prompt["prompt_id"],
            "category": prompt["category"],
            "subcategory": prompt["subcategory"],
            "prompt_text": prompt["prompt_text"],
            "success_criteria": prompt["success_criteria"],
            "current_motionbricks_support": prompt["current_motionbricks_support"],
            "motionbricks_mode_hint": prompt["motionbricks_mode_hint"],
            "proxy_mode": mode,
            "executable_status": status,
            "semantic_validity": semantic_validity,
            "audit_interpretation": explanation,
            "before_K": 1 if before else "",
            "after_K": 8 if after else "",
            "before_risk": before_risk,
            "after_risk": after_risk,
            "risk_improvement_pct": improvement_pct,
            "before_qpos_path": before_path,
            "after_qpos_path": after_path,
            "render_status": render_status,
            "video_path": str(OUT_DIR / video),
        })
    return rows


def write_report(path: Path, rows: list[dict[str, object]], force_proxy_all: bool) -> None:
    counts = Counter(str(row["render_status"]) for row in rows)
    support_counts = Counter(str(row["current_motionbricks_support"]) for row in rows)
    semantic_counts = Counter(str(row["semantic_validity"]) for row in rows)
    lines = [
        "# Humanoid Robotics 100-Prompt Audit",
        "",
        "This report evaluates every target prompt against the currently exposed",
        "local MotionBricks G1 preview.",
        "",
        "## Summary",
        "",
    ]
    if force_proxy_all:
        lines.extend([
            "Mode: full forced-proxy experiment. Every prompt has an actual",
            "K=1-vs-K=8 before/after video, but forced proxies are not semantic",
            "prompt-following successes.",
            "",
        ])
    else:
        lines.extend([
            "Mode: support audit. Proxy rows use actual K=1 vs K=8 qpos clips;",
            "unsupported rows are explicit non-executable placeholders.",
            "",
        ])
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Current MotionBricks Support Labels", ""])
    for key, value in sorted(support_counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Semantic Validity", ""])
    for key, value in sorted(semantic_counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `proxy_before_after` means the video uses an exposed local mode as a proxy.",
        "- `forced_proxy_not_prompt_following` means the video is useful for",
        "  physics-screening mechanics but does not satisfy the natural-language",
        "  prompt semantics.",
        "- `unsupported_placeholder` means no current local MotionBricks mode can",
        "  honestly execute that prompt.",
        "- Do not present proxy videos as solved prompt-following examples; present",
        "  them as coverage limits plus before/after physics-screening examples.",
        "",
        "## Outputs",
        "",
        f"- CSV: `{OUT_DIR / 'humanoid100_eval.csv'}`",
        f"- Videos: `{OUT_DIR / 'videos'}`",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    global OUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--force_proxy_all", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    OUT_DIR = args.out_dir

    metrics = load_metrics(GUIDED_CSV)
    rows = build_eval_rows(args.seed, metrics, force_proxy_all=args.force_proxy_all)[: args.limit]
    write_rows(OUT_DIR / "humanoid100_eval.csv", rows)
    write_report(OUT_DIR / "README.md", rows, args.force_proxy_all)

    if args.render:
        for row in rows:
            out = Path(str(row["video_path"]))
            if row["render_status"] == "proxy_before_after":
                render_proxy_video(row, out)
            else:
                render_unsupported_video(row, out)
            print(f"Wrote {out}")

    counts = Counter(str(row["render_status"]) for row in rows)
    print(f"Wrote {len(rows)} prompt audit rows to {OUT_DIR / 'humanoid100_eval.csv'}")
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
