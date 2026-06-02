from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.build_humanoid_robotics_prompt_suite import build_rows
from scripts.build_sports_acrobatics_prompt_suite import (
    PARTIAL_MODE_HINTS,
    build_rows as build_sports_rows,
)
from scripts.evaluate_humanoid_100_prompts import build_eval_rows, load_metrics
from scripts.repair_humanoid100_references import qpos_slerp, repair_variants, validate_qpos


GUIDED_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "guided_ablation_extended.csv"


def test_humanoid_robotics_suite_is_diverse() -> None:
    rows = build_rows()
    assert len(rows) == 100
    assert len({row["prompt_text"].lower() for row in rows}) == 100

    categories = Counter(row["category"] for row in rows)
    assert len(categories) >= 8
    assert categories["dynamic_locomotion"] >= 10
    assert categories["dance_expressive"] >= 10
    assert categories["floor_low_posture"] >= 10
    assert categories["athletic_stress"] >= 8

    prompts = [row["prompt_text"].lower() for row in rows]
    assert sum("walk" in prompt for prompt in prompts) <= 25
    assert sum(any(token in prompt for token in ["jump", "hop", "skip", "bound"]) for prompt in prompts) >= 8
    assert sum(any(token in prompt for token in ["crawl", "floor", "roll", "kneel", "plank", "handstand", "cartwheel"]) for prompt in prompts) >= 17


def test_prompt_suite_labeling_avoids_substring_artifacts() -> None:
    rows = {row["subcategory"]: row for row in build_rows()}

    assert rows["forward_walk"]["expected_arm_role"] == "natural_or_unspecified"
    assert rows["one_leg_hop_right"]["expected_primary_contacts"] == "one_foot"
    assert rows["one_leg_hop_right"]["expected_root_motion"] == "aerial_or_impulsive"
    assert rows["hip_hop_heel_toe"]["motionbricks_mode_hint"] == "walk_happy_dance"
    assert rows["handstand_kickup"]["hardness"] == "extreme_negative_control"


def test_sports_acrobatics_suite_covers_requested_domains() -> None:
    rows = build_sports_rows()
    assert len(rows) == 32
    assert len({row["prompt_text"].lower() for row in rows}) == 32

    domains = Counter(row["domain"] for row in rows)
    assert domains["acrobatics"] >= 10
    assert domains["soccer"] >= 6
    assert domains["baseball"] >= 6
    assert domains["basketball"] >= 4
    assert domains["racket_sports"] >= 3

    subcategories = {row["subcategory"]: row for row in rows}
    assert subcategories["cartwheel_left"]["current_motionbricks_support"] == "__NO__"
    assert subcategories["cartwheel_right"]["current_motionbricks_support"] == "__NO__"
    assert subcategories["soccer_power_shot"]["domain"] == "soccer"
    assert subcategories["baseball_pitch_right"]["domain"] == "baseball"
    assert subcategories["baseball_bat_swing_right"]["domain"] == "baseball"


def test_sports_acrobatics_suite_marks_only_coarse_proxies_partial() -> None:
    rows = {row["subcategory"]: row for row in build_sports_rows()}
    partial_rows = {
        subcategory: row
        for subcategory, row in rows.items()
        if row["current_motionbricks_support"] == "__PARTIAL__"
    }

    assert set(partial_rows) == set(PARTIAL_MODE_HINTS)
    for subcategory, mode_hint in PARTIAL_MODE_HINTS.items():
        assert rows[subcategory]["motionbricks_mode_hint"] == mode_hint

    unsupported = [
        "soccer_inside_pass_right",
        "baseball_bat_swing_left",
        "basketball_jump_shot",
        "tennis_serve",
        "sprinter_start",
    ]
    for subcategory in unsupported:
        assert rows[subcategory]["current_motionbricks_support"] == "__NO__"


def test_humanoid_100_audit_keeps_unsupported_prompts_explicit() -> None:
    metrics = load_metrics(GUIDED_FIXTURE)
    rows = build_eval_rows(seed=0, metrics=metrics)

    assert len(rows) == 100
    render_statuses = Counter(row["render_status"] for row in rows)
    assert render_statuses["proxy_before_after"] == 22
    assert render_statuses["unsupported_placeholder"] == 78

    by_subcategory = {row["subcategory"]: row for row in rows}
    assert by_subcategory["forward_walk"]["proxy_mode"] == "walk"
    assert by_subcategory["side_shuffle_left"]["proxy_mode"] == "walk_left"
    assert by_subcategory["hand_crawl"]["proxy_mode"] == "hand_crawling"
    assert by_subcategory["elbow_crawl"]["proxy_mode"] == "elbow_crawling"
    assert by_subcategory["cartwheel_attempt"]["render_status"] == "unsupported_placeholder"


def test_humanoid_100_forced_proxy_experiment_covers_every_prompt() -> None:
    metrics = load_metrics(GUIDED_FIXTURE)
    rows = build_eval_rows(seed=0, metrics=metrics, force_proxy_all=True)

    assert len(rows) == 100
    assert Counter(row["render_status"] for row in rows) == {"proxy_before_after": 100}

    semantic = Counter(row["semantic_validity"] for row in rows)
    assert semantic["supported_proxy"] == 22
    assert semantic["forced_proxy_not_prompt_following"] == 78

    by_subcategory = {row["subcategory"]: row for row in rows}
    assert by_subcategory["cartwheel_attempt"]["proxy_mode"] == "hand_crawling"
    assert by_subcategory["carry_box_front"]["proxy_mode"] == "walk_gun"
    assert by_subcategory["vertical_jump"]["proxy_mode"] == "walk_happy_dance"


def test_retiming_preserves_qpos_shape_and_normalized_root_quaternion() -> None:
    qpos = np.zeros((4, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:, 0] = np.linspace(0.0, 0.3, len(qpos))
    qpos[:, 7:] = np.linspace(0.0, 1.0, len(qpos))[:, None]

    retimed = qpos_slerp(qpos, 2.0)

    assert retimed.shape == (8, 36)
    assert np.allclose(retimed[0, :7], qpos[0, :7])
    assert np.allclose(retimed[-1, :7], qpos[-1, :7], atol=1e-6)
    assert np.allclose(np.linalg.norm(retimed[:, 3:7], axis=1), 1.0, atol=1e-6)
    assert np.isfinite(retimed).all()


def test_repair_variants_include_identity_and_controller_friendly_retiming() -> None:
    qpos = np.zeros((10, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qpos[:, 0] = np.linspace(0.0, 0.5, len(qpos))
    qpos[:, 7:] = np.sin(np.linspace(0.0, 1.0, len(qpos)))[:, None]

    variants = repair_variants(qpos)

    assert "identity" in variants
    assert "retime_2p0x" in variants
    assert "retime_2p0x_savgol" in variants
    assert variants["identity"].shape == qpos.shape
    assert variants["retime_2p0x"].shape[0] == 20
    for candidate in variants.values():
        assert candidate.shape[1] == 36
        assert np.isfinite(candidate).all()
        assert np.allclose(np.linalg.norm(candidate[:, 3:7], axis=1), 1.0, atol=1e-5)


def test_repair_validation_rejects_bad_reference_motion() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_qpos(np.zeros((4, 35), dtype=np.float32), "bad")

    bad_quat = np.zeros((4, 36), dtype=np.float32)
    with pytest.raises(ValueError, match="near-zero"):
        validate_qpos(bad_quat, "bad")
