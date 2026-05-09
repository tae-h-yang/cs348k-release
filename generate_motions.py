"""
MotionBricks motion generation script (local GPU).

Generates diverse kinematic qpos sequences from MotionBricks, categorized
by motion type, and saves them as .npy files for physics evaluation.

Setup (one-time):
    git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
    cd GR00T-WholeBodyControl/motionbricks
    git lfs pull
    pip install -e .
    cd <this repo>

Run:
    python generate_motions.py

Output: one .npy file per clip of shape (T, 36) — mujoco qpos format,
        saved to data/motionbricks/.
"""

import sys
import numpy as np
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
RESULT_DIR = Path(__file__).parent / "data" / "motionbricks"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Motion types — exact allowed_mode keys from clip_holder_G1 in
# motionbricks/motion_backbone/demo/clips.py
MOTION_CONFIGS = [
    {"mode": "idle",             "n_frames": 150, "type": "static"},
    {"mode": "walk",             "n_frames": 200, "type": "locomotion"},
    {"mode": "slow_walk",        "n_frames": 200, "type": "locomotion"},
    {"mode": "stealth_walk",     "n_frames": 200, "type": "locomotion"},
    {"mode": "injured_walk",     "n_frames": 200, "type": "locomotion"},
    {"mode": "walk_zombie",      "n_frames": 200, "type": "locomotion"},
    {"mode": "walk_stealth",     "n_frames": 180, "type": "locomotion"},
    {"mode": "walk_boxing",      "n_frames": 180, "type": "expressive"},
    {"mode": "walk_happy_dance", "n_frames": 180, "type": "expressive"},
    {"mode": "walk_gun",         "n_frames": 180, "type": "expressive"},
    {"mode": "walk_scared",      "n_frames": 180, "type": "expressive"},
    {"mode": "hand_crawling",    "n_frames": 150, "type": "whole_body"},
    {"mode": "elbow_crawling",   "n_frames": 150, "type": "whole_body"},
]

N_SEEDS = 3   # generate N random seeds per mode for variability


def generate_clip(demo_agent, mode: str, n_frames: int, seed: int) -> np.ndarray:
    """
    Generate a single kinematic trajectory from MotionBricks.

    Returns: (n_frames, 36) float32 qpos array in MuJoCo format.
    """
    import torch
    import mujoco
    np.random.seed(seed)
    torch.manual_seed(seed)

    demo_agent.full_agent.reset()
    qpos_list = []

    generate_dt = 2.0  # matches navigation_demo default

    for step in range(n_frames):
        qpos = demo_agent.full_agent.get_next_frame()
        qpos_list.append(qpos.copy())

        context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
        demo_agent.mj_data.qpos[:] = qpos

        force_idle = step + 100 > n_frames
        control_signals = demo_agent.controller.generate_control_signals(
            None, demo_agent.mj_model, demo_agent.mj_data, visualize=False,
            control_info={"force_idle": force_idle, "allowed_mode": mode}
        )
        control_signals["context_mujoco_qpos"] = context_mujoco_qpos

        with torch.no_grad():
            demo_agent.full_agent.generate_new_frames(
                control_signals,
                demo_agent.controller.get_controller_dt() * generate_dt
            )

        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)

    return np.stack(qpos_list, axis=0).astype(np.float32)


MOTIONBRICKS_DIR = Path(__file__).parent.parent / "GR00T-WholeBodyControl" / "motionbricks"


def main():
    import os, argparse
    from motionbricks.motion_backbone.demo.utils import navigation_demo

    # MotionBricks resolves asset paths relative to its own directory
    original_dir = Path.cwd()
    os.chdir(MOTIONBRICKS_DIR)

    # Let navigation_demo resolve humanoid_xml, result_dir, data_root from the
    # package install location — only override fields that don't have sensible defaults.
    args = argparse.Namespace(
        explicit_dataset_folder=None,
        reprocess_clips=0,
        controller="random",
        lookat_movement_direction=0,
        has_viewer=0,
        pre_filter_qpos=1,
        source_root_realignment=1,
        target_root_realignment=1,
        force_canonicalization=1,
        skip_ending_target_cond=0,
        random_speed_scale=0,
        speed_scale=[0.8, 1.2],
        generate_dt=2.0,
        max_steps=10000,
        random_seed=42,
        num_runs=1,
        use_qpos=1,
        planner="default",
        allowed_mode=None,
        clips="G1",
        return_model_configs=True,
        return_dataloader=True,
        recording_dir=None,
        EXP="default",
    )

    print("Loading MotionBricks model...")
    demo_agent = navigation_demo(args)
    print("Model loaded.")

    labels = {}
    for cfg in MOTION_CONFIGS:
        mode, n_frames, mtype = cfg["mode"], cfg["n_frames"], cfg["type"]
        for seed in range(N_SEEDS):
            clip_name = f"{mode}_seed{seed}"
            print(f"  Generating: {clip_name} ({n_frames} frames)...")
            try:
                qpos = generate_clip(demo_agent, mode, n_frames, seed=seed * 1000)
                out_path = RESULT_DIR / f"{clip_name}.npy"
                np.save(out_path, qpos)
                labels[clip_name] = mtype
                print(f"    Saved: {out_path} — shape {qpos.shape}")
            except Exception as e:
                print(f"    FAILED: {e}")

    os.chdir(original_dir)
    np.save(RESULT_DIR / "motion_labels.npy", labels)
    print(f"\nDone. {len(labels)} clips saved to {RESULT_DIR}/")
    print("Run evaluation with: python run_eval.py --data_dir data/motionbricks")


if __name__ == "__main__":
    main()
