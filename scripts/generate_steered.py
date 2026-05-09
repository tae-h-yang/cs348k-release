"""
Per-segment online critic-steered generation for MotionBricks.

At each MotionBricks generation step (every ~N frames), instead of committing
the first candidate, we sample K variants (different seeds + stochastic Gumbel)
and commit the lowest-risk one according to the physics critic.

This is strictly more powerful than whole-clip best-of-K:
  - Whole-clip BoK: generate K full clips, pick best (no feedback between gens)
  - Per-segment steering: at each segment, pick best of K → next segment
    conditions on the committed best, not a random one. The critic guidance
    propagates forward through the generation process.

Analogous to per-token best-of-N in LLMs vs. sequence-level reranking.
"""
from __future__ import annotations

import sys, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MOTIONBRICKS_DIR = Path(__file__).parent.parent.parent / "GR00T-WholeBodyControl" / "motionbricks"

from physics_eval.online_critic import OnlineSegmentCritic


# ── Stochastic sampling patch (shared with generate_guided.py) ────────────────

def _enable_stochastic(inferencer):
    orig = inferencer.predict
    def _stoch(*a, config=None, **kw):
        config = dict(config or {})
        config["pose_token_sampling_use_argmax"] = False
        return orig(*a, config=config, **kw)
    inferencer._orig_predict = orig
    inferencer.predict = _stoch

def _disable_stochastic(inferencer):
    if hasattr(inferencer, "_orig_predict"):
        inferencer.predict = inferencer._orig_predict
        del inferencer._orig_predict


# ── Per-segment state save/restore ────────────────────────────────────────────

def _save_agent_state(full_agent):
    return {
        "mujoco_qpos": full_agent.frames["mujoco_qpos"].clone(),
        "model_features": full_agent.frames["model_features"].clone(),
        "frame_idx": full_agent._current_frame_idx,
        "mode": full_agent.frames.get("mode"),
    }

def _restore_agent_state(full_agent, state):
    full_agent.frames["mujoco_qpos"] = state["mujoco_qpos"].clone()
    full_agent.frames["model_features"] = state["model_features"].clone()
    full_agent._current_frame_idx = state["frame_idx"]
    if state["mode"] is not None:
        full_agent.frames["mode"] = state["mode"]


# ── Core generation ───────────────────────────────────────────────────────────

def generate_clip_steered(
    demo_agent,
    mode: str,
    n_frames: int,
    seed: int,
    K_per_segment: int = 4,
    critic: OnlineSegmentCritic | None = None,
    generate_dt: float = 2.0,
    switch_margin: float = 0.25,
) -> tuple[np.ndarray, dict]:
    """
    Generate one clip with per-segment online critic steering.

    At each generation step that triggers new MotionBricks inference, we run K
    candidates (seed 0 = deterministic, seeds 1..K-1 = stochastic Gumbel) and
    commit the lowest-risk segment. The committed segment becomes the context
    for the next step.

    Returns:
        qpos: (n_frames, 36) float32 array
        meta: dict with per-segment risks and summary stats
    """
    import mujoco

    np.random.seed(seed)
    torch.manual_seed(seed)

    full_agent = demo_agent.full_agent
    inferencer = full_agent._inferencer
    controller_dt = demo_agent.controller.get_controller_dt()
    effective_dt = controller_dt * generate_dt

    full_agent.reset()

    qpos_list: list[np.ndarray] = []
    segment_risks: list[float] = []   # risk of the committed segment at each gen step
    segment_k_chosen: list[int] = []  # which candidate was selected
    n_segments_steered = 0
    n_segments_total = 0

    for step in range(n_frames):
        # ── pop next frame from buffer ─────────────────────────────────────
        qpos = full_agent.get_next_frame()
        qpos_list.append(qpos.copy())

        context_mujoco_qpos = full_agent.get_context_mujoco_qpos()
        demo_agent.mj_data.qpos[:] = qpos

        force_idle = step + 100 > n_frames
        control_signals = demo_agent.controller.generate_control_signals(
            None, demo_agent.mj_model, demo_agent.mj_data, visualize=False,
            control_info={"force_idle": force_idle, "allowed_mode": mode}
        )
        control_signals["context_mujoco_qpos"] = context_mujoco_qpos

        # ── determine whether generation will trigger ──────────────────────
        will_generate = (
            full_agent._current_frame_idx >= effective_dt * full_agent._fps
            or full_agent._current_frame_idx >= full_agent.frames["mujoco_qpos"].shape[1] - 1
        )
        should_steer = will_generate and K_per_segment > 1 and critic is not None

        if should_steer:
            # Save agent state so all candidates start from the same context
            saved_state = _save_agent_state(full_agent)

            # Include last N accumulated frames in scoring to penalize
            # inter-segment boundary jerk/acceleration
            N_OVERLAP = 8
            prev_context = (
                np.stack(qpos_list[-N_OVERLAP:]) if len(qpos_list) >= N_OVERLAP
                else np.stack(qpos_list)
            )

            best_risk = float("inf")
            best_state_after: dict | None = None
            best_k = 0
            risk_k0 = float("inf")   # risk of the deterministic (natural) candidate

            for k in range(K_per_segment):
                _restore_agent_state(full_agent, saved_state)

                seed_k = seed + k * 137
                cs_k = dict(control_signals)
                cs_k["random_seed"] = torch.tensor([seed_k])

                if k > 0:
                    _enable_stochastic(inferencer)
                try:
                    with torch.no_grad():
                        full_agent.generate_new_frames(cs_k, effective_dt,
                                                       force_generation=True)
                    new_qpos = full_agent.frames["mujoco_qpos"][0].detach().cpu().numpy()
                    combined = np.vstack([prev_context, new_qpos])
                    risk_k = critic.score_segment(combined)
                except Exception:
                    risk_k = float("inf")
                finally:
                    if k > 0:
                        _disable_stochastic(inferencer)

                if k == 0:
                    risk_k0 = risk_k

                if risk_k < best_risk:
                    best_risk = risk_k
                    best_state_after = _save_agent_state(full_agent)
                    best_k = k

            # Only switch to stochastic candidate if it's meaningfully better.
            # Small-margin wins are noise: the segment critic is coarser than the
            # full-clip critic, so switching for <25% improvement tends to hurt.
            min_gain = switch_margin * max(risk_k0, 1e-6)
            if best_k > 0 and (risk_k0 - best_risk) < min_gain:
                # Gain is too small — revert to k=0 (natural deterministic)
                _restore_agent_state(full_agent, saved_state)
                cs0 = dict(control_signals)
                cs0["random_seed"] = torch.tensor([seed])
                with torch.no_grad():
                    full_agent.generate_new_frames(cs0, effective_dt, force_generation=True)
                best_k = 0
                best_risk = risk_k0
            else:
                # Commit the winning candidate's state
                _restore_agent_state(full_agent, best_state_after)

            segment_risks.append(best_risk)
            segment_k_chosen.append(best_k)
            n_segments_steered += (1 if best_k > 0 else 0)
            n_segments_total += 1

        else:
            # Normal (unsteered) generation
            control_signals["random_seed"] = torch.tensor([seed])
            with torch.no_grad():
                full_agent.generate_new_frames(control_signals, effective_dt)
            if will_generate:
                n_segments_total += 1

        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)

    qpos_arr = np.stack(qpos_list, axis=0).astype(np.float32)
    meta = {
        "segment_risks": segment_risks,
        "segment_k_chosen": segment_k_chosen,
        "mean_segment_risk": float(np.mean(segment_risks)) if segment_risks else float("nan"),
        "min_segment_risk": float(np.min(segment_risks)) if segment_risks else float("nan"),
        "n_segments_steered": n_segments_steered,
        "n_segments_total": n_segments_total,
        "K_per_segment": K_per_segment,
    }
    return qpos_arr, meta
