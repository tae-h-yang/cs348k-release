"""
Main evaluation script: runs kinematic qpos sequences through physics simulation
and reports the kinematic-to-dynamic gap metrics.

Usage:
    python run_eval.py --data_dir data/synthetic
    python run_eval.py --data_dir data/motionbricks
    python run_eval.py --data_dir data/motionbricks --kinematic_baseline
    python run_eval.py --data_dir data/motionbricks --full_report
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from physics_eval.simulator import PhysicsSimulator
from physics_eval.inverse_dynamics import InverseDynamicsEvaluator
from physics_eval.validation import validate_qpos_sequence
from analysis.visualize import plot_all, plot_inverse_all
from analysis.render import render_clip_video, render_reference_video

DEMO_CLIPS = ("idle_seed0", "walk_seed2", "hand_crawling_seed2")


def load_motion_dir(data_dir: Path):
    """
    Load all .npy motion files from a directory.

    Expected file format: (T, 36) float32/64 array.
    Optional: motion_labels.npy — dict mapping filename stem → motion_type string.
    """
    labels_path = data_dir / "motion_labels.npy"
    labels = np.load(labels_path, allow_pickle=True).item() if labels_path.exists() else {}

    motions = []
    for p in sorted(data_dir.glob("*.npy")):
        if p.stem == "motion_labels":
            continue
        try:
            seq = validate_qpos_sequence(np.load(p), name=p.name, min_frames=1)
        except ValueError as e:
            print(f"Skipping {p.name}: {e}")
            continue
        motions.append((p.stem, labels.get(p.stem, "unknown"), seq))

    return motions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/synthetic",
                        help="Directory of .npy qpos files to evaluate")
    parser.add_argument("--kinematic_baseline", action="store_true",
                        help="Also run kinematic replay (mj_forward) baseline for gap comparison")
    parser.add_argument("--inverse_dynamics", action="store_true",
                        help="Run inverse-dynamics feasibility analysis")
    parser.add_argument("--full_report", action="store_true",
                        help="Enable the final-report defaults: kinematic baseline + inverse dynamics")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Continue simulation even after fall (slower)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--render", action="store_true",
                        help="Render selected demo videos to results/videos/")
    parser.add_argument("--render_mode", choices=("reference", "side_by_side", "physics"),
                        default="reference",
                        help="Video mode for --render. Use reference for presentation-safe videos.")
    parser.add_argument("--render_all", action="store_true",
                        help="With --render, render every clip instead of the representative demo subset")
    parser.add_argument("--pd_kp_scale", type=float, default=0.5,
                        help="Scale nominal PD stiffness gains (default: 0.5)")
    parser.add_argument("--pd_kd_scale", type=float, default=1.0,
                        help="Scale nominal PD damping gains (default: 1.0)")
    parser.add_argument("--pd_force_scale", type=float, default=1.0,
                        help="Scale actuator torque limits for controller stress tests (default: 1.0)")
    args = parser.parse_args()

    if args.full_report:
        args.kinematic_baseline = True
        args.inverse_dynamics = True

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    motions = load_motion_dir(data_dir)
    if not motions:
        print(f"No valid .npy motion files found in {data_dir}")
        sys.exit(1)

    print(f"Loaded {len(motions)} motion clips from {data_dir}")
    print()

    sim = PhysicsSimulator(
        pd_kp_scale=args.pd_kp_scale,
        pd_kd_scale=args.pd_kd_scale,
        pd_force_scale=args.pd_force_scale,
    )
    inv_eval = InverseDynamicsEvaluator() if args.inverse_dynamics else None
    physics_results = []
    kinematic_results = []
    inverse_results = []

    render_names = {name for name in DEMO_CLIPS}
    if args.render_all:
        render_names = {name for name, _, _ in motions}

    for clip_name, motion_type, qpos_seq in motions:
        print(f"  [{clip_name}] {motion_type} — {len(qpos_seq)} frames")

        try:
            phys = sim.evaluate_clip(
                qpos_seq, clip_name=clip_name, motion_type=motion_type,
                early_stop_on_fall=not args.no_early_stop
            )
        except ValueError as e:
            print(f"    physics  → skipped: {e}")
            continue
        physics_results.append(phys)
        s = phys.summary()
        fell_str = f"FELL at frame {s['time_to_fall']}" if s['fell'] else "survived"
        print(f"    physics  → {fell_str} | tracking={s['tracking_rmse']:.4f} rad | "
              f"root_err={s['root_pos_error']:.3f} m | power={s['mech_power_W']:.1f} W")

        if args.kinematic_baseline:
            kin = sim.evaluate_clip_kinematic(qpos_seq, clip_name=clip_name, motion_type=motion_type)
            kinematic_results.append(kin)
            sk = kin.summary()
            fell_str_k = f"height<0.45m at {sk['time_to_fall']}" if sk['fell'] else "ok"
            print(f"    kinematic→ {fell_str_k} | joint_viol={sk['max_joint_viol']} | "
                  f"foot_pen={sk['foot_penetration_m']:.4f} m")

        if inv_eval is not None:
            inv = inv_eval.evaluate_clip(qpos_seq, clip_name=clip_name, motion_type=motion_type)
            inverse_results.append(inv)
            si = inv.summary()
            print(f"    inverse → p95_tau/limit={si['p95_torque_limit_ratio']:.2f} | "
                  f"exceed={si['exceeded_joint_pct']:.1f}% joints | "
                  f"rootF95={si['p95_root_force_N']:.0f} N")

        if args.render and clip_name in render_names:
            if args.render_mode == "reference":
                vid_path = render_reference_video(sim, qpos_seq, clip_name=clip_name)
            else:
                vid_path = render_clip_video(
                    sim, qpos_seq, clip_name=clip_name, motion_type=motion_type,
                    side_by_side=args.render_mode == "side_by_side"
                )
            print(f"    video    → {vid_path}")

    print()
    if not args.no_plot:
        all_results = physics_results + kinematic_results
        plot_all(all_results)
        if inverse_results:
            plot_inverse_all(inverse_results, physics_results)


if __name__ == "__main__":
    main()
