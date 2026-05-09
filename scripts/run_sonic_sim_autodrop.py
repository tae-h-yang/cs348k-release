"""Headless SONIC MuJoCo simulator launcher with automatic elastic-band release.

This uses the official GR00T-WholeBodyControl simulator classes, but removes
the need for a human to click the viewer and press `9` before policy playback.
It is intended for local integration checks of the native SONIC deployment
binary against the official MuJoCo sim loop.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groot_root", type=Path, default=Path("/home/rewardai/repos/GR00T-WholeBodyControl"))
    parser.add_argument("--drop_after", type=float, default=3.0)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--enable_onscreen", action="store_true")
    parser.add_argument("--enable_offscreen", action="store_true")
    parser.add_argument("--qpos_log", type=Path, default=None)
    parser.add_argument("--release_trigger", type=Path, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(args.groot_root))

    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    from gear_sonic.scripts.run_sim_loop import SimWrapper
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    config = SimLoopConfig(
        interface=args.interface,
        enable_onscreen=args.enable_onscreen,
        enable_offscreen=args.enable_offscreen,
        verbose=False,
    )
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name

    robot_model = instantiate_g1_robot_model()
    sim_wrapper = SimWrapper(
        robot_model=robot_model,
        env_name=config.env_name,
        config=wbc_config,
        onscreen=wbc_config.get("ENABLE_ONSCREEN", False),
        offscreen=wbc_config.get("ENABLE_OFFSCREEN", False),
        enable_image_publish=False,
    )

    start = time.time()
    released = False
    sim = sim_wrapper.sim
    release_desc = f"{args.drop_after:.1f}s"
    if args.release_trigger is not None:
        release_desc += f" or trigger {args.release_trigger}"
    print(
        f"[autodrop] official SONIC sim running on {args.interface}; "
        f"elastic band release in {release_desc}"
    )
    log_file = None
    writer = None
    if args.qpos_log is not None:
        args.qpos_log.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.qpos_log, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(
            ["wall_time", "sim_time"]
            + [f"qpos_{i}" for i in range(sim.sim_env.mj_model.nq)]
            + [f"qvel_{i}" for i in range(sim.sim_env.mj_model.nv)]
        )
    try:
        while sim._running and (
            (sim.sim_env.viewer and sim.sim_env.viewer.is_running()) or sim.sim_env.viewer is None
        ):
            trigger_release = args.release_trigger is not None and args.release_trigger.exists()
            if not released and (time.time() - start >= args.drop_after or trigger_release):
                if sim.sim_env.elastic_band is not None:
                    sim.sim_env.elastic_band.enable = False
                released = True
                reason = "trigger" if trigger_release else "timer"
                print(f"[autodrop] elastic band disabled by {reason}")

            step_start = time.monotonic()
            sim.sim_env.sim_step()
            if writer is not None:
                writer.writerow(
                    [time.time(), sim.sim_env.mj_data.time]
                    + sim.sim_env.mj_data.qpos.tolist()
                    + sim.sim_env.mj_data.qvel.tolist()
                )
            elapsed = time.monotonic() - step_start
            sleep_time = sim.sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("[autodrop] interrupted")
    finally:
        if log_file is not None:
            log_file.close()
        sim.close()


if __name__ == "__main__":
    main()
