"""Record native SONIC ZMQ debug output to an MP4.

The official deployment binary publishes `g1_debug` messages with both target
and measured robot states. This script renders those messages offscreen using
the same G1 visualization assets shipped with GR00T-WholeBodyControl.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from lxml import etree
import msgpack
import mujoco
import numpy as np
import zmq


MUJOCO_ORDER_JOINTS = 29
ROBOT_QPOS = 7 + MUJOCO_ORDER_JOINTS


def _prepend_names(elem, prefix: str) -> None:
    if "name" in elem.attrib:
        elem.attrib["name"] = prefix + elem.attrib["name"]
    for child in elem:
        _prepend_names(child, prefix)


def _replace_attribute(elem, attribute: str, value: str) -> None:
    if attribute in elem.attrib:
        elem.attrib[attribute] = value
    for child in elem:
        _replace_attribute(child, attribute, value)


def build_two_robot_model(sonic_deploy_root: Path) -> mujoco.MjModel:
    main_scene = etree.parse(str(sonic_deploy_root / "g1/scene_empty.xml"))
    scene_asset = main_scene.find("asset")
    scene_default = main_scene.find("default")
    scene_worldbody = main_scene.find("worldbody")

    robot1 = etree.parse(str(sonic_deploy_root / "g1/g1_29dof_old.xml"))
    for mesh in robot1.find("asset").findall("mesh"):
        mesh.set("file", str(sonic_deploy_root / "g1" / "meshes" / mesh.get("file")))
        scene_asset.append(mesh)
    for default in robot1.find("default").findall("default"):
        scene_default.append(default)
    robot1_body = robot1.find("worldbody").find("body")
    _prepend_names(robot1_body, "target_")
    robot1_body.set("pos", "0 -0.55 0")
    scene_worldbody.append(robot1_body)

    robot2 = etree.parse(str(sonic_deploy_root / "g1/g1_29dof_old.xml"))
    robot2_body = robot2.find("worldbody").find("body")
    _prepend_names(robot2_body, "measured_")
    _replace_attribute(robot2_body, "rgba", "0.9 0.15 0.12 0.82")
    robot2_body.set("pos", "0 0.55 0")
    scene_worldbody.append(robot2_body)

    return mujoco.MjModel.from_xml_string(
        etree.tostring(main_scene, pretty_print=True, encoding="unicode")
    )


def unpack_debug_message(raw: bytes, topic: str) -> dict:
    payload = raw.split(topic.encode(), 1)[1]
    return msgpack.unpackb(payload, raw=False)


def set_robot_qpos(qpos: np.ndarray, base: int, pos, quat, joints) -> None:
    qpos[base : base + 3] = np.asarray(pos, dtype=np.float64)
    qpos[base + 3 : base + 7] = np.asarray(quat, dtype=np.float64)
    qpos[base + 7 : base + ROBOT_QPOS] = np.asarray(joints, dtype=np.float64)


def put_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    pad = 14
    line_h = 30
    box_h = pad * 2 + line_h * len(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], box_h), (20, 20, 20), -1)
    out = cv2.addWeighted(overlay, 0.62, out, 0.38, 0)
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (18, pad + 22 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--url", default="tcp://localhost:5557")
    parser.add_argument("--topic", default="g1_debug")
    parser.add_argument("--label", default="live SONIC debug stream")
    parser.add_argument("--sonic_deploy_root", type=Path, default=Path("/home/rewardai/repos/GR00T-WholeBodyControl/gear_sonic_deploy"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = build_two_robot_model(args.sonic_deploy_root)
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / args.fps
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 4.2
    cam.azimuth = 115
    cam.elevation = -18
    cam.lookat[:] = np.array([0.0, 0.0, 0.8])

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(args.url)
    socket.setsockopt(zmq.SUBSCRIBE, args.topic.encode())
    socket.setsockopt(zmq.RCVTIMEO, 5000)

    writer = cv2.VideoWriter(
        str(args.out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    start = time.time()
    frames = 0
    last_motion = args.label
    try:
        while time.time() - start < args.duration:
            msg = unpack_debug_message(socket.recv(), args.topic)
            set_robot_qpos(
                data.qpos,
                0,
                msg["base_trans_target"],
                msg["base_quat_target"],
                msg["body_q_target"],
            )
            measured_pos = np.asarray(msg["base_trans_measured"], dtype=np.float64)
            measured_pos[1] += 1.1
            set_robot_qpos(
                data.qpos,
                ROBOT_QPOS,
                measured_pos,
                msg["base_quat_measured"],
                msg["body_q_measured"],
            )
            # Kinematic visualization only; full mj_forward can fail on extreme
            # reference poses because it factorizes dynamics/contact Hessians.
            mujoco.mj_kinematics(model, data)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frame = put_text(
                frame,
                [
                    f"SONIC native sim2sim | {last_motion}",
                    f"target reference: white/left   measured policy: red/right   elapsed={time.time() - start:05.2f}s",
                ],
            )
            writer.write(frame)
            frames += 1
    finally:
        writer.release()
        socket.close(0)
        context.term()
        print(f"[record] wrote {frames} frames to {args.out}")


if __name__ == "__main__":
    main()
