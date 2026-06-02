"""Render official SONIC examples from the simulator's actual MuJoCo qpos.

The native C++ `g1_debug` publisher is useful for quick visualization, but its
measured base translation is hard-coded. This script records the official
MuJoCo simulator state directly and renders reference-vs-actual videos from
that logged qpos, so root motion/floating artifacts in the debug stream do not
contaminate the visual check.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pty
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from lxml import etree
import mujoco
import numpy as np


CS348K_ROOT = Path(__file__).resolve().parents[1]
GROOT_ROOT = Path("/home/rewardai/repos/GR00T-WholeBodyControl")
DEPLOY_ROOT = GROOT_ROOT / "gear_sonic_deploy"
SIM_PYTHON = GROOT_ROOT / ".venv_sim/bin/python"
DEFAULT_REFERENCE_ROOT = DEPLOY_ROOT / "reference/example"

MUJOCO_ORDER_JOINTS = 29
ROBOT_QPOS = 7 + MUJOCO_ORDER_JOINTS

# gear_sonic_deploy/src/g1/.../policy_parameters.hpp
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)

LD_LIBRARY_PATH = ":".join(
    [
        "/home/rewardai/anaconda3/lib/python3.12/site-packages/nvidia/cu13/lib",
        "/home/rewardai/anaconda3/lib/python3.12/site-packages/tensorrt_libs",
        str(DEPLOY_ROOT / "thirdparty/unitree_sdk2/lib/x86_64"),
        str(DEPLOY_ROOT / "thirdparty/unitree_sdk2/thirdparty/lib/x86_64"),
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]
)


@dataclass(frozen=True)
class MotionJob:
    name: str
    duration: float


DEFAULT_JOBS = [
    MotionJob("forward_lunge_R_001__A359_M", 10.0),
    MotionJob("walking_quip_360_R_002__A428", 11.0),
    MotionJob("squat_001__A359", 10.0),
    MotionJob("neutral_kick_R_001__A543", 6.0),
    MotionJob("dance_in_da_party_001__A464", 12.0),
    MotionJob("macarena_001__A545_M", 16.0),
]


class PtyProcess:
    def __init__(self, cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path):
        self.master_fd, slave_fd = pty.openpty()
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
        os.close(slave_fd)
        os.set_blocking(self.master_fd, False)
        self.log = bytearray()
        self.log_path = log_path

    def read_available(self, timeout: float = 0.0) -> str:
        chunks: list[bytes] = []
        end = time.time() + timeout
        while True:
            wait = max(0.0, end - time.time()) if timeout else 0.0
            readable, _, _ = select.select([self.master_fd], [], [], wait)
            if not readable:
                break
            try:
                data = os.read(self.master_fd, 65536)
            except BlockingIOError:
                break
            except OSError:
                break
            if not data:
                break
            self.log.extend(data)
            chunks.append(data)
            if timeout and time.time() >= end:
                break
        if chunks:
            self.log_path.write_bytes(self.log)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def wait_for(self, needle: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        text = self.log.decode("utf-8", errors="replace")
        while time.time() < deadline:
            if needle in text:
                return True
            text += self.read_available(timeout=0.25)
            if self.proc.poll() is not None:
                return needle in text
        return needle in text

    def send(self, text: str) -> None:
        os.write(self.master_fd, text.encode("utf-8"))

    def terminate(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        self.read_available()
        try:
            os.close(self.master_fd)
        except OSError:
            pass


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


def build_two_robot_model() -> mujoco.MjModel:
    main_scene = etree.parse(str(DEPLOY_ROOT / "g1/scene_empty.xml"))
    scene_asset = main_scene.find("asset")
    scene_default = main_scene.find("default")
    scene_worldbody = main_scene.find("worldbody")

    ref_robot = etree.parse(str(DEPLOY_ROOT / "g1/g1_29dof_old.xml"))
    for mesh in ref_robot.find("asset").findall("mesh"):
        mesh.set("file", str(DEPLOY_ROOT / "g1" / "meshes" / mesh.get("file")))
        scene_asset.append(mesh)
    for default in ref_robot.find("default").findall("default"):
        scene_default.append(default)
    ref_body = ref_robot.find("worldbody").find("body")
    _prepend_names(ref_body, "ref_")
    _replace_attribute(ref_body, "rgba", "0.9 0.15 0.12 0.62")
    ref_body.set("pos", "0 0 0")
    scene_worldbody.append(ref_body)

    actual_robot = etree.parse(str(DEPLOY_ROOT / "g1/g1_29dof_old.xml"))
    actual_body = actual_robot.find("worldbody").find("body")
    _prepend_names(actual_body, "actual_")
    actual_body.set("pos", "0 0 0")
    scene_worldbody.append(actual_body)

    return mujoco.MjModel.from_xml_string(etree.tostring(main_scene, pretty_print=True, encoding="unicode"))


def read_csv_array(path: Path) -> np.ndarray:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[float(x) for x in row] for row in reader if row]
    return np.asarray(rows, dtype=np.float64)


def load_reference_qpos(motion_dir: Path) -> np.ndarray:
    joints_isaac = read_csv_array(motion_dir / "joint_pos.csv")
    root_pos = read_csv_array(motion_dir / "body_pos.csv").reshape(len(joints_isaac), -1, 3)[:, 0, :]
    root_quat = read_csv_array(motion_dir / "body_quat.csv").reshape(len(joints_isaac), -1, 4)[:, 0, :]
    root_quat /= np.linalg.norm(root_quat, axis=1, keepdims=True).clip(1e-8)

    qpos = np.zeros((len(joints_isaac), ROBOT_QPOS), dtype=np.float64)
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = root_quat
    qpos[:, 7:] = joints_isaac[:, ISAACLAB_TO_MUJOCO]
    return qpos


def extract_actual_29dof_qpos(sim_qpos: np.ndarray) -> np.ndarray:
    if sim_qpos.shape[1] == ROBOT_QPOS:
        return sim_qpos[:, :ROBOT_QPOS].copy()
    if sim_qpos.shape[1] >= 50:
        # scene_29dof_with_hand layout:
        # free root, 6+6 legs, 3 waist, 7 left arm, 7 left hand,
        # 7 right arm, 7 right hand. Drop hand joints for the 29-dof renderer.
        out = np.zeros((len(sim_qpos), ROBOT_QPOS), dtype=np.float64)
        out[:, :7] = sim_qpos[:, :7]
        out[:, 7:29] = sim_qpos[:, 7:29]
        out[:, 29:36] = sim_qpos[:, 36:43]
        return out
    raise ValueError(f"Unsupported simulator qpos width {sim_qpos.shape[1]}; expected 36 or >=50")


def align_root_xy(qpos: np.ndarray, side_y: float, mode: str) -> np.ndarray:
    aligned = qpos.copy()
    if mode == "initial":
        aligned[:, 0] -= aligned[0, 0]
        aligned[:, 1] -= aligned[0, 1]
    elif mode == "per_frame":
        aligned[:, 0] = 0.0
        aligned[:, 1] = 0.0
    else:
        raise ValueError(f"Unknown root alignment mode: {mode}")
    aligned[:, 1] += side_y
    return aligned


def ground_align_sequence_z(qpos: np.ndarray) -> np.ndarray:
    """Align a rendered qpos sequence to the floor in the render model.

    Native SONIC simulator logs and the 29-DoF presentation mesh use slightly
    different root-height conventions. For visualization only, place the lowest
    ankle/foot collision geometry at z=0 on each rendered frame so the mesh does
    not appear to float above the floor.
    """
    model = mujoco.MjModel.from_xml_path(str(DEPLOY_ROOT / "g1/g1_29dof_old.xml"))
    data = mujoco.MjData(model)
    foot_geoms = [
        gid
        for gid in range(model.ngeom)
        if "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or "")
    ]
    if not foot_geoms:
        return qpos
    aligned = qpos.copy()
    for i in range(len(aligned)):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = aligned[i]
        mujoco.mj_forward(model, data)
        min_foot_z = min(float(data.geom_xpos[gid, 2] - model.geom_size[gid, 0]) for gid in foot_geoms)
        aligned[i, 2] -= min_foot_z
    return aligned


def put_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    pad = 14
    line_h = 27
    box_h = pad * 2 + line_h * len(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], box_h), (18, 18, 18), -1)
    out = cv2.addWeighted(overlay, 0.68, out, 0.32, 0)
    for i, line in enumerate(lines):
        max_width = max(80, out.shape[1] - 2 * pad)
        font_scale = 0.56
        thickness = 2
        text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if text_width > max_width:
            font_scale = max(0.36, font_scale * max_width / text_width)
        cv2.putText(
            out,
            line,
            (18, pad + 21 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )
    return out


def body_prefix_for_geom(model: mujoco.MjModel, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    if body_name.startswith("ref_"):
        return "ref"
    if body_name.startswith("actual_"):
        return "actual"
    return ""


def add_contact_markers(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> None:
    for i in range(data.ncon):
        contact = data.contact[i]
        prefix1 = body_prefix_for_geom(model, int(contact.geom1))
        prefix2 = body_prefix_for_geom(model, int(contact.geom2))
        if prefix1 and prefix2 and prefix1 != prefix2:
            # In overlay renders, reference and actual robots occupy the same
            # scene. Skip artificial robot-vs-robot contacts; keep floor/body
            # contacts for the reference or the physical robot.
            continue
        prefix = prefix1 or prefix2
        if not prefix:
            continue
        if renderer.scene.ngeom >= renderer.scene.maxgeom:
            break
        rgba = np.array([0.1, 0.95, 0.25, 1.0]) if prefix == "ref" else np.array([1.0, 0.85, 0.05, 1.0])
        mujoco.mjv_initGeom(
            renderer.scene.geoms[renderer.scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.055, 0.055, 0.055], dtype=np.float64),
            np.asarray(contact.pos, dtype=np.float64),
            np.eye(3).reshape(-1),
            rgba,
        )
        renderer.scene.ngeom += 1


def render_qpos_video(
    motion_name: str,
    ref_qpos: np.ndarray,
    actual_qpos: np.ndarray,
    out_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    align_mode: str,
    contact_markers: bool = False,
    camera_track: bool = False,
    no_text_overlay: bool = False,
    overlay_same_origin: bool = False,
    ground_align_render: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = min(len(ref_qpos), len(actual_qpos))
    if n_frames < 2:
        raise ValueError(f"Not enough frames for {motion_name}: {n_frames}")
    side_offset = 0.0 if overlay_same_origin else 0.55
    ref_qpos = align_root_xy(ref_qpos[:n_frames], -side_offset, align_mode)
    actual_qpos = align_root_xy(actual_qpos[:n_frames], side_offset, align_mode)
    if ground_align_render:
        ref_qpos = ground_align_sequence_z(ref_qpos)
        actual_qpos = ground_align_sequence_z(actual_qpos)

    model = build_two_robot_model()
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 4.4
    cam.azimuth = 115
    cam.elevation = -18
    cam.lookat[:] = np.array([0.0, 0.0, 0.78])

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for i in range(n_frames):
            data.qpos[:ROBOT_QPOS] = ref_qpos[i]
            data.qpos[ROBOT_QPOS : 2 * ROBOT_QPOS] = actual_qpos[i]
            contacts_available = False
            if contact_markers:
                try:
                    mujoco.mj_forward(model, data)
                    contacts_available = True
                except mujoco.FatalError:
                    # Some extreme replay poses make MuJoCo's constraint
                    # factorization singular. Keep the diagnostic video alive
                    # and render the pose without contact dots for that frame.
                    mujoco.mj_kinematics(model, data)
            else:
                mujoco.mj_kinematics(model, data)
            if camera_track:
                cam.lookat[:] = np.array(
                    [
                        0.5 * (ref_qpos[i, 0] + actual_qpos[i, 0]),
                        0.5 * (ref_qpos[i, 1] + actual_qpos[i, 1]),
                        max(0.55, 0.5 * (ref_qpos[i, 2] + actual_qpos[i, 2])),
                    ]
                )
            renderer.update_scene(data, camera=cam)
            if contacts_available:
                add_contact_markers(model, data, renderer)
            rgb = renderer.render()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not no_text_overlay:
                align_desc = "initial-root aligned" if align_mode == "initial" else "pose-aligned each frame"
                frame = put_text(
                    frame,
                    [
                        f"SONIC actual sim qpos | {motion_name}",
                        f"red: reference   mesh: SONIC physics actual   t={i / fps:05.2f}s",
                        f"{align_desc}; contact dots: green=reference, yellow=actual.",
                    ],
                )
            writer.write(frame)
    finally:
        writer.release()


def make_single_motion_reference(motion_name: str, tmp_root: Path) -> Path:
    motion_src = DEFAULT_REFERENCE_ROOT / motion_name
    if not motion_src.exists():
        raise FileNotFoundError(f"Missing official reference motion: {motion_src}")
    single_ref_parent = tmp_root / "reference"
    single_ref_parent.mkdir(parents=True, exist_ok=True)
    dst = single_ref_parent / motion_name
    try:
        os.symlink(motion_src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(motion_src, dst)
    return single_ref_parent


def make_single_motion_reference_from_root(motion_name: str, reference_root: Path, tmp_root: Path) -> Path:
    motion_src = reference_root / motion_name
    if not motion_src.exists():
        raise FileNotFoundError(f"Missing reference motion: {motion_src}")
    single_ref_parent = tmp_root / "reference"
    single_ref_parent.mkdir(parents=True, exist_ok=True)
    dst = single_ref_parent / motion_name
    try:
        os.symlink(motion_src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(motion_src, dst)
    return single_ref_parent


def infer_reference_duration(motion_dir: Path, fps: float = 50.0) -> float:
    joints = read_csv_array(motion_dir / "joint_pos.csv")
    return max(1.0, (len(joints) - 1) / fps)


def rollout_metrics(ref_qpos: np.ndarray, actual_qpos: np.ndarray, fps: float) -> dict[str, float | bool | int]:
    n = min(len(ref_qpos), len(actual_qpos))
    if n == 0:
        return {
            "frames": 0,
            "duration_s": 0.0,
            "fell": True,
            "fall_time_s": 0.0,
            "min_root_z": float("nan"),
            "mean_joint_rmse": float("nan"),
            "mean_root_xy_error": float("nan"),
        }
    ref = ref_qpos[:n]
    actual = actual_qpos[:n]
    root_z = actual[:, 2]
    fall_idx = np.where(root_z < 0.55)[0]
    fell = len(fall_idx) > 0
    fall_frame = int(fall_idx[0]) if fell else n

    ref_root_z = ref[:, 2]
    # Fixed root-height thresholds are useful for upright motions but unfair to
    # intentional low-posture references such as crawls, rolls, and pushups.
    # This threshold follows the lower envelope of the reference while keeping a
    # small hard floor to catch simulator collapses through the ground.
    ref_low_threshold = max(0.10, min(0.55, float(np.percentile(ref_root_z, 5)) - 0.10))
    ref_aware_fall_idx = np.where(root_z < ref_low_threshold)[0]
    ref_aware_fell = len(ref_aware_fall_idx) > 0
    ref_aware_fall_frame = int(ref_aware_fall_idx[0]) if ref_aware_fell else n
    return {
        "frames": int(n),
        "duration_s": float(n / fps),
        "fell": bool(fell),
        "fall_time_s": float(fall_frame / fps),
        "ref_aware_fell": bool(ref_aware_fell),
        "ref_aware_fall_time_s": float(ref_aware_fall_frame / fps),
        "ref_root_z_p05": float(np.percentile(ref_root_z, 5)),
        "ref_aware_root_z_threshold": float(ref_low_threshold),
        "min_root_z": float(np.min(root_z)),
        "mean_joint_rmse": float(np.sqrt(np.mean((actual[:, 7:] - ref[:, 7:]) ** 2))),
        "mean_root_xy_error": float(np.mean(np.linalg.norm(actual[:, :2] - ref[:, :2], axis=1))),
    }


def run_one(job: MotionJob, out_dir: Path, args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    job_dir = out_dir / job.name
    job_dir.mkdir(parents=True, exist_ok=True)
    sim_log = job_dir / "sim_qpos.csv"
    sim_stdout = job_dir / "sim_stdout.log"
    deploy_log = job_dir / "deploy_output.log"
    playback_window_path = job_dir / "playback_window.json"
    release_trigger = job_dir / "release_elastic_band.trigger"
    if release_trigger.exists():
        release_trigger.unlink()

    with tempfile.TemporaryDirectory(prefix=f"sonic_ref_{job.name}_") as tmp:
        single_ref_parent = make_single_motion_reference_from_root(job.name, args.reference_root, Path(tmp))
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"

        with sim_stdout.open("wb") as sim_out:
            sim = subprocess.Popen(
                [
                    str(SIM_PYTHON),
                    str(CS348K_ROOT / "scripts/run_sonic_sim_autodrop.py"),
                    "--drop_after",
                    str(args.drop_after),
                    "--interface",
                    args.interface,
                    "--qpos_log",
                    str(sim_log),
                    "--release_trigger",
                    str(release_trigger),
                ],
                cwd=CS348K_ROOT,
                env=env,
                stdout=sim_out,
                stderr=subprocess.STDOUT,
            )

            time.sleep(args.sim_warmup)
            deploy_env = env.copy()
            deploy_env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
            deploy = PtyProcess(
                [
                    str(DEPLOY_ROOT / "target/release/g1_deploy_onnx_ref"),
                    args.interface,
                    "policy/release/model_decoder.onnx",
                    str(single_ref_parent),
                    "--obs-config",
                    "policy/release/observation_config.yaml",
                    "--encoder-file",
                    "policy/release/model_encoder.onnx",
                    "--planner-file",
                    "planner/target_vel/V2/planner_sonic.onnx",
                    "--input-type",
                    "keyboard",
                    "--output-type",
                    "zmq",
                    "--disable-crc-check",
                    "--enable-csv-logs",
                    "--logs-dir",
                    str(job_dir / "deploy_csv_logs"),
                    "--target-motion-logfile",
                    str(job_dir / "target_motion.csv"),
                    "--policy-input-logfile",
                    str(job_dir / "policy_input.csv"),
                ],
                cwd=DEPLOY_ROOT,
                env=deploy_env,
                log_path=deploy_log,
            )

            try:
                if not deploy.wait_for("Init Done", args.startup_timeout):
                    raise RuntimeError(f"Native SONIC deploy did not initialize for {job.name}")
                deploy.send("]")
                if not deploy.wait_for("transitioning to CONTROL", 20.0):
                    raise RuntimeError(f"Native SONIC deploy did not enter CONTROL for {job.name}")
                time.sleep(args.control_settle)
                if args.release_before_play:
                    release_trigger.write_text("release\n")
                    time.sleep(args.release_settle)
                play_start = time.time()
                deploy.send("T")
                time.sleep(job.duration + args.tail_time)
                playback_window_path.write_text(
                    json.dumps(
                        {
                            "motion": job.name,
                            "play_start_wall_time": play_start,
                            "play_end_wall_time": play_start + job.duration,
                            "duration": job.duration,
                            "release_before_play": args.release_before_play,
                            "release_settle": args.release_settle,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            finally:
                deploy.terminate()
                if sim.poll() is None:
                    sim.send_signal(signal.SIGINT)
                    try:
                        sim.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        sim.terminate()
                        try:
                            sim.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            sim.kill()

    qpos_log = read_csv_array(sim_log)
    wall = qpos_log[:, 0]
    sim_qpos = qpos_log[:, 2:]
    mask = (wall >= play_start) & (wall <= play_start + job.duration)
    if mask.sum() < 20:
        raise RuntimeError(f"Only {mask.sum()} qpos rows captured during playback for {job.name}")
    actual = extract_actual_29dof_qpos(sim_qpos[mask])

    # Downsample simulator log to the render/reference rate. The official motion
    # CSVs are 40 Hz; the MuJoCo bridge usually logs faster.
    ref_qpos = load_reference_qpos(args.reference_root / job.name)
    ref_fps = (len(ref_qpos) - 1) / job.duration
    render_fps = args.fps if args.fps > 0 else ref_fps
    actual_idx = np.linspace(0, len(actual) - 1, num=min(len(ref_qpos), int(job.duration * render_fps)), dtype=np.int64)
    ref_idx = np.linspace(0, len(ref_qpos) - 1, num=len(actual_idx), dtype=np.int64)
    actual_render = actual[actual_idx]
    ref_render = ref_qpos[ref_idx]

    out_path = out_dir / f"{job.name}_actual_sim_qpos.mp4"
    render_qpos_video(
        job.name,
        ref_render,
        actual_render,
        out_path,
        fps=render_fps,
        width=args.width,
        height=args.height,
        align_mode=args.align_mode,
        contact_markers=args.contact_markers,
        camera_track=args.camera_track,
        no_text_overlay=args.no_text_overlay,
        overlay_same_origin=args.overlay_same_origin,
        ground_align_render=args.ground_align_render,
    )
    metrics = rollout_metrics(ref_render, actual_render, render_fps)
    metrics.update(
        {
            "motion": job.name,
            "video": str(out_path),
            "reference_root": str(args.reference_root),
            "requested_duration_s": float(job.duration),
            "render_fps": float(render_fps),
            "actual_rows_captured": int(mask.sum()),
        }
    )
    return out_path, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=CS348K_ROOT / "results/sonic_actual_sim_examples")
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--motions", nargs="*", default=[job.name for job in DEFAULT_JOBS])
    parser.add_argument("--duration", type=float, default=None, help="Override duration for all motions")
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--drop_after", type=float, default=999999.0)
    parser.add_argument("--release_before_play", action="store_true")
    parser.add_argument("--release_settle", type=float, default=1.0)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--startup_timeout", type=float, default=90.0)
    parser.add_argument("--sim_warmup", type=float, default=4.0)
    parser.add_argument("--control_settle", type=float, default=2.0)
    parser.add_argument("--tail_time", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--align_mode", choices=["initial", "per_frame"], default="initial")
    parser.add_argument("--contact_markers", action="store_true")
    parser.add_argument("--camera_track", action="store_true")
    parser.add_argument("--no_text_overlay", action="store_true", help="Render clean videos with no text burned into frames.")
    parser.add_argument("--overlay_same_origin", action="store_true", help="Overlay reference and physics robot at the same root origin instead of side-by-side.")
    parser.add_argument("--ground_align_render", action="store_true", help="For visualization, align both rendered robots to the floor using foot collision geoms.")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.reference_root = args.reference_root.resolve()
    if args.summary_csv is None:
        args.summary_csv = args.out_dir / "native_tracking_summary.csv"

    default_durations = {job.name: job.duration for job in DEFAULT_JOBS}
    jobs = []
    for name in args.motions:
        if args.duration is not None:
            duration = args.duration
        elif name in default_durations and args.reference_root == DEFAULT_REFERENCE_ROOT.resolve():
            duration = default_durations[name]
        else:
            duration = infer_reference_duration(args.reference_root / name)
        jobs.append(MotionJob(name, duration))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    metrics_rows: list[dict[str, object]] = []
    for job in jobs:
        print(f"[sonic-qpos] running {job.name} for {job.duration:.1f}s")
        path, metrics = run_one(job, args.out_dir, args)
        rendered.append(path)
        metrics_rows.append(metrics)
        print(
            f"[sonic-qpos] wrote {path} | fell={metrics['fell']} "
            f"fall={float(metrics['fall_time_s']):.2f}s rmse={float(metrics['mean_joint_rmse']):.3f}"
        )

    with args.summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"Summary: {args.summary_csv}")

    print("Rendered actual-qpos videos:")
    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()
