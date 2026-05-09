"""Batch-render official SONIC native sim2sim example videos.

This launches the official MuJoCo sim bridge and native C++ deploy binary,
starts control, then sends keyboard commands to play selected reference
motions while recording the native `g1_debug` stream.
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


CS348K_ROOT = Path(__file__).resolve().parents[1]
GROOT_ROOT = Path("/home/rewardai/repos/GR00T-WholeBodyControl")
DEPLOY_ROOT = GROOT_ROOT / "gear_sonic_deploy"
SIM_PYTHON = GROOT_ROOT / ".venv_sim/bin/python"


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
    index: int
    name: str
    duration: float


MOTION_JOBS = [
    MotionJob(0, "forward_lunge_R_001__A359_M", 10.0),
    MotionJob(1, "walking_quip_360_R_002__A428", 11.0),
    MotionJob(2, "squat_001__A359", 10.0),
    MotionJob(4, "neutral_kick_R_001__A543", 6.0),
    MotionJob(7, "dance_in_da_party_001__A464", 12.0),
    MotionJob(8, "macarena_001__A545_M", 16.0),
]


class PtyProcess:
    def __init__(self, cmd: list[str], cwd: Path, env: dict[str, str]):
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

    def read_available(self, timeout: float = 0.0) -> str:
        chunks = []
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
            if not data:
                break
            self.log.extend(data)
            chunks.append(data.decode("utf-8", errors="replace"))
            if timeout and time.time() >= end:
                break
        return "".join(chunks)

    def wait_for(self, needle: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        text = self.log.decode("utf-8", errors="replace")
        while time.time() < deadline:
            if needle in text:
                return True
            text += self.read_available(timeout=0.25)
            if self.proc.poll() is not None:
                return needle in text
        return False

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
        try:
            os.close(self.master_fd)
        except OSError:
            pass


def run_recorder(job: MotionJob, out_dir: Path) -> Path:
    out_path = out_dir / f"{job.index:02d}_{job.name}.mp4"
    cmd = [
        sys.executable,
        str(CS348K_ROOT / "scripts/record_sonic_realtime_debug.py"),
        "--duration",
        str(job.duration),
        "--label",
        job.name,
        "--out",
        str(out_path),
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    subprocess.run(cmd, cwd=CS348K_ROOT, env=env, check=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=CS348K_ROOT / "results/sonic_native_examples")
    parser.add_argument("--startup_timeout", type=float, default=90.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"

    sim = subprocess.Popen(
        [
            str(SIM_PYTHON),
            str(CS348K_ROOT / "scripts/run_sonic_sim_autodrop.py"),
            "--drop_after",
            "999999",
            "--interface",
            "lo",
        ],
        cwd=CS348K_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    time.sleep(4.0)

    deploy_env = env.copy()
    deploy_env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    deploy = PtyProcess(
        [
            str(DEPLOY_ROOT / "target/release/g1_deploy_onnx_ref"),
            "lo",
            "policy/release/model_decoder.onnx",
            "reference/example/",
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
            str(args.out_dir / "logs"),
            "--target-motion-logfile",
            str(args.out_dir / "target_motion.csv"),
            "--policy-input-logfile",
            str(args.out_dir / "policy_input.csv"),
        ],
        cwd=DEPLOY_ROOT,
        env=deploy_env,
    )

    rendered: list[Path] = []
    try:
        if not deploy.wait_for("Init Done", args.startup_timeout):
            raise RuntimeError("Native SONIC deploy did not finish initialization")
        deploy.send("]")
        if not deploy.wait_for("transitioning to CONTROL", 20.0):
            raise RuntimeError("Native SONIC deploy did not enter CONTROL state")
        time.sleep(2.0)

        current_index = 0
        for job in MOTION_JOBS:
            if job.index < current_index:
                raise RuntimeError("Motion jobs must be sorted by increasing index")
            for _ in range(job.index - current_index):
                deploy.send("N")
                time.sleep(0.35)
            current_index = job.index
            time.sleep(0.5)
            recorder = subprocess.Popen(
                [
                    sys.executable,
                    str(CS348K_ROOT / "scripts/record_sonic_realtime_debug.py"),
                    "--duration",
                    str(job.duration),
                    "--label",
                    job.name,
                    "--out",
                    str(args.out_dir / f"{job.index:02d}_{job.name}.mp4"),
                ],
                cwd=CS348K_ROOT,
                env={**env, "MUJOCO_GL": "egl"},
            )
            time.sleep(0.8)
            deploy.send("T")
            if recorder.wait(timeout=job.duration + 20.0) != 0:
                raise RuntimeError(f"Recorder failed for {job.name}")
            rendered.append(args.out_dir / f"{job.index:02d}_{job.name}.mp4")
            time.sleep(1.0)

        log_path = args.out_dir / "deploy_output.log"
        log_path.write_bytes(deploy.log)
        print("Rendered videos:")
        for path in rendered:
            print(path)
        print(f"Deploy log: {log_path}")
    finally:
        deploy.terminate()
        if sim.poll() is None:
            sim.send_signal(signal.SIGINT)
            try:
                sim.wait(timeout=5)
            except subprocess.TimeoutExpired:
                sim.terminate()


if __name__ == "__main__":
    main()
