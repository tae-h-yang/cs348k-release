#!/usr/bin/env python3
"""Generate the Humanoid100 prompt suite with Kimodo-G1.

Kimodo's G1 model exports MuJoCo qpos CSV files with shape (T, 36), which is
the same qpos representation used by the MotionBricks/SONIC evaluation stack.
This runner is designed to be resumable: completed CSV/NPY pairs are skipped,
and a manifest records both successes and blockers.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_CSV = ROOT / "configs" / "humanoid_robotics_100_prompts.csv"
DEFAULT_OUT = ROOT / "results" / "kimodo_humanoid100_g1"
DEFAULT_DATA = ROOT / "data" / "kimodo_humanoid100_g1"
DEFAULT_KIMODO_BIN = ROOT / ".venv" / "kimodo" / "bin" / "kimodo_gen"
REQUIRED_LLAMA_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def hf_token_present() -> bool:
    for path in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if path.exists() and path.read_text().strip():
            return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def required_llama_file_access() -> tuple[bool, str]:
    """Kimodo's released LLM2Vec encoder needs gated Llama-3 files.

    A token can be syntactically valid while the account is still awaiting
    approval for the gated base model. Checking a small file gives a reliable
    go/no-go signal before launching a long 100-prompt generation run.
    """
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(REQUIRED_LLAMA_REPO, "config.json")
    except Exception as exc:
        return False, str(exc).split("\n")[0]
    return True, "ok"


def validate_qpos_csv(csv_path: Path, npy_path: Path) -> tuple[int, int]:
    arr = np.loadtxt(csv_path, delimiter=",")
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 36:
        raise ValueError(f"{csv_path}: expected Kimodo-G1 CSV shape (T, 36), got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"{csv_path}: expected at least two frames, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{csv_path}: qpos contains non-finite values")
    quat_norm = np.linalg.norm(arr[:, 3:7], axis=1)
    if np.any(quat_norm < 1e-6):
        raise ValueError(f"{csv_path}: root quaternion contains near-zero norm")
    arr = arr.copy()
    arr[:, 3:7] /= quat_norm[:, None]
    np.save(npy_path, arr)
    return int(arr.shape[0]), int(arr.shape[1])


def write_blocked_report(out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Kimodo Humanoid100 Generation",
        "",
        "Generation is currently blocked because no Hugging Face token is available.",
        "Kimodo's local text encoder depends on gated Llama-3 / LLM2Vec weights.",
        "",
        "Once a token is configured, rerun:",
        "",
        "```bash",
        " ".join(shlex.quote(x) for x in [
            str(args.kimodo_bin),
            "<prompt>",
            "--model",
            args.model,
            "--duration",
            str(args.duration),
            "--diffusion_steps",
            str(args.diffusion_steps),
            "--output",
            str(args.data_dir / "example"),
        ]),
        "```",
        "",
        "Or run this resumable suite script again.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def write_llama_access_report(out_dir: Path, message: str) -> None:
    lines = [
        "# Kimodo Humanoid100 Generation",
        "",
        "Generation is currently blocked because the Hugging Face account/token",
        f"cannot download `{REQUIRED_LLAMA_REPO}` files yet.",
        "",
        "Kimodo-G1 uses LLM2Vec for prompt embeddings, and that encoder depends",
        "on Meta's gated Llama-3 base model. A valid token is not enough; the",
        "Hugging Face account must also be approved on the model page.",
        "",
        "Action:",
        f"1. Open https://huggingface.co/{REQUIRED_LLAMA_REPO}",
        "2. Request/accept access with the same account used by `hf auth whoami`.",
        "3. After approval, rerun `bash scripts/run_kimodo_humanoid100_full_pipeline.sh`.",
        "",
        f"Last access check: `{message}`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def log_tail(path: Path, max_chars: int = 1200) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_csv", type=Path, default=PROMPTS_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--kimodo_bin", type=Path, default=DEFAULT_KIMODO_BIN)
    parser.add_argument("--model", default="Kimodo-G1-RP-v1")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=348100)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--timeout_s", type=int, default=1800)
    parser.add_argument("--max_runtime_s", type=int, default=0, help="0 means no wall-clock limit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_missing_token", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.kimodo_bin.exists():
        raise FileNotFoundError(f"Kimodo CLI not found: {args.kimodo_bin}. Run scripts/dual_track_kimodo_motionbricks_loop.sh first.")

    if not hf_token_present() and not args.allow_missing_token:
        manifest = [{
            "status": "blocked_missing_hf_token",
            "model": args.model,
            "message": "No Hugging Face token found; Kimodo text encoder cannot load gated Llama/LLM2Vec weights.",
        }]
        write_csv(args.out_dir / "manifest.csv", manifest)
        write_blocked_report(args.out_dir, args)
        print("Kimodo generation blocked: no Hugging Face token found.")
        return 0

    if not args.allow_missing_token:
        llama_ok, llama_message = required_llama_file_access()
        if not llama_ok:
            manifest = [{
                "status": "blocked_llama_access",
                "model": args.model,
                "required_repo": REQUIRED_LLAMA_REPO,
                "message": llama_message,
            }]
            write_csv(args.out_dir / "manifest.csv", manifest)
            write_llama_access_report(args.out_dir, llama_message)
            print(f"Kimodo generation blocked: cannot access {REQUIRED_LLAMA_REPO} files yet.")
            return 0

    rows = read_rows(args.prompts_csv)[args.start : args.start + args.limit]
    manifest: list[dict[str, object]] = []
    commands: list[str] = []
    t0 = time.time()
    for idx, row in enumerate(rows, start=args.start + 1):
        if args.max_runtime_s and time.time() - t0 > args.max_runtime_s:
            print(f"Reached max runtime {args.max_runtime_s}s; stopping after {len(manifest)} rows.")
            break

        stem = args.data_dir / f"{row['prompt_id']}_{row['subcategory']}"
        csv_path = stem.with_suffix(".csv")
        npz_path = stem.with_suffix(".npz")
        npy_path = stem.with_suffix(".npy")
        log_path = log_dir / f"{row['prompt_id']}_{row['subcategory']}.log"
        command = [
            str(args.kimodo_bin),
            row["prompt_text"],
            "--model",
            args.model,
            "--duration",
            str(args.duration),
            "--num_samples",
            str(args.num_samples),
            "--diffusion_steps",
            str(args.diffusion_steps),
            "--seed",
            str(args.seed_base + idx),
            "--output",
            str(stem),
        ]
        commands.append(" ".join(shlex.quote(x) for x in command))

        base = {
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "prompt_text": row["prompt_text"],
            "success_criteria": row["success_criteria"],
            "expected_primary_contacts": row["expected_primary_contacts"],
            "expected_root_motion": row["expected_root_motion"],
            "expected_arm_role": row["expected_arm_role"],
            "hardness": row["hardness"],
            "model": args.model,
            "duration_s": args.duration,
            "diffusion_steps": args.diffusion_steps,
            "seed": args.seed_base + idx,
            "csv_path": str(csv_path),
            "npz_path": str(npz_path),
            "qpos_npy_path": str(npy_path),
            "log_path": str(log_path),
        }

        if csv_path.exists() and npy_path.exists() and not args.overwrite:
            frames, dims = validate_qpos_csv(csv_path, npy_path)
            manifest.append({**base, "status": "skipped_existing", "frames": frames, "dims": dims})
            continue

        env = os.environ.copy()
        env.setdefault("TEXT_ENCODER_DEVICE", "cpu")
        env.setdefault("PYTHONUNBUFFERED", "1")
        print(f"[{idx:03d}] Kimodo-G1 {row['prompt_id']} {row['subcategory']}")
        with log_path.open("w") as log:
            log.write("$ " + " ".join(shlex.quote(x) for x in command) + "\n\n")
            try:
                proc = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                manifest.append({**base, "status": "timeout", "returncode": "timeout"})
                write_csv(args.out_dir / "manifest.csv", manifest)
                continue

        if proc.returncode != 0:
            manifest.append({
                **base,
                "status": "failed",
                "returncode": proc.returncode,
                "error_tail": log_tail(log_path),
            })
            write_csv(args.out_dir / "manifest.csv", manifest)
            continue

        try:
            frames, dims = validate_qpos_csv(csv_path, npy_path)
        except Exception as exc:
            manifest.append({**base, "status": "invalid_output", "returncode": proc.returncode, "error": str(exc)})
        else:
            manifest.append({**base, "status": "success", "returncode": proc.returncode, "frames": frames, "dims": dims})
        write_csv(args.out_dir / "manifest.csv", manifest)

    (args.out_dir / "commands.sh").write_text("\n".join(commands) + "\n")
    write_csv(args.out_dir / "manifest.csv", manifest)
    successes = sum(row.get("status") in {"success", "skipped_existing"} for row in manifest)
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Kimodo Humanoid100 Generation",
                "",
                f"- Prompts attempted in this run: {len(manifest)}",
                f"- Successful/skipped qpos exports: {successes}",
                f"- Model: `{args.model}`",
                f"- Duration per prompt: `{args.duration}` seconds",
                f"- Diffusion steps: `{args.diffusion_steps}`",
                f"- Manifest: `{args.out_dir / 'manifest.csv'}`",
                f"- Qpos NPY directory: `{args.data_dir}`",
                "",
            ]
        )
    )
    print(f"Wrote manifest: {args.out_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
