"""Run many native SONIC release checks with resume and a wall-clock cap.

This wrapper calls `render_sonic_actual_sim_examples.py` one reference at a
time. That is slower than a monolithic run, but much safer for long jobs: every
completed motion gets its own summary CSV, the aggregate summary is rewritten
after each attempt, and rerunning the command skips completed motions.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = ROOT / "results" / "sonic_references_210_fixed"
DEFAULT_TRACKING_CSV = ROOT / "results" / "sonic_policy_mujoco_tracking_210_fixed.csv"


def parse_mode(name: str) -> str:
    selector = selector_prefix(name)
    if selector:
        name = name.removeprefix(selector + "_")
    if "_seed" in name:
        return name.rsplit("_seed", 1)[0]
    if "_K" in name:
        return name.rsplit("_K", 1)[0]
    return name


def category(name: str) -> str:
    mode = parse_mode(name)
    if "crawling" in mode:
        return "low_posture_crawling"
    if mode == "idle":
        return "idle"
    return "upright"


def load_tracking_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["reference"]: row for row in csv.DictReader(f)}


def build_broad100(reference_root: Path, tracking_csv: Path, limit: int) -> list[str]:
    names = sorted(p.name for p in reference_root.iterdir() if (p / "joint_pos.csv").exists())
    tracking = load_tracking_rows(tracking_csv)

    def approx_key(name: str) -> tuple[float, float, str]:
        row = tracking.get(name, {})
        seconds = float(row.get("track_seconds", 0.0) or 0.0)
        rmse = float(row.get("mean_tracking_rmse", 999.0) or 999.0)
        return (-seconds, rmse, name)

    upright = [n for n in names if category(n) == "upright"]
    idle = [n for n in names if category(n) == "idle"]
    crawling = [n for n in names if category(n) == "low_posture_crawling"]

    upright_sorted = sorted(upright, key=approx_key)
    idle_sorted = sorted(idle, key=approx_key)
    crawling_sorted = sorted(crawling, key=approx_key)

    controls_each = min(8, max(0, limit // 12))
    candidates = (
        upright_sorted[: max(0, limit - 2 * controls_each)]
        + idle_sorted[:controls_each]
        + crawling_sorted[:controls_each]
    )
    seen: set[str] = set()
    out: list[str] = []
    for name in candidates:
        if name not in seen:
            out.append(name)
            seen.add(name)
        if len(out) >= limit:
            break
    return out


def build_all(reference_root: Path, limit: int) -> list[str]:
    names = sorted(p.name for p in reference_root.iterdir() if (p / "joint_pos.csv").exists())
    return names[:limit]


def selector_prefix(name: str) -> str:
    for prefix in (
        "baseline_k0",
        "lowest_id_risk",
        "best_precontroller",
        "gated_precontroller",
        "learned_acceptance",
        "hybrid_acceptance",
    ):
        if name.startswith(prefix + "_"):
            return prefix
    return ""


def interleave_by_identity(names: list[str]) -> list[str]:
    selectors = ("baseline_k0", "best_precontroller", "gated_precontroller", "lowest_id_risk")
    grouped = group_by_identity(names)
    interleaved: list[str] = []
    for identity in sorted(grouped):
        for selector in selectors:
            if selector in grouped[identity]:
                interleaved.append(grouped[identity][selector])
        for selector in sorted(set(grouped[identity]) - set(selectors)):
            interleaved.append(grouped[identity][selector])
    return interleaved


def group_by_identity(names: list[str]) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    passthrough: list[str] = []
    for name in names:
        selector = selector_prefix(name)
        if not selector:
            passthrough.append(name)
            continue
        identity = name.removeprefix(selector + "_")
        if "_cand" in identity:
            identity = identity.rsplit("_cand", 1)[0]
        grouped.setdefault(identity, {})[selector] = name

    for name in passthrough:
        grouped.setdefault(name, {})[""] = name
    return grouped


def interleave_by_mode_identity(names: list[str]) -> list[str]:
    selectors = ("baseline_k0", "best_precontroller", "gated_precontroller", "lowest_id_risk", "")
    grouped = group_by_identity(names)
    by_mode: dict[str, list[str]] = {}
    for identity in sorted(grouped):
        by_mode.setdefault(parse_mode(identity), []).append(identity)

    interleaved: list[str] = []
    while any(by_mode.values()):
        for mode in sorted(by_mode):
            if not by_mode[mode]:
                continue
            identity = by_mode[mode].pop(0)
            for selector in selectors:
                if selector in grouped[identity]:
                    interleaved.append(grouped[identity][selector])
            for selector in sorted(set(grouped[identity]) - set(selectors)):
                interleaved.append(grouped[identity][selector])
    return interleaved


def read_one_summary(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}, got {len(rows)}")
    return rows[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], started: float, cap_seconds: float) -> None:
    completed = [r for r in rows if r.get("status") == "completed"]
    passed = [r for r in completed if str(r.get("fell")) == "False"]
    upright = [r for r in completed if r.get("category") == "upright"]
    upright_pass = [r for r in upright if str(r.get("fell")) == "False"]
    elapsed = time.time() - started
    mean_rmse = (
        sum(float(r["mean_joint_rmse"]) for r in completed if r.get("mean_joint_rmse") not in ("", None))
        / max(1, len(completed))
    )
    path.write_text(
        "\n".join(
            [
                "# Native SONIC Overnight Batch",
                "",
                f"- elapsed_hours: {elapsed / 3600:.2f}",
                f"- cap_hours: {cap_seconds / 3600:.2f}",
                f"- attempts_recorded: {len(rows)}",
                f"- completed: {len(completed)}",
                f"- pass_count_root_threshold: {len(passed)}",
                f"- upright_completed: {len(upright)}",
                f"- upright_pass_count: {len(upright_pass)}",
                f"- mean_joint_rmse_completed: {mean_rmse:.3f}",
                "",
                "See `batch_summary.csv` for per-motion rows and MP4 paths.",
                "",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--tracking_csv", type=Path, default=DEFAULT_TRACKING_CSV)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "sonic_native_release_overnight")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--strategy",
        choices=("broad100", "all"),
        default="broad100",
        help="Candidate selection policy when --motions is not provided.",
    )
    parser.add_argument(
        "--order",
        choices=("sorted", "interleaved", "mode_interleaved"),
        default="sorted",
        help="Run order for the selected candidate list.",
    )
    parser.add_argument("--max_hours", type=float, default=8.0)
    parser.add_argument("--motions", nargs="*", default=[])
    parser.add_argument("--motions_file", type=Path, default=None, help="Optional newline-delimited motion list.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--release_settle", type=float, default=1.0)
    parser.add_argument("--startup_timeout", type=float, default=90.0)
    parser.add_argument("--contact_markers", action="store_true")
    parser.add_argument("--camera_track", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.reference_root = args.reference_root.resolve()
    args.out_dir = args.out_dir.resolve()
    summaries_dir = args.out_dir / "per_motion_summaries"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    file_motions: list[str] = []
    if args.motions_file is not None:
        file_motions = [
            line.strip()
            for line in args.motions_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if file_motions:
        candidates = file_motions[: args.limit]
    elif args.motions:
        candidates = args.motions[: args.limit]
    elif args.strategy == "all":
        candidates = build_all(args.reference_root, args.limit)
    else:
        candidates = build_broad100(args.reference_root, args.tracking_csv, args.limit)
    if args.order == "interleaved":
        candidates = interleave_by_identity(candidates)
        candidates = candidates[: args.limit]
    elif args.order == "mode_interleaved":
        candidates = interleave_by_mode_identity(candidates)
        candidates = candidates[: args.limit]

    candidate_csv = args.out_dir / "candidate_list.csv"
    write_rows(
        candidate_csv,
        [
            {
                "index": i + 1,
                "motion": name,
                "mode": parse_mode(name),
                "category": category(name),
            }
            for i, name in enumerate(candidates)
        ],
    )

    started = time.time()
    cap_seconds = args.max_hours * 3600.0
    batch_rows: list[dict[str, object]] = []
    batch_csv = args.out_dir / "batch_summary.csv"
    batch_md = args.out_dir / "batch_summary.md"

    if args.resume and batch_csv.exists():
        with batch_csv.open(newline="") as f:
            batch_rows = list(csv.DictReader(f))
        for row in batch_rows:
            motion = str(row.get("motion", ""))
            row["mode"] = parse_mode(motion)
            row["category"] = category(motion)
    recorded_names = {str(r["motion"]) for r in batch_rows if r.get("motion")}
    if args.resume:
        for i, motion in enumerate(candidates, start=1):
            if motion in recorded_names:
                continue
            summary_csv = summaries_dir / f"{motion}.csv"
            if not summary_csv.exists():
                continue
            row = dict(read_one_summary(summary_csv))
            row["status"] = "completed"
            row["index"] = i
            row["mode"] = parse_mode(motion)
            row["category"] = category(motion)
            row["attempt_seconds"] = ""
            batch_rows.append(row)
            recorded_names.add(motion)
        if batch_rows:
            write_rows(batch_csv, batch_rows)
    completed_names = {str(r["motion"]) for r in batch_rows if r.get("status") == "completed"}

    for i, motion in enumerate(candidates, start=1):
        if time.time() - started >= cap_seconds:
            print(f"[batch] stopping at wall-clock cap before {motion}")
            break
        if motion in completed_names:
            print(f"[batch] skip completed {i}/{len(candidates)} {motion}")
            continue

        summary_csv = summaries_dir / f"{motion}.csv"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "render_sonic_actual_sim_examples.py"),
            "--reference_root",
            str(args.reference_root),
            "--out_dir",
            str(args.out_dir),
            "--summary_csv",
            str(summary_csv),
            "--motions",
            motion,
            "--release_before_play",
            "--release_settle",
            str(args.release_settle),
            "--align_mode",
            "initial",
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--startup_timeout",
            str(args.startup_timeout),
        ]
        if args.contact_markers:
            cmd.append("--contact_markers")
        if args.camera_track:
            cmd.append("--camera_track")
        env = os.environ.copy()
        env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
        print(f"[batch] run {i}/{len(candidates)} {motion}")
        attempt_start = time.time()
        result = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
        elapsed = time.time() - attempt_start
        row: dict[str, object]
        if result.returncode == 0 and summary_csv.exists():
            row = dict(read_one_summary(summary_csv))
            row["status"] = "completed"
        else:
            row = {
                "motion": motion,
                "status": "failed",
                "returncode": result.returncode,
                "fell": "",
                "fall_time_s": "",
                "mean_joint_rmse": "",
                "min_root_z": "",
                "video": "",
            }
        row["index"] = i
        row["mode"] = parse_mode(motion)
        row["category"] = category(motion)
        row["attempt_seconds"] = f"{elapsed:.2f}"
        batch_rows.append(row)
        write_rows(batch_csv, batch_rows)
        write_markdown(batch_md, batch_rows, started, cap_seconds)

    write_rows(batch_csv, batch_rows)
    write_markdown(batch_md, batch_rows, started, cap_seconds)
    print(f"[batch] wrote {batch_csv}")
    print(f"[batch] wrote {batch_md}")
    print(f"[batch] candidates {candidate_csv}")


if __name__ == "__main__":
    main()
