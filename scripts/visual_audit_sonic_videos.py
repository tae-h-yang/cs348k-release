"""Frame-level visual audit for native SONIC rendered videos.

This intentionally reads the rendered MP4 pixels instead of trusting rollout
CSV metrics. It segments the red actual robot and white reference robot in each
frame, then reports visual flags such as missing bodies, sudden jumps, large
reference/actual separation, and fallen-looking aspect ratios.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def strict_pass(row: dict[str, str]) -> bool:
    if row.get("ref_aware_fell") not in ("", None):
        no_fall = row.get("ref_aware_fell") == "False"
    else:
        no_fall = row.get("fell") == "False"
    return (
        no_fall
        and as_float(row, "mean_joint_rmse", 999.0) <= 0.20
        and as_float(row, "mean_root_xy_error", 999.0) <= 1.5
    )


def motion_from_video(video: Path) -> str:
    name = video.name
    for suffix in ("_actual_sim_qpos.mp4", "_diagnostic_contacts.mp4"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return video.stem


def largest_component(mask: np.ndarray, min_area: int) -> dict[str, float] | None:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    best = None
    best_area = min_area
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= best_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        best = {
            "area": float(area),
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "cx": float(cx),
            "cy": float(cy),
        }
        best_area = area
    return best


def segment_frame(frame: np.ndarray, crop_top: int) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    # Work below the text overlay. OpenCV frames are BGR.
    image = frame.copy()
    image[:crop_top, :, :] = 0
    b, g, r = cv2.split(image)

    red_mask = (r > 95) & (r > 1.35 * g + 12) & (r > 1.35 * b + 12)
    # Reference body is matte white/gray; constrain to low saturation and
    # sufficient value to avoid grid-floor highlights.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    white_mask = (s < 80) & (v > 120) & ~red_mask

    kernel = np.ones((3, 3), dtype=np.uint8)
    red_mask = cv2.morphologyEx(red_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    red = largest_component(red_mask, min_area=35)
    white = largest_component(white_mask, min_area=35)
    return red, white


def audit_video(video: Path, crop_top: int, every_frame: bool) -> dict[str, object]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    diag = math.sqrt(width * width + height * height)
    stride = 1 if every_frame else max(1, frame_count // 150)

    rows = []
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride:
            continue
        red, white = segment_frame(frame, crop_top)
        if red and white:
            sep = math.hypot(red["cx"] - white["cx"], red["cy"] - white["cy"])
            scale = max(red["h"], white["h"], 1.0)
        else:
            sep = float("nan")
            scale = float("nan")
        rows.append({"idx": frame_idx, "red": red, "white": white, "sep": sep, "scale": scale})
    cap.release()

    analyzed = len(rows)
    red_missing = sum(row["red"] is None for row in rows)
    white_missing = sum(row["white"] is None for row in rows)
    both = [row for row in rows if row["red"] and row["white"]]
    sep_body = [float(row["sep"]) / max(float(row["scale"]), 1.0) for row in both]
    sep_px = [float(row["sep"]) for row in both]
    actual_aspect = [float(row["red"]["h"]) / max(float(row["red"]["w"]), 1.0) for row in rows if row["red"]]
    ref_aspect = [float(row["white"]["h"]) / max(float(row["white"]["w"]), 1.0) for row in rows if row["white"]]
    flat_mismatch = []
    for row in rows:
        if row["red"] is None or row["white"] is None:
            continue
        actual_ratio = float(row["red"]["h"]) / max(float(row["red"]["w"]), 1.0)
        ref_ratio = float(row["white"]["h"]) / max(float(row["white"]["w"]), 1.0)
        flat_mismatch.append(actual_ratio < 1.05 and ref_ratio >= 1.05)

    jumps = []
    prev = None
    for row in rows:
        red = row["red"]
        if red is None:
            prev = None
            continue
        if prev is not None:
            jumps.append(math.hypot(float(red["cx"]) - float(prev["cx"]), float(red["cy"]) - float(prev["cy"])))
        prev = red

    def pct(condition_count: int) -> float:
        return 100.0 * condition_count / max(analyzed, 1)

    large_sep_count = sum(value > 1.55 for value in sep_body)
    very_large_sep_count = sum(value > 2.20 for value in sep_body)
    actual_flat_count = sum(value < 1.05 for value in actual_aspect)
    ref_flat_count = sum(value < 1.05 for value in ref_aspect)
    flat_mismatch_count = sum(flat_mismatch)
    max_jump = max(jumps) if jumps else 0.0
    jump_thresh = max(38.0, 0.060 * diag)
    jump_count = sum(value > jump_thresh for value in jumps)

    red_missing_pct = pct(red_missing)
    white_missing_pct = pct(white_missing)
    large_sep_pct = pct(large_sep_count)
    very_large_sep_pct = pct(very_large_sep_count)
    actual_flat_pct = pct(actual_flat_count)
    ref_flat_pct = pct(ref_flat_count)
    actual_flat_mismatch_pct = 100.0 * flat_mismatch_count / max(len(flat_mismatch), 1)
    jump_pct = 100.0 * jump_count / max(len(jumps), 1)

    fail_reasons = []
    warn_reasons = []
    if red_missing_pct > 20.0:
        fail_reasons.append("actual_missing")
    if white_missing_pct > 20.0:
        fail_reasons.append("reference_missing")
    if very_large_sep_pct > 20.0:
        fail_reasons.append("very_large_visual_separation")
    if actual_flat_mismatch_pct > 20.0:
        fail_reasons.append("actual_flat_or_fallen")
    if jump_pct > 5.0 or max_jump > 120.0:
        fail_reasons.append("actual_visual_jump")
    if not fail_reasons and large_sep_pct > 20.0:
        warn_reasons.append("large_visual_separation")
    if not fail_reasons and ref_flat_pct > 20.0:
        warn_reasons.append("reference_flat_low_posture")

    status = "fail" if fail_reasons else "warn" if warn_reasons else "pass"
    return {
        "video": str(video),
        "motion": motion_from_video(video),
        "frames_total": frame_count,
        "frames_analyzed": analyzed,
        "fps": fps,
        "width": width,
        "height": height,
        "visual_status": status,
        "visual_reasons": ";".join(fail_reasons or warn_reasons),
        "red_missing_pct": red_missing_pct,
        "white_missing_pct": white_missing_pct,
        "mean_sep_body": float(np.mean(sep_body)) if sep_body else float("nan"),
        "p95_sep_body": float(np.percentile(sep_body, 95)) if sep_body else float("nan"),
        "p95_sep_px": float(np.percentile(sep_px, 95)) if sep_px else float("nan"),
        "large_sep_pct": large_sep_pct,
        "very_large_sep_pct": very_large_sep_pct,
        "actual_flat_pct": actual_flat_pct,
        "reference_flat_pct": ref_flat_pct,
        "actual_flat_mismatch_pct": actual_flat_mismatch_pct,
        "actual_jump_pct": jump_pct,
        "max_actual_jump_px": max_jump,
    }


def copy_videos(rows: list[dict[str, object]], out_dir: Path, limit: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows[:limit]:
        src = Path(str(row["video"]))
        if src.exists():
            dst = out_dir / src.name
            if not dst.exists():
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)


def contact_sheet(video_dir: Path, out: Path, limit: int) -> None:
    if not video_dir.exists() or not any(video_dir.glob("*.mp4")):
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_video_contact_sheet.py"),
            "--video_dir",
            str(video_dir),
            "--out",
            str(out),
            "--samples",
            "6",
            "--thumb_width",
            "180",
            "--limit",
            str(limit),
        ],
        cwd=ROOT,
        check=True,
    )


def write_report(out_dir: Path, rows: list[dict[str, object]], batch_rows: dict[str, dict[str, str]]) -> None:
    completed = len(rows)
    status_counts = {status: sum(row["visual_status"] == status for row in rows) for status in ("pass", "warn", "fail")}
    strict_rows = [row for row in rows if strict_pass(batch_rows.get(str(row["motion"]), {}))]
    strict_fail = [row for row in strict_rows if row["visual_status"] == "fail"]
    native_fail_rows = [row for row in rows if batch_rows.get(str(row["motion"]), {}).get("fell") == "True"]
    native_fail_visual_pass = [row for row in native_fail_rows if row["visual_status"] == "pass"]

    lines = [
        "# Visual Frame Audit",
        "",
        "This audit reads rendered MP4 pixels directly and segments the visible red",
        "actual robot plus white reference robot in every analyzed frame.",
        "",
        f"- videos analyzed: {completed}",
        f"- visual pass: {status_counts['pass']}",
        f"- visual warn: {status_counts['warn']}",
        f"- visual fail: {status_counts['fail']}",
        f"- native strict-pass videos with visual fail flags: {len(strict_fail)}/{len(strict_rows)}",
        f"- native fallen videos with visual pass flags: {len(native_fail_visual_pass)}/{len(native_fail_rows)}",
        "",
        "## Worst Visual Separations",
        "",
        "| motion | status | p95 sep/body | large sep % | reasons | native fell | native strict |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    worst = sorted(rows, key=lambda row: float(row["p95_sep_body"]), reverse=True)[:25]
    for row in worst:
        native = batch_rows.get(str(row["motion"]), {})
        lines.append(
            f"| `{row['motion']}` | {row['visual_status']} | {float(row['p95_sep_body']):.2f} | "
            f"{float(row['large_sep_pct']):.1f} | {row['visual_reasons']} | "
            f"{native.get('fell', '')} | {'yes' if strict_pass(native) else 'no'} |"
        )
    lines += [
        "",
        "## Review Folders",
        "",
        "- `visual_fail_videos/`: strongest visual fail flags.",
        "- `visual_warn_videos/`: largest warn-only clips.",
        "- `strict_but_visual_fail_videos/`: highest-priority contradictions.",
        "- `*_contact_sheet.jpg`: frame-sampled sheets for rapid human/VLM review.",
        "",
    ]
    (out_dir / "visual_frame_audit.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=Path, required=True)
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument("--glob", default="*_actual_sim_qpos.mp4")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--crop_top", type=int, default=88)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sampled", action="store_true", help="Analyze a dense sample instead of every frame.")
    parser.add_argument("--sheet_limit", type=int, default=60)
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    out_dir = args.out_dir or (batch_dir / "visual_frame_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_rows = {row["motion"]: row for row in read_rows(batch_dir / "batch_summary.csv") if row.get("status") == "completed"}
    if args.video_dir:
        videos = sorted(args.video_dir.resolve().glob(args.glob))
    else:
        videos = [Path(row["video"]) for row in batch_rows.values() if row.get("video") and Path(row["video"]).exists()]
    videos = sorted(videos)
    if args.limit:
        videos = videos[: args.limit]

    rows = []
    for idx, video in enumerate(videos, start=1):
        row = audit_video(video, crop_top=args.crop_top, every_frame=not args.sampled)
        native = batch_rows.get(str(row["motion"]), {})
        row["native_fell"] = native.get("fell", "")
        row["native_strict_pass"] = "__YES__" if strict_pass(native) else "__NO__"
        row["native_mean_joint_rmse"] = native.get("mean_joint_rmse", "")
        rows.append(row)
        if idx == 1 or idx % 25 == 0:
            print(f"[visual-audit] {idx}/{len(videos)} {row['motion']} {row['visual_status']}")

    write_rows(out_dir / "visual_frame_audit.csv", rows)
    fail_rows = sorted([row for row in rows if row["visual_status"] == "fail"], key=lambda row: float(row["p95_sep_body"]), reverse=True)
    warn_rows = sorted([row for row in rows if row["visual_status"] == "warn"], key=lambda row: float(row["p95_sep_body"]), reverse=True)
    strict_fail_rows = [row for row in fail_rows if row["native_strict_pass"] == "__YES__"]
    copy_videos(fail_rows, out_dir / "visual_fail_videos", args.sheet_limit)
    copy_videos(warn_rows, out_dir / "visual_warn_videos", args.sheet_limit)
    copy_videos(strict_fail_rows, out_dir / "strict_but_visual_fail_videos", args.sheet_limit)
    contact_sheet(out_dir / "visual_fail_videos", out_dir / "visual_fail_contact_sheet.jpg", args.sheet_limit)
    contact_sheet(out_dir / "visual_warn_videos", out_dir / "visual_warn_contact_sheet.jpg", args.sheet_limit)
    contact_sheet(out_dir / "strict_but_visual_fail_videos", out_dir / "strict_but_visual_fail_contact_sheet.jpg", args.sheet_limit)
    write_report(out_dir, rows, batch_rows)
    print(f"Wrote visual audit to {out_dir}")


if __name__ == "__main__":
    main()
