"""Create a contact sheet from Humanoid100 video artifacts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "results" / "humanoid100_final_eval" / "before_after_overlay_videos.csv"
DEFAULT_OUT = ROOT / "results" / "humanoid100_final_eval" / "before_after_overlay_contact_sheet.jpg"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def frame_from_video(path: Path, fraction: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frames - 1, int(frames * fraction))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame from {path}")
    return frame


def put(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--video_column", default="comparison_video")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--thumb_width", type=int, default=320)
    parser.add_argument("--frame_fraction", type=float, default=0.35)
    args = parser.parse_args()

    rows = sorted(read_rows(args.index_csv), key=lambda row: row["prompt_id"])
    if not rows:
        raise ValueError(f"No rows in {args.index_csv}")
    thumbs: list[np.ndarray] = []
    thumb_h = None
    for row in rows:
        frame = frame_from_video(Path(row[args.video_column]), args.frame_fraction)
        scale = args.thumb_width / frame.shape[1]
        h = int(round(frame.shape[0] * scale))
        thumb = cv2.resize(frame, (args.thumb_width, h), interpolation=cv2.INTER_AREA)
        thumb[:24, :, :] = (0.55 * thumb[:24, :, :]).astype(np.uint8)
        put(thumb, f"{row['prompt_id']} {row['subcategory']}", 5, 17)
        thumbs.append(thumb)
        thumb_h = h
    assert thumb_h is not None

    cols = args.cols
    rows_count = math.ceil(len(thumbs) / cols)
    sheet = np.zeros((rows_count * thumb_h, cols * args.thumb_width, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        y = (i // cols) * thumb_h
        x = (i % cols) * args.thumb_width
        sheet[y:y + thumb_h, x:x + args.thumb_width] = thumb
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"Wrote {args.out} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
