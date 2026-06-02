#!/usr/bin/env python3
"""Recompute SONIC rollout metrics from saved simulator qpos logs.

This is useful after metric fixes: it avoids rerendering MP4s and reuses each
motion's `sim_qpos.csv` plus the exported reference CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from render_sonic_actual_sim_examples import (
    extract_actual_29dof_qpos,
    infer_reference_duration,
    load_reference_qpos,
    read_csv_array,
    rollout_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
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


def recompute_one(batch_dir: Path, reference_root: Path, row: dict[str, str]) -> dict[str, object]:
    motion = row["motion"]
    job_dir = batch_dir / motion
    sim_log = job_dir / "sim_qpos.csv"
    window_path = job_dir / "playback_window.json"
    if not sim_log.exists() or not window_path.exists():
        out = dict(row)
        out["ref_aware_recompute_status"] = "missing_logs"
        return out

    window = json.loads(window_path.read_text())
    play_start = float(window["play_start_wall_time"])
    duration = float(window.get("duration") or row.get("requested_duration_s") or infer_reference_duration(reference_root / motion))

    qpos_log = read_csv_array(sim_log)
    wall = qpos_log[:, 0]
    sim_qpos = qpos_log[:, 2:]
    mask = (wall >= play_start) & (wall <= play_start + duration)
    if mask.sum() < 20:
        out = dict(row)
        out["ref_aware_recompute_status"] = f"too_few_rows:{int(mask.sum())}"
        return out

    actual = extract_actual_29dof_qpos(sim_qpos[mask])
    ref_qpos = load_reference_qpos(reference_root / motion)
    render_fps = float(row.get("render_fps") or 30.0)
    actual_idx = np.linspace(0, len(actual) - 1, num=min(len(ref_qpos), int(duration * render_fps)), dtype=np.int64)
    ref_idx = np.linspace(0, len(ref_qpos) - 1, num=len(actual_idx), dtype=np.int64)
    metrics = rollout_metrics(ref_qpos[ref_idx], actual[actual_idx], render_fps)

    out = dict(row)
    for key, value in metrics.items():
        out[key] = value
    out["ref_aware_recompute_status"] = "ok"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, default=None)
    parser.add_argument("--summary_csv", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    batch_dir = args.batch_dir.resolve()
    summary_csv = args.summary_csv or batch_dir / "batch_summary.csv"
    rows = read_rows(summary_csv)
    if args.reference_root is None:
        reference_roots = {row.get("reference_root", "") for row in rows if row.get("reference_root")}
        if len(reference_roots) != 1:
            raise ValueError(f"Please pass --reference_root; found roots: {sorted(reference_roots)}")
        reference_root = Path(next(iter(reference_roots))).resolve()
    else:
        reference_root = args.reference_root.resolve()
    out_csv = args.out_csv or batch_dir / "batch_summary_ref_aware.csv"

    out = [recompute_one(batch_dir, reference_root, row) for row in rows]
    write_rows(out_csv, out)
    ok = sum(row.get("ref_aware_recompute_status") == "ok" for row in out)
    no_fall = sum(str(row.get("ref_aware_fell", "")).lower() == "false" for row in out)
    print(f"Wrote {out_csv}")
    print(f"Recomputed {ok}/{len(out)} rows; ref-aware no-fall {no_fall}/{len(out)}")


if __name__ == "__main__":
    main()
