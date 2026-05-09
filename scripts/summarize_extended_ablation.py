"""Regenerate summary statistics for the 105-identity extended ablation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "results" / "guided_ablation_extended.csv"
OUT_CSV = ROOT / "results" / "guided_ablation_extended_summary.csv"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict]:
    with open(IN_CSV) as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for k in sorted({int(r["K"]) for r in rows}):
        group = [
            r for r in rows
            if int(r["K"]) == k and np.isfinite(float(r["full_risk"]))
        ]
        risks = np.array([float(r["full_risk"]) for r in group])
        out.append({
            "K": k,
            "n": len(group),
            "mean_risk": float(np.mean(risks)) if len(risks) else float("nan"),
            "median_risk": float(np.median(risks)) if len(risks) else float("nan"),
            "accept": sum(r["action"] == "accept" for r in group),
            "repair_or_rerank": sum(r["action"] == "repair_or_rerank" for r in group),
            "reject_or_regenerate": sum(r["action"] == "reject_or_regenerate" for r in group),
        })

    for motion_type in sorted({r["type"] for r in rows}):
        type_rows = [r for r in rows if r["type"] == motion_type]
        for k in sorted({int(r["K"]) for r in type_rows}):
            group = [
                r for r in type_rows
                if int(r["K"]) == k and np.isfinite(float(r["full_risk"]))
            ]
            risks = np.array([float(r["full_risk"]) for r in group])
            out.append({
                "K": f"{motion_type}_K{k}",
                "n": len(group),
                "mean_risk": float(np.mean(risks)) if len(risks) else float("nan"),
                "median_risk": float(np.median(risks)) if len(risks) else float("nan"),
                "accept": sum(r["action"] == "accept" for r in group),
                "repair_or_rerank": sum(r["action"] == "repair_or_rerank" for r in group),
                "reject_or_regenerate": sum(r["action"] == "reject_or_regenerate" for r in group),
            })

    by_clip: dict[str, dict[int, dict]] = {}
    for row in rows:
        by_clip.setdefault(row["clip"], {})[int(row["K"])] = row
    paired = [values for values in by_clip.values() if 1 in values and 8 in values]
    if paired:
        k1 = np.array([float(p[1]["full_risk"]) for p in paired])
        k8 = np.array([float(p[8]["full_risk"]) for p in paired])
        finite_mask = k1 > 1e-6
        out.append({
            "K": "K8_vs_K1_paired",
            "n": len(paired),
            "mean_risk_delta": float(np.mean(k8 - k1)),
            "median_risk_delta": float(np.median(k8 - k1)),
            "aggregate_reduction_pct": float(
                100.0 * (np.mean(k1) - np.mean(k8)) / max(np.mean(k1), 1e-8)
            ),
            "mean_per_clip_reduction_pct": (
                float(np.mean(100.0 * (k1[finite_mask] - k8[finite_mask]) / k1[finite_mask]))
                if np.any(finite_mask) else float("nan")
            ),
            "median_per_clip_reduction_pct": (
                float(np.median(100.0 * (k1[finite_mask] - k8[finite_mask]) / k1[finite_mask]))
                if np.any(finite_mask) else float("nan")
            ),
            "finite_baseline_count": int(np.sum(finite_mask)),
            "improved_count": int(np.sum(k8 < k1)),
        })
        for motion_type in sorted({p[1]["type"] for p in paired}):
            mask = np.array([p[1]["type"] == motion_type for p in paired])
            k1_type = k1[mask]
            k8_type = k8[mask]
            finite_type = k1_type > 1e-6
            out.append({
                "K": f"{motion_type}_K8_vs_K1_paired",
                "n": int(np.sum(mask)),
                "mean_risk_delta": float(np.mean(k8_type - k1_type)),
                "median_risk_delta": float(np.median(k8_type - k1_type)),
                "aggregate_reduction_pct": float(
                    100.0 * (np.mean(k1_type) - np.mean(k8_type)) / max(np.mean(k1_type), 1e-8)
                ),
                "mean_per_clip_reduction_pct": (
                    float(np.mean(100.0 * (k1_type[finite_type] - k8_type[finite_type]) / k1_type[finite_type]))
                    if np.any(finite_type) else float("nan")
                ),
                "median_per_clip_reduction_pct": (
                    float(np.median(100.0 * (k1_type[finite_type] - k8_type[finite_type]) / k1_type[finite_type]))
                    if np.any(finite_type) else float("nan")
                ),
                "finite_baseline_count": int(np.sum(finite_type)),
                "improved_count": int(np.sum(k8_type < k1_type)),
            })
    return out


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    write_csv(OUT_CSV, summary)
    print(f"Saved: {OUT_CSV}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
