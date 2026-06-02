"""Write a slide metrics snapshot from tracked release artifacts."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "slides" / "assets"
OUT = ASSETS / "metrics_snapshot.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt_int(value: str) -> int:
    return int(round(float(value)))


def fmt_float(value: str) -> str:
    return f"{float(value):.3f}"


def main() -> None:
    audit = read_rows(ASSETS / "kimodo_audit_snapshot.csv")
    flags = read_rows(ASSETS / "kimodo_failure_flag_stats.csv")

    lines = [
        "# Metrics Snapshot",
        "",
        "Source: tracked release assets under `slides/assets/`.",
        "",
        "## First-Pass KIMODO Audit",
        "",
        "| Set | Physical Pass | SONIC No-Fall | Both Gates | Mean SONIC Sec. | Mean RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit:
        n = fmt_int(row["n"])
        lines.append(
            f"| {row['set']} | {fmt_int(row['physical_pass'])}/{n} | "
            f"{fmt_int(row['sonic_no_fall'])}/{n} | {fmt_int(row['both'])}/{n} | "
            f"{fmt_float(row['mean_sonic_seconds'])} | {fmt_float(row['mean_rmse'])} |"
        )

    lines += [
        "",
        "## Non-Exclusive Failure Flags",
        "",
        "| Flag | Count | Mean SONIC Sec. |",
        "|---|---:|---:|",
    ]
    for row in flags:
        lines.append(
            f"| {row['flag']} | {fmt_int(row['count'])} | "
            f"{fmt_float(row['mean_sonic_seconds'])} |"
        )

    lines += [
        "",
        "## Deterministic Repair Snapshot",
        "",
        "| Metric | Original | Repaired |",
        "|---|---:|---:|",
        "| physical pass | 48/100 | 53/100 |",
        "| critic accept | 47/100 | 54/100 |",
        "| reject / regenerate | 18/100 | 16/100 |",
        "| SONIC 4s no-fall | 53/100 | 56/100 |",
        "| mean SONIC seconds | 2.855 | 3.007 |",
        "| mean RMSE | 0.156 | 0.142 |",
        "| mean risk | 41.548 | 37.916 |",
        "| contact artifact | 0.264 | 0.247 |",
        "",
        "Repair evidence comes from `docs/kimodo_repair_results_2026-05-31.md`: "
        "seven SONIC rescues and four regressions.",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
