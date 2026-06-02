"""Plot approximate SONIC tracking summary for supported Humanoid100 proxies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results" / "humanoid100_final_eval" / "sonic_supported_summary.csv"
DEFAULT_OUT = ROOT / "results" / "humanoid100_final_eval" / "sonic_supported_summary.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = [r for r in read_rows(args.summary_csv) if r["group"] in {"K1", "K8", "K9"}]
    labels = ["K=1", "K=8", "repaired"]
    seconds = [float(r["mean_track_seconds"]) for r in rows]
    rmse = [float(r["mean_tracking_rmse"]) for r in rows]
    falls = [int(r["fell_count"]) for r in rows]
    colors = ["#707070", "#3267a8", "#2d8f67"]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    axes[0].bar(labels, seconds, color=colors)
    axes[0].set_ylabel("Mean tracking seconds")
    axes[0].set_title("Survival")
    axes[1].bar(labels, rmse, color=colors)
    axes[1].set_ylabel("Mean joint RMSE")
    axes[1].set_title("Tracking Error")
    axes[2].bar(labels, falls, color=colors)
    n = max(int(r["n"]) for r in rows)
    axes[2].set_ylim(0, max(max(falls) + 1, n))
    axes[2].set_ylabel(f"Falls / {n}")
    axes[2].set_title("Falls")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Approximate SONIC tracking on {n} prompt proxy references", y=1.03)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
