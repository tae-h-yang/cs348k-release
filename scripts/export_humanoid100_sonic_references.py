"""Export Humanoid100 K1/K8/repaired qpos clips to SONIC reference folders."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_sonic_references import SONIC_FPS, SOURCE_FPS, export_clip  # noqa: E402


DEFAULT_FINAL_CSV = ROOT / "results" / "humanoid100_final_eval" / "final_metrics.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "humanoid100_final_eval" / "sonic_references_supported"

METHOD_TO_K = {
    "K1_first": 1,
    "K8_best_of_8": 8,
    "repaired_retime": 9,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_csv", type=Path, default=DEFAULT_FINAL_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supported_only", action="store_true", default=True)
    parser.add_argument("--all", action="store_true", help="Export forced proxies too.")
    parser.add_argument("--physical_pass_only", action="store_true")
    parser.add_argument("--methods", nargs="*", default=list(METHOD_TO_K))
    args = parser.parse_args()

    rows = read_rows(args.final_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for row in rows:
        if row["method"] not in set(args.methods):
            continue
        if args.supported_only and not args.all and row["semantic_supported"] != "__YES__":
            continue
        if args.physical_pass_only and row["physical_pass"] != "__YES__":
            continue
        k = METHOD_TO_K[row["method"]]
        name = f"{row['prompt_id']}_{row['subcategory']}_K{k}"
        exported = export_clip(Path(row["qpos_path"]), args.out_dir / name, SOURCE_FPS, SONIC_FPS)
        exported.update({
            "prompt_id": row["prompt_id"],
            "subcategory": row["subcategory"],
            "category": row["category"],
            "method": row["method"],
            "semantic_supported": row["semantic_supported"],
            "physical_pass": row["physical_pass"],
            "risk_score": row["risk_score"],
            "contact_artifact_score": row["contact_artifact_score"],
        })
        manifest.append(exported)

    write_csv(args.out_dir / "manifest.csv", manifest)
    print(f"Exported {len(manifest)} SONIC references to {args.out_dir}")
    print(f"Manifest: {args.out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
