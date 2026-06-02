#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="$ROOT/results/kimodo_humanoid100_full_${RUN_ID}"
DATA_DIR="$ROOT/data/kimodo_humanoid100_full_${RUN_ID}"
LOG="$OUT_ROOT/pipeline.log"

mkdir -p "$OUT_ROOT" "$DATA_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "# Kimodo Humanoid100 Full Pipeline"
echo "run_id=$RUN_ID"
echo "started=$(date -Is)"
echo "root=$ROOT"
echo "out_root=$OUT_ROOT"
echo "data_dir=$DATA_DIR"

cd "$ROOT"

CUDA_LIBS="$(python scripts/python_nvidia_lib_paths.py 2>/dev/null || true)"
if [[ -n "$CUDA_LIBS" ]]; then
  export LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
  echo "cuda_libs=$CUDA_LIBS"
fi

python scripts/run_kimodo_humanoid100_experiment.py \
  --prompts_csv configs/humanoid_robotics_100_prompts.csv \
  --out_dir "$OUT_ROOT/generation" \
  --data_dir "$DATA_DIR/qpos" \
  --limit "${LIMIT:-100}" \
  --duration "${DURATION:-4.0}" \
  --diffusion_steps "${DIFFUSION_STEPS:-50}" \
  --timeout_s "${TIMEOUT_S:-1800}" \
  --max_runtime_s "${MAX_RUNTIME_S:-0}"

python - "$OUT_ROOT/generation/manifest.csv" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
rows = list(csv.DictReader(manifest.open())) if manifest.exists() else []
ok = sum(row.get("status") in {"success", "skipped_existing"} for row in rows)
if ok == 0:
    statuses = sorted({row.get("status", "missing") for row in rows}) or ["missing_manifest"]
    print(f"No Kimodo qpos exports available; stopping before eval/SONIC. statuses={statuses}")
    sys.exit(2)
print(f"Kimodo qpos exports available: {ok}")
PY

python scripts/evaluate_kimodo_humanoid100.py \
  --manifest "$OUT_ROOT/generation/manifest.csv" \
  --out_dir "$OUT_ROOT/eval" \
  --export_sonic_refs \
  --sonic_ref_dir "$OUT_ROOT/eval/sonic_references"

python scripts/evaluate_sonic_policy_mujoco.py \
  --reference_dir "$OUT_ROOT/eval/sonic_references" \
  --out_csv "$OUT_ROOT/eval/sonic_tracking_cuda.csv" \
  --summary_csv "$OUT_ROOT/eval/sonic_summary_cuda.csv" \
  --provider cuda \
  --init_reference_pose \
  --max_seconds "${SONIC_MAX_SECONDS:-5.0}" \
  --video_dir "$OUT_ROOT/eval/sonic_videos_cuda" \
  --save_rollouts_dir "$OUT_ROOT/eval/sonic_rollouts_cuda"

python scripts/render_video_contact_sheet.py \
  --video_dir "$OUT_ROOT/eval/sonic_videos_cuda" \
  --out "$OUT_ROOT/eval/sonic_videos_cuda_contact_sheet.jpg" \
  --glob '*.mp4' \
  --samples 4 \
  --thumb_width 180

echo "finished=$(date -Is)"
echo "Primary outputs:"
echo "$OUT_ROOT/generation/manifest.csv"
echo "$OUT_ROOT/eval/final_metrics.csv"
echo "$OUT_ROOT/eval/sonic_tracking_cuda.csv"
echo "$OUT_ROOT/eval/sonic_videos_cuda/"
echo "$OUT_ROOT/eval/sonic_videos_cuda_contact_sheet.jpg"
