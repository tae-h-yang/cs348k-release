#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p slides/build
python scripts/sync_slide_assets.py --markdown slides/deck.md --source slides/assets --dest slides/build/assets

if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  find slides/build/assets/videos -type f -name '*.mp4' -print0 | while IFS= read -r -d '' video; do
    codec="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$video" || true)"
    if [[ "$codec" != "h264" ]]; then
      tmp="${video%.mp4}.h264.tmp.mp4"
      ffmpeg -nostdin -y -v error -i "$video" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart -an "$tmp"
      mv "$tmp" "$video"
      echo "Transcoded $video to H.264"
    fi
  done
else
  echo "ffmpeg/ffprobe not found; skipping video transcoding"
fi

python scripts/update_slide_metrics.py
python scripts/build_slides_html.py --input slides/deck.md --output slides/build/deck.html --css slides/theme.css

if command -v marp >/dev/null 2>&1; then
  MARP=(marp)
elif command -v npx >/dev/null 2>&1; then
  MARP=(npx --yes @marp-team/marp-cli)
fi

if [[ "${MARP+x}" == x ]]; then
  "${MARP[@]}" slides/deck.md --html --allow-local-files --theme-set slides/theme.css -o slides/build/deck.marp.html
  "${MARP[@]}" slides/deck.md --pdf --allow-local-files --theme-set slides/theme.css -o slides/build/deck.pdf
  "${MARP[@]}" slides/deck.md --pptx --allow-local-files --theme-set slides/theme.css -o slides/build/deck.pptx
  echo "Marp exports written to slides/build/deck.{marp.html,pdf,pptx}"
else
  echo "Marp CLI not found. HTML fallback written to slides/build/deck.html"
  echo "Install with: conda install -y -c conda-forge nodejs && npm install -g @marp-team/marp-cli"
fi
