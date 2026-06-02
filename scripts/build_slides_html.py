"""Build a lightweight HTML slide deck from slides/deck.md.

This is a fallback for machines without Node/Marp. It supports the subset of
Markdown used by the project deck and keeps local image paths working.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "slides" / "deck.md"
DEFAULT_OUT = ROOT / "slides" / "build" / "deck.html"
DEFAULT_CSS = ROOT / "slides" / "theme.css"


def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :]
    return text


def split_slides(text: str) -> list[str]:
    slides: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip() == "---":
            slides.append("\n".join(buf).strip())
            buf = []
        else:
            if not line.lstrip().startswith("<!--"):
                buf.append(line)
    if buf:
        slides.append("\n".join(buf).strip())
    return [s for s in slides if s]


def markdown_to_html(text: str) -> str:
    # Marp allows raw HTML; Python-Markdown keeps it by default.
    return markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )


def make_html(slides: list[str], css: str) -> str:
    rendered = []
    for i, slide in enumerate(slides, start=1):
        body = markdown_to_html(slide)
        rendered.append(f'<section class="slide" id="slide-{i}">{body}<div class="page">{i}</div></section>')
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <base href="../">
  <title>Controller-in-the-Loop Curation for Generated Humanoid Motion</title>
  <style>
{css}

body {{
  margin: 0;
  background: #222;
  font-family: Inter, Aptos, Helvetica, Arial, sans-serif;
}}
.slide {{
  box-sizing: border-box;
  width: 1280px;
  height: 720px;
  margin: 24px auto;
  position: relative;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0,0,0,.35);
}}
.page {{
  position: absolute;
  right: 24px;
  bottom: 18px;
  color: #6b7780;
  font-size: 16px;
}}
pre {{
  background: #101820;
  color: #f2f6f8;
  padding: 16px;
  border-radius: 6px;
  font-size: 18px;
  overflow: hidden;
}}
code {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}}
@media print {{
  body {{ background: white; }}
  .slide {{ margin: 0; box-shadow: none; page-break-after: always; }}
}}
  </style>
</head>
<body>
{''.join(rendered)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--css", type=Path, default=DEFAULT_CSS)
    args = parser.parse_args()

    text = strip_frontmatter(args.input.read_text())
    css = args.css.read_text()
    html_text = make_html(split_slides(text), css)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text)
    print(args.output)


if __name__ == "__main__":
    main()
