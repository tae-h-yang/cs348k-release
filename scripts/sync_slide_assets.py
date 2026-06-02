#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ASSET_RE = re.compile(r"""(?P<path>assets/[^\s"'<>)]*)""")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def referenced_assets(markdown: Path) -> set[str]:
    text = markdown.read_text()
    text = COMMENT_RE.sub("", text)
    assets: set[str] = set()
    for match in ASSET_RE.finditer(text):
        path = match.group("path").rstrip(".,;")
        if "://" in path:
            continue
        assets.add(path)
    return assets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", default="slides/deck.md")
    parser.add_argument("--source", default="slides/assets")
    parser.add_argument("--dest", default="slides/build/assets")
    args = parser.parse_args()

    markdown = Path(args.markdown)
    source = Path(args.source)
    dest = Path(args.dest)

    dest.mkdir(parents=True, exist_ok=True)
    assets = referenced_assets(markdown)
    expected = {asset.removeprefix("assets/") for asset in assets}
    for existing in list(dest.rglob("*")):
        if existing.is_file() and existing.relative_to(dest).as_posix() not in expected:
            existing.unlink()
    for directory in sorted((p for p in dest.rglob("*") if p.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass

    copied = 0
    missing: list[str] = []
    for asset in sorted(assets):
        rel = asset.removeprefix("assets/")
        src = source / rel
        dst = dest / rel
        if not src.exists():
            missing.append(asset)
            continue
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"[assets] copied {copied} referenced assets to {dest}")
    if missing:
        print("[assets] missing referenced assets:")
        for asset in missing:
            print(f"  {asset}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
