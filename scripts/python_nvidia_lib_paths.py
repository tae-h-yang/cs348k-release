"""Print Python-installed NVIDIA library paths for CUDA ONNXRuntime."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__import__("site").getsitepackages()[0]) / "nvidia"
    if not root.exists():
        print("")
        return
    paths = [p for p in root.glob("*/lib") if p.is_dir()]
    # CUDA 12 wheels must come before CUDA 13 torch libraries for onnxruntime-gpu.
    paths = sorted(paths, key=lambda p: ("/cu13/" in str(p), str(p)))
    print(":".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
