"""Fetch third-party model files into weights/ (never committed).

    python scripts/fetch_models.py

- YuNet face detector: Apache-2.0, from the OpenCV Zoo
- ArcFace w600k_r50: InsightFace buffalo_l recognition model (research /
  non-commercial license — check before commercial deployment)

Downloads are pinned by URL; verify hashes change only when you intend it.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"

MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    # buffalo_l recognition model re-hosted on HuggingFace by the maintainers
    "arcface_w600k_r50.onnx":
        "https://huggingface.co/public-data/insightface/resolve/main/"
        "models/buffalo_l/w600k_r50.onnx",
}


def fetch(name: str, url: str) -> None:
    dest = WEIGHTS / name
    if dest.exists():
        print(f"[skip] {name} already present")
        return
    print(f"[get ] {name}\n       {url}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — pinned https URLs
    tmp.rename(dest)
    print(f"[ ok ] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    WEIGHTS.mkdir(exist_ok=True)
    failures = []
    for name, url in MODELS.items():
        try:
            fetch(name, url)
        except Exception as e:  # noqa: BLE001
            failures.append((name, e))
            print(f"[fail] {name}: {e}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
