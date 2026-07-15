"""Shared dataset-path helpers.

Canonical renders may be stored as PNG (lossless, larger) or JPEG q95
(~6x smaller — the right call on small SSDs, and arguably more realistic
since real KYC uploads always arrive with a JPEG history). Loaders resolve
whichever exists.
"""

from __future__ import annotations

from pathlib import Path

CANONICAL_EXTS = (".png", ".jpg")


def canonical_path(root: str | Path, sample_id: str) -> Path:
    base = Path(root) / "canonical"
    for ext in CANONICAL_EXTS:
        p = base / f"{sample_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no canonical image for sample {sample_id} under {base}")
