"""Canonical path resolution: PNG and JPEG datasets must both load."""

import pytest

from seer.datautil import canonical_path


def test_resolves_png_then_jpg(tmp_path):
    (tmp_path / "canonical").mkdir()
    (tmp_path / "canonical" / "000001.jpg").write_bytes(b"x")
    assert canonical_path(tmp_path, "000001").suffix == ".jpg"
    # png takes precedence when both exist (lossless preferred)
    (tmp_path / "canonical" / "000001.png").write_bytes(b"x")
    assert canonical_path(tmp_path, "000001").suffix == ".png"


def test_missing_raises(tmp_path):
    (tmp_path / "canonical").mkdir()
    with pytest.raises(FileNotFoundError):
        canonical_path(tmp_path, "999999")
