"""Cross-platform TTF discovery for the renderer.

OCR training data must vary fonts the way scanned documents do, so we pick
from whatever serif/sans/mono faces the host provides rather than shipping a
single font. MRZ zones use a monospace face as an OCR-B stand-in (we are
rendering specimens, not reproducing the genuine typeface).
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

_SEARCH_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/System/Library/Fonts"),
    Path.home() / ".fonts",
]

_SANS = ["arial.ttf", "calibri.ttf", "segoeui.ttf", "tahoma.ttf", "verdana.ttf",
         "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
_SANS_BOLD = ["arialbd.ttf", "calibrib.ttf", "segoeuib.ttf", "tahomabd.ttf",
              "verdanab.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]
_SERIF = ["times.ttf", "georgia.ttf", "cambria.ttc", "DejaVuSerif.ttf",
          "LiberationSerif-Regular.ttf"]
_MONO = ["consola.ttf", "cour.ttf", "lucon.ttf", "DejaVuSansMono.ttf",
         "LiberationMono-Regular.ttf"]


@lru_cache(maxsize=1)
def _index() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for d in _SEARCH_DIRS:
        if d.is_dir():
            for p in d.rglob("*.tt[fc]"):
                found.setdefault(p.name.lower(), p)
    return found


def _first(names: list[str]) -> Path | None:
    idx = _index()
    for n in names:
        if n.lower() in idx:
            return idx[n.lower()]
    return None


def load(kind: str, size: int, rng: random.Random | None = None) -> ImageFont.FreeTypeFont:
    """kind: sans | sans_bold | serif | mono. With an rng, choose randomly
    among available candidates of that kind (font augmentation for OCR)."""
    pool = {"sans": _SANS, "sans_bold": _SANS_BOLD, "serif": _SERIF, "mono": _MONO}[kind]
    if rng is not None:
        avail = [n for n in pool if n.lower() in _index()]
        path = _index()[rng.choice(avail).lower()] if avail else _first(pool)
    else:
        path = _first(pool)
    if path is None:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 scalable fallback
    return ImageFont.truetype(str(path), size=size)
