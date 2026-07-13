"""Portrait sources for specimen documents.

Two providers behind one interface:

- ``DirectoryFaces``: samples from a folder of externally generated synthetic
  faces (e.g. the SFHQ dataset or StyleGAN "this person does not exist"
  dumps). This is what you want when the *face verification* stage must be
  exercised realistically.

- ``ProceduralFaces``: a fully self-contained parametric portrait generator.
  The faces are stylized, but they carry the properties the *localization and
  OCR* stages care about — a photo-shaped region with plausible luminance
  structure, hair/skin contrast and per-identity variation — so the whole
  data pipeline runs with zero external assets.

Both return an RGB PIL image plus a stable identity key, so tamper synthesis
can deliberately swap in a *different* identity's portrait.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class Portrait:
    image: Image.Image
    identity: str


class DirectoryFaces:
    """Sample portraits from a directory of synthetic face images."""

    def __init__(self, root: str | Path):
        self.paths = sorted(
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if not self.paths:
            raise FileNotFoundError(f"no face images under {root}")

    def sample(self, rng: random.Random) -> Portrait:
        p = rng.choice(self.paths)
        return Portrait(Image.open(p).convert("RGB"), identity=p.stem)


class ProceduralFaces:
    """Parametric portrait generator: no external assets, infinite identities.

    An identity is a deterministic parameter vector (skin/hair tone, face
    geometry, feature placement). Rendering adds smooth studio-style shading
    so crops behave like photographs under downstream augmentation.
    """

    def sample(self, rng: random.Random, size: int = 256) -> Portrait:
        seed = rng.getrandbits(32)
        r = random.Random(seed)
        s = size
        img = Image.new("RGB", (s, s))
        d = ImageDraw.Draw(img)

        # studio backdrop with a soft vertical gradient
        top = r.randint(150, 220)
        for y in range(s):
            v = int(top - 40 * y / s)
            d.line([(0, y), (s, y)], fill=(v, v, min(255, v + r.randint(0, 6))))

        skin_bases = [(141, 85, 36), (168, 110, 60), (198, 134, 66),
                      (110, 68, 30), (90, 56, 28), (222, 170, 110)]
        skin = skin_bases[r.randrange(len(skin_bases))]
        skin = tuple(min(255, max(0, c + r.randint(-12, 12))) for c in skin)
        dark = tuple(int(c * 0.55) for c in skin)

        cx, cy = s // 2, int(s * 0.52)
        fw = int(s * r.uniform(0.26, 0.33))   # face half-width
        fh = int(fw * r.uniform(1.25, 1.45))  # face half-height

        # shoulders / torso
        d.rectangle([cx - int(fw * 2.2), cy + int(fh * 0.9), cx + int(fw * 2.2), s],
                    fill=tuple(r.randint(30, 90) for _ in range(3)))
        d.ellipse([cx - int(fw * 0.35), cy + int(fh * 0.55),
                   cx + int(fw * 0.35), cy + int(fh * 1.3)], fill=skin)  # neck

        # head
        d.ellipse([cx - fw, cy - fh, cx + fw, cy + fh], fill=skin)

        # hair: cap + optional sides, tone varies
        hair = tuple(int(c) for c in (r.randint(10, 60),) * 3)
        hair_drop = r.uniform(0.15, 0.55)
        d.pieslice([cx - fw - 2, cy - fh - 2, cx + fw + 2, cy + int(fh * hair_drop)],
                   180, 360, fill=hair)

        # eyes with sclera + iris
        ey = cy - int(fh * r.uniform(0.10, 0.20))
        ex = int(fw * r.uniform(0.38, 0.52))
        ew, eh = int(fw * 0.22), int(fh * 0.07)
        for sx in (-1, 1):
            x = cx + sx * ex
            d.ellipse([x - ew, ey - eh, x + ew, ey + eh], fill=(235, 232, 225))
            ir = int(eh * r.uniform(0.8, 1.0))
            d.ellipse([x - ir, ey - ir, x + ir, ey + ir], fill=(30, 20, 15))
            # brow
            by = ey - int(fh * r.uniform(0.10, 0.16))
            d.line([(x - ew, by), (x + ew, by - r.randint(-3, 3))], fill=hair,
                   width=max(2, int(fh * 0.035)))

        # nose
        nw = int(fw * r.uniform(0.10, 0.16))
        ny = cy + int(fh * 0.18)
        d.polygon([(cx, ey + eh), (cx - nw, ny), (cx + nw, ny)], fill=dark)

        # mouth
        my = cy + int(fh * r.uniform(0.42, 0.52))
        mw = int(fw * r.uniform(0.35, 0.5))
        d.line([(cx - mw, my), (cx + mw, my)], fill=(90, 40, 35),
               width=max(2, int(fh * 0.045)))

        # ears
        for sx in (-1, 1):
            d.ellipse([cx + sx * fw - int(fw * 0.12), ey - int(fh * 0.05),
                       cx + sx * fw + int(fw * 0.12), ey + int(fh * 0.18)], fill=skin)

        # soft directional shading, then mild blur to kill vector-art edges
        arr = np.asarray(img).astype(np.float32)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) / s
        angle = r.uniform(0, 2 * math.pi)
        shade = 1.0 + 0.18 * ((xx - 0.5) * math.cos(angle) + (yy - 0.5) * math.sin(angle))
        arr *= shade[..., None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=s * 0.006))
        return Portrait(img, identity=f"proc-{seed:08x}")


def make_provider(faces_dir: str | Path | None):
    return DirectoryFaces(faces_dir) if faces_dir else ProceduralFaces()
