"""Scene compositing and domain randomization.

Takes a canonical document render and produces a "photo someone took with a
phone": the document is perspective-warped onto a procedural background, then
degraded photometrically. Because we build the homography ourselves, the four
document corners in scene coordinates are *exact* ground truth — the whole
reason to synthesize rather than annotate.

Randomization axes (each one is a failure mode observed in real KYC uploads):
geometry (tilt, perspective, scale, off-center), background clutter, ambient
light level and color temperature, directional shading, specular glare,
cast shadows, defocus and motion blur, sensor noise, JPEG recompression.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from seer.synth.template import DocumentRender

SCENE_SIZE = (896, 1152)  # w, h — portrait phone-ish frame


@dataclass
class SceneSample:
    image: np.ndarray            # HxWx3 uint8, RGB
    corners: np.ndarray          # 4x2 float32, TL TR BR BL, scene coords
    homography: np.ndarray      # 3x3, canonical -> scene
    render: DocumentRender


# ---------------------------------------------------------------- backgrounds

def _perlin_like(shape: tuple[int, int], rng: random.Random, octaves: int = 4) -> np.ndarray:
    """Cheap multi-octave value noise in [0,1] via upsampled random grids."""
    h, w = shape
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh, gw = 2 ** (o + 2), 2 ** (o + 2)
        grid = np.random.RandomState(rng.getrandbits(31)).rand(gh, gw).astype(np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    return np.clip(out / total, 0, 1)


def random_background(rng: random.Random, size: tuple[int, int] = SCENE_SIZE) -> np.ndarray:
    w, h = size
    kind = rng.choice(["wood", "fabric", "paper", "dark", "gradient"])
    noise = _perlin_like((h, w), rng)
    if kind == "wood":
        base = np.array(rng.choice([(120, 82, 45), (150, 105, 62), (90, 60, 35)]), np.float32)
        streaks = _perlin_like((h, w // 8), rng)
        streaks = cv2.resize(streaks, (w, h))
        tex = 0.6 * noise + 0.4 * streaks
    elif kind == "fabric":
        base = np.array(rng.choice([(70, 80, 95), (110, 100, 90), (60, 90, 70)]), np.float32)
        tex = noise
    elif kind == "paper":
        base = np.array((215, 212, 205), np.float32)
        tex = noise * 0.3 + 0.7
    elif kind == "dark":
        base = np.array(rng.choice([(35, 35, 40), (50, 45, 42)]), np.float32)
        tex = noise * 0.5 + 0.5
    else:
        base = np.array([rng.randint(60, 190) for _ in range(3)], np.float32)
        gy = np.linspace(0.6, 1.1, h, dtype=np.float32)[:, None]
        tex = noise * 0.4 + 0.6 * gy
    img = base[None, None, :] * (0.55 + 0.45 * tex[..., None])
    # occasional clutter: a few flat rectangles (other papers, phone, pen)
    img = img.astype(np.float32)
    for _ in range(rng.randint(0, 3)):
        x0, y0 = rng.randint(0, w - 60), rng.randint(0, h - 60)
        x1, y1 = min(w, x0 + rng.randint(60, 300)), min(h, y0 + rng.randint(40, 260))
        color = [rng.randint(20, 235) for _ in range(3)]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- placement

def _target_quad(doc_wh: tuple[int, int], rng: random.Random,
                 size: tuple[int, int] = SCENE_SIZE) -> np.ndarray:
    """Random plausible placement: scale, rotation, perspective jitter."""
    sw, sh = size
    dw, dh = doc_wh
    # scale so the document occupies 45–85% of scene width
    scale = rng.uniform(0.45, 0.85) * sw / dw
    w2, h2 = dw * scale / 2, dh * scale / 2
    theta = math.radians(rng.uniform(-28, 28))
    c, s = math.cos(theta), math.sin(theta)
    base = np.array([[-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]], np.float32)
    rot = base @ np.array([[c, -s], [s, c]], np.float32).T
    # perspective: independent corner jitter, bounded so the quad stays convex
    jitter = rng.uniform(0.02, 0.10) * scale * dw
    rot += np.random.RandomState(rng.getrandbits(31)) \
        .uniform(-jitter, jitter, (4, 2)).astype(np.float32)
    # center placement, keep fully inside with a margin
    margin = 12
    minxy, maxxy = rot.min(0), rot.max(0)
    cx = rng.uniform(margin - minxy[0], sw - margin - maxxy[0])
    cy = rng.uniform(margin - minxy[1], sh - margin - maxxy[1])
    return rot + np.array([cx, cy], np.float32)


# -------------------------------------------------------------- photometrics

def _photometrics(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    # ambient level + color temperature
    out *= rng.uniform(0.55, 1.15)
    temp = rng.uniform(-1, 1)
    out[..., 0] *= 1 + 0.10 * temp   # R
    out[..., 2] *= 1 - 0.10 * temp   # B

    # directional shading plane
    angle = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    plane = ((xx / w - 0.5) * math.cos(angle) + (yy / h - 0.5) * math.sin(angle))
    out *= (1 + rng.uniform(0.0, 0.35) * plane)[..., None]

    # specular glare blobs
    for _ in range(rng.randint(0, 2)):
        gx, gy = rng.uniform(0, w), rng.uniform(0, h)
        sigma = rng.uniform(w * 0.05, w * 0.22)
        blob = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2)))
        out += (rng.uniform(60, 160) * blob)[..., None]

    # cast shadow: soft dark band
    if rng.random() < 0.35:
        mask = np.zeros((h, w), np.float32)
        x0 = rng.randint(0, w)
        pts = np.array([[x0, 0], [x0 + rng.randint(-200, 200), h],
                        [x0 + rng.randint(100, 400), h], [x0 + rng.randint(100, 400), 0]])
        cv2.fillPoly(mask, [pts], 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(15, 60))
        out *= (1 - rng.uniform(0.15, 0.4) * mask)[..., None]

    out = np.clip(out, 0, 255)

    # blur: defocus or motion
    r = rng.random()
    if r < 0.35:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 2.2))
    elif r < 0.55:
        k = rng.randint(5, 15)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        rotm = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, rotm, (k, k))
        kernel /= max(kernel.sum(), 1e-6)
        out = cv2.filter2D(out, -1, kernel)

    # sensor noise
    out += np.random.RandomState(rng.getrandbits(31)) \
        .normal(0, rng.uniform(1.0, 7.0), out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # JPEG round trip
    q = rng.randint(55, 95)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


# -------------------------------------------------------------------- main

def compose_scene(render: DocumentRender, rng: random.Random,
                  size: tuple[int, int] = SCENE_SIZE) -> SceneSample:
    doc = np.asarray(render.image.convert("RGB"))
    dh, dw = doc.shape[:2]
    src = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], np.float32)
    dst = _target_quad((dw, dh), rng, size)
    H = cv2.getPerspectiveTransform(src, dst)

    bg = random_background(rng, size)
    warped = cv2.warpPerspective(doc, H, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((dh, dw), 255, np.uint8), H, size,
                               flags=cv2.INTER_LINEAR)
    # feather the edge slightly so the composite doesn't have a razor seam
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255.0
    scene = (warped * mask + bg * (1 - mask)).astype(np.uint8)

    scene = _photometrics(scene, rng)
    return SceneSample(image=scene, corners=dst.astype(np.float32),
                       homography=H.astype(np.float64), render=render)


def scene_to_pil(sample: SceneSample) -> Image.Image:
    return Image.fromarray(sample.image)
