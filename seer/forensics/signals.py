"""Deterministic forensic signal maps.

- ELA (Error Level Analysis): recompress at a fixed JPEG quality and take
  the amplified absolute difference. Regions with a different compression
  history than their surroundings (a pasted patch, a screenshot splice)
  respond differently to recompression.

- Noise residual: image minus an edge-preserving denoise, i.e. the sensor/
  compression noise field. Splices interrupt the spatial consistency of
  this field; copy-moves duplicate it.

Both are normalized to [0,1] maps at the input resolution and stacked with
luminance as the 3-channel input of the tamper CNN.
"""

from __future__ import annotations

import cv2
import numpy as np

ELA_QUALITY = 90
ELA_GAIN = 12.0


def ela_map(rgb: np.ndarray, quality: int = ELA_QUALITY) -> np.ndarray:
    """HxWx3 uint8 -> HxW float32 in [0,1]."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return np.zeros(rgb.shape[:2], np.float32)
    re = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    diff = np.abs(bgr.astype(np.float32) - re.astype(np.float32)).mean(axis=2)
    return np.clip(diff * ELA_GAIN / 255.0, 0, 1).astype(np.float32)


def noise_residual(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> HxW float32 in [0,1]: |image - median-denoised|,
    contrast-normalized so the map reflects structure, not exposure."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    den = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    res = np.abs(gray - den)
    p99 = max(float(np.percentile(res, 99)), 1e-3)
    return np.clip(res / p99, 0, 1).astype(np.float32)


def forensic_stack(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Build the (3, H, W) float32 CNN input: luminance, ELA, residual.

    Signals are computed at native resolution *before* resizing — resizing
    first would destroy exactly the compression statistics we measure.
    """
    lum = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ela = ela_map(rgb)
    res = noise_residual(rgb)
    w, h = size
    chans = [cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
             for c in (lum, ela, res)]
    return np.stack(chans).astype(np.float32)
