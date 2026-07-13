"""Forensics invariants: signals respond to real manipulations, AUC math."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cv2  # noqa: E402

from seer.forensics.model import INPUT_SIZE, TamperLoss, TamperNet  # noqa: E402
from seer.forensics.signals import ela_map, forensic_stack, noise_residual  # noqa: E402
from seer.forensics.train import roc_auc  # noqa: E402


def _document_like(seed=0):
    """Smooth 'paper' with text-like strokes, single JPEG history."""
    rng = np.random.RandomState(seed)
    img = np.full((256, 384, 3), 235, np.uint8)
    for _ in range(40):
        x, y = rng.randint(10, 350), rng.randint(10, 240)
        cv2.line(img, (x, y), (x + rng.randint(5, 30), y), (20, 20, 30), 2)
    img = img + rng.normal(0, 2, img.shape).astype(np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def test_ela_highlights_recompressed_patch():
    img = _document_like(1)
    tampered = img.copy()
    # splice a patch with a very different JPEG history
    patch = tampered[60:120, 100:220]
    _, buf = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 30])
    tampered[60:120, 100:220] = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    ela = ela_map(cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB))
    inside = ela[60:120, 100:220].mean()
    outside_mask = np.ones(ela.shape, bool)
    outside_mask[50:130, 90:230] = False
    outside = ela[outside_mask].mean()
    # An already-crushed q30 patch sits near its JPEG fixed point, so it
    # responds *less* to recompression than the q92 surroundings — the ELA
    # anomaly is a divergence in error level, not always an increase.
    ratio = max(inside, outside) / max(min(inside, outside), 1e-6)
    assert ratio > 1.3


def test_noise_residual_shape_and_range():
    img = cv2.cvtColor(_document_like(2), cv2.COLOR_BGR2RGB)
    res = noise_residual(img)
    assert res.shape == img.shape[:2]
    assert 0.0 <= res.min() and res.max() <= 1.0


def test_forensic_stack_shape():
    img = cv2.cvtColor(_document_like(3), cv2.COLOR_BGR2RGB)
    stack = forensic_stack(img, INPUT_SIZE)
    w, h = INPUT_SIZE
    assert stack.shape == (3, h, w)
    assert stack.dtype == np.float32


def test_tampernet_shapes_and_loss():
    model = TamperNet().eval()
    w, h = INPUT_SIZE
    x = torch.randn(2, 3, h, w)
    with torch.no_grad():
        logit, hm = model(x)
    assert logit.shape == (2,)
    assert hm.shape == (2, 1, h // 8, w // 8)

    masks = torch.zeros(2, 1, h, w)
    masks[1, 0, 40:80, 60:120] = 1.0
    losses = TamperLoss()(logit, hm, torch.tensor([0.0, 1.0]), masks)
    assert torch.isfinite(losses["loss"])


def test_roc_auc_known_cases():
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    rng = np.random.RandomState(0)
    labels = rng.randint(0, 2, 2000)
    assert roc_auc(labels, rng.rand(2000)) == pytest.approx(0.5, abs=0.05)
