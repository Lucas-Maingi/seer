"""Localization stage invariants: DSNT math and rectification geometry."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seer.localize.model import CornerNet, dsnt, js_regularization  # noqa: E402
from seer.localize.rectify import (  # noqa: E402
    classify_kind, estimate_aspect, order_corners, quad_iou, rectify,
)


def test_dsnt_recovers_peak_location():
    # a sharp peak at a known cell must yield its normalized center coords
    h, w = 64, 64
    logits = torch.full((1, 1, h, w), -20.0)
    py, px = 10, 50
    logits[0, 0, py, px] = 20.0
    coords, probs = dsnt(logits)
    exp_x = (px * 2 + 1) / w - 1
    exp_y = (py * 2 + 1) / h - 1
    assert torch.allclose(coords[0, 0], torch.tensor([exp_x, exp_y]), atol=1e-4)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_js_regularization_prefers_correct_gaussian():
    h, w = 32, 32
    target = torch.tensor([[[0.25, -0.5]]])
    # heatmap peaked at the target vs peaked elsewhere
    def peaked(nx, ny):
        gx = ((torch.arange(w) * 2 + 1) / w - 1).view(1, 1, 1, w)
        gy = ((torch.arange(h) * 2 + 1) / h - 1).view(1, 1, h, 1)
        g = torch.exp(-((gx - nx) ** 2 + (gy - ny) ** 2) / 0.02)
        return g / g.sum()
    good = js_regularization(peaked(0.25, -0.5), target)
    bad = js_regularization(peaked(-0.7, 0.7), target)
    assert good < bad


def test_cornernet_shapes():
    model = CornerNet(pretrained=False).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        coords, logits = model(x)
    assert coords.shape == (2, 4, 2)
    assert logits.shape[:2] == (2, 4)
    assert coords.abs().max() <= 1.0


def test_order_corners_any_permutation():
    quad = np.array([[10, 20], [200, 30], [210, 150], [5, 140]], np.float32)
    for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        ordered = order_corners(quad[perm])
        assert np.allclose(ordered, quad)


def test_rectify_roundtrip_and_kind():
    import cv2
    from seer.synth.template import ID_SIZE
    w, h = ID_SIZE
    # smooth content: high-frequency noise cannot survive a down/up warp
    # round trip, so use gradients that interpolation preserves
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    canonical = np.stack([
        255 * xx / w, 255 * yy / h, 128 + 127 * np.sin(xx / 40) * np.cos(yy / 40),
    ], axis=-1).astype(np.uint8)
    dst_quad = np.array([[80, 60], [700, 100], [660, 480], [50, 430]], np.float32)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst_quad)
    scene = cv2.warpPerspective(canonical, H, (800, 600))

    assert classify_kind(dst_quad) == "national_id"
    assert 1.2 < estimate_aspect(dst_quad) < 2.0
    out, kind, _ = rectify(scene, dst_quad)
    assert kind == "national_id"
    assert out.shape == (h, w, 3)
    # interior should match the original up to interpolation error
    inner = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    diff = np.abs(out[inner].astype(int) - canonical[inner].astype(int)).mean()
    assert diff < 30


def test_quad_iou_identity_and_disjoint():
    q = np.array([[10, 10], [100, 10], [100, 80], [10, 80]], np.float32)
    assert quad_iou(q, q, (200, 200)) > 0.95
    assert quad_iou(q, q + 150, (400, 400)) < 0.05
