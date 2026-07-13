"""Corner keypoint network with DSNT (differentiable spatial-to-numerical).

Architecture
------------
MobileNetV3-Small backbone (ImageNet init) → two upsampling refinement
blocks with skip connections → 4 corner heatmaps at 1/4 resolution →
spatial softmax → DSNT expectation → normalized (x, y) per corner.

Why DSNT instead of argmax heatmaps or direct regression:

- direct FC regression throws away spatial equivariance and trains slowly;
- argmax decoding is non-differentiable and quantized to the heatmap grid;
- DSNT keeps the dense spatial representation *and* yields sub-pixel
  coordinates trained end-to-end with a coordinate loss. Sub-pixel matters:
  a 2 px corner error at 256 px input becomes ~8 px on a 1000 px document
  after rectification — enough to blur OCR strokes.

Reference: Nibali et al., "Numerical Coordinate Regression with
Convolutional Neural Networks" (2018).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import mobilenet_v3_small

INPUT_SIZE = 256
HEATMAP_STRIDE = 4
N_CORNERS = 4  # TL, TR, BR, BL — order is part of the contract


def _coord_grids(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized pixel-center coordinate grids in [-1, 1]."""
    xs = (torch.arange(w, device=device, dtype=torch.float32) * 2 + 1) / w - 1
    ys = (torch.arange(h, device=device, dtype=torch.float32) * 2 + 1) / h - 1
    return xs.view(1, 1, 1, w), ys.view(1, 1, h, 1)


def dsnt(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatial softmax + coordinate expectation.

    heatmaps: (B, K, H, W) raw logits.
    Returns (coords (B, K, 2) in [-1, 1], probs (B, K, H, W)).
    """
    b, k, h, w = heatmaps.shape
    probs = F.softmax(heatmaps.flatten(2), dim=-1).view(b, k, h, w)
    gx, gy = _coord_grids(h, w, heatmaps.device)
    x = (probs * gx).sum(dim=(2, 3))
    y = (probs * gy).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1), probs


def js_regularization(probs: torch.Tensor, targets: torch.Tensor,
                      sigma: float = 1.5) -> torch.Tensor:
    """Jensen–Shannon divergence between predicted heatmaps and an isotropic
    Gaussian centered on the ground-truth corner. Shapes the spatial
    distribution so the expectation is meaningful (a bimodal heatmap can
    have a perfect expectation and still be garbage).

    probs: (B, K, H, W); targets: (B, K, 2) in [-1, 1].
    """
    b, k, h, w = probs.shape
    gx, gy = _coord_grids(h, w, probs.device)
    # per-axis sigma in normalized units
    sx, sy = 2 * sigma / w, 2 * sigma / h
    tx = targets[..., 0].view(b, k, 1, 1)
    ty = targets[..., 1].view(b, k, 1, 1)
    gauss = torch.exp(-((gx - tx) ** 2 / (2 * sx ** 2) + (gy - ty) ** 2 / (2 * sy ** 2)))
    gauss = gauss / gauss.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)

    eps = 1e-12
    m = 0.5 * (probs + gauss)
    kl_pm = (probs * ((probs + eps).log() - (m + eps).log())).sum(dim=(2, 3))
    kl_gm = (gauss * ((gauss + eps).log() - (m + eps).log())).sum(dim=(2, 3))
    return (0.5 * kl_pm + 0.5 * kl_gm).mean()


class _UpBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out), nn.Hardswish(),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(c_skip, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out), nn.Hardswish(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.Hardswish(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(self.reduce(x), size=skip.shape[2:], mode="bilinear",
                          align_corners=False)
        return self.fuse(x + self.skip(skip))


class CornerNet(nn.Module):
    """4-corner keypoint model. Input (B,3,256,256) in [0,1] ImageNet-normed;
    output coords (B,4,2) in [-1,1] plus heatmap logits for the JS loss."""

    # feature taps from mobilenet_v3_small.features:
    #   idx 3 -> 24ch stride 8, idx 8 -> 48ch stride 16, idx 12 -> 576ch stride 32
    _TAPS = {3: 24, 8: 48, 12: 576}

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.up1 = _UpBlock(576, 48, 128)   # 1/32 -> 1/16
        self.up2 = _UpBlock(128, 24, 64)    # 1/16 -> 1/8
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 1/4
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.Hardswish(),
            nn.Conv2d(64, N_CORNERS, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = {}
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self._TAPS:
                feats[i] = x
        y = self.up1(feats[12], feats[8])
        y = self.up2(y, feats[3])
        logits = self.head(y)
        coords, _ = dsnt(logits)
        return coords, logits


class CornerLoss(nn.Module):
    """L1 on DSNT coordinates + JS divergence shaping the heatmaps."""

    def __init__(self, js_weight: float = 1.0):
        super().__init__()
        self.js_weight = js_weight

    def forward(self, coords: torch.Tensor, logits: torch.Tensor,
                targets: torch.Tensor) -> dict[str, torch.Tensor]:
        _, probs = dsnt(logits)  # recompute probs from logits (cheap, keeps API small)
        l1 = F.l1_loss(coords, targets)
        js = js_regularization(probs, targets)
        return {"loss": l1 + self.js_weight * js, "l1": l1.detach(), "js": js.detach()}
