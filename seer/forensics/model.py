"""Tamper CNN: joint document-level score and coarse localization.

A compact fully-convolutional encoder over the (luminance, ELA, residual)
stack produces a 1/8-resolution tamper heatmap; the document-level logit
combines global average and max pooling of that heatmap. Tying the global
score to the localization map (instead of a separate FC head) forces the
model to point at evidence — a score with a "where", which is what a human
reviewer needs from a flag.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

INPUT_SIZE = (384, 256)  # w, h — canonical documents resized to this
HEATMAP_STRIDE = 8


def _block(ci: int, co: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )


class TamperNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            _block(3, 32), _block(32, 32, stride=2),     # 1/2
            _block(32, 64), _block(64, 64, stride=2),    # 1/4
            _block(64, 128), _block(128, 128, stride=2),  # 1/8
            _block(128, 128), _block(128, 128),
        )
        self.heatmap = nn.Conv2d(128, 1, 1)
        # global logit from pooled heatmap statistics (evidence-tied)
        self.score = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B,3,H,W) -> (doc logit (B,), heatmap logits (B,1,H/8,W/8))."""
        f = self.encoder(x)
        hm = self.heatmap(f)
        pooled = torch.cat([
            hm.mean(dim=(2, 3)),
            hm.amax(dim=(2, 3)),
        ], dim=1)
        return self.score(pooled).squeeze(1), hm


class TamperLoss(nn.Module):
    """Document BCE + masked heatmap BCE.

    Heatmap supervision only applies to tampered samples (clean documents
    supervise the map toward zero everywhere, which the doc loss already
    encourages via the pooled statistics).
    """

    def __init__(self, heatmap_weight: float = 1.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight

    def forward(self, doc_logit: torch.Tensor, hm_logits: torch.Tensor,
                tampered: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        doc = F.binary_cross_entropy_with_logits(doc_logit, tampered.float())
        target = F.interpolate(masks, size=hm_logits.shape[2:], mode="area")
        target = (target > 0.25).float()
        # weight positive pixels up: tamper regions are a small area fraction
        pos_weight = torch.tensor(8.0, device=hm_logits.device)
        hm = F.binary_cross_entropy_with_logits(hm_logits, target,
                                                pos_weight=pos_weight)
        return {"loss": doc + self.heatmap_weight * hm,
                "doc": doc.detach(), "heatmap": hm.detach()}
