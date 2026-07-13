"""CRNN text-line recognizer (Shi et al., 2015) with CTC decoding.

Input: grayscale line crop resized to height 32, width 384 (aspect-preserving
with right padding). The conv stack downsamples width by 4, giving 96 time
steps — comfortably above the 2*44+1 alignment floor a 44-char MRZ needs
under CTC.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from seer.ocr.charset import BLANK, NUM_CLASSES, decode_greedy

LINE_HEIGHT = 32
LINE_WIDTH = 384
TIME_STEPS = LINE_WIDTH // 4


def prepare_line(gray: np.ndarray) -> np.ndarray:
    """Aspect-preserving resize to 32x384 with right padding; float [0,1],
    mean-var normalized per crop (robust to paper tone and lighting)."""
    h, w = gray.shape[:2]
    scale = LINE_HEIGHT / h
    new_w = min(LINE_WIDTH, max(8, int(round(w * scale))))
    resized = cv2.resize(gray, (new_w, LINE_HEIGHT), interpolation=cv2.INTER_AREA)
    canvas = np.full((LINE_HEIGHT, LINE_WIDTH), resized[:, -1:].mean(), np.float32)
    canvas[:, :new_w] = resized
    canvas /= 255.0
    canvas = (canvas - canvas.mean()) / max(canvas.std(), 1e-4)
    return canvas


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 256):
        super().__init__()
        def block(ci, co, pool):
            return [nn.Conv2d(ci, co, 3, padding=1, bias=False),
                    nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                    nn.MaxPool2d(pool) if pool else nn.Identity()]
        self.cnn = nn.Sequential(
            *block(1, 64, (2, 2)),      # 32x384 -> 16x192
            *block(64, 128, (2, 2)),    # -> 8x96
            *block(128, 256, None),
            *block(256, 256, (2, 1)),   # -> 4x96
            *block(256, 512, None),
            *block(512, 512, (2, 1)),   # -> 2x96
            nn.Conv2d(512, 512, (2, 1), bias=False),  # collapse height: -> 1x96
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True,
                           batch_first=False, dropout=0.1)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,32,384) -> log-probs (T, B, C) for CTCLoss."""
        f = self.cnn(x)                    # (B, 512, 1, T)
        f = f.squeeze(2).permute(2, 0, 1)  # (T, B, 512)
        y, _ = self.rnn(f)
        return F.log_softmax(self.fc(y), dim=-1)


@torch.no_grad()
def recognize(model: nn.Module, gray_crops: list[np.ndarray],
              device: torch.device) -> list[tuple[str, float]]:
    """Batch-recognize line crops. Returns (text, confidence) per crop,
    confidence being the mean max-probability over non-blank frames."""
    if not gray_crops:
        return []
    batch = np.stack([prepare_line(g) for g in gray_crops])[:, None]
    logp = model(torch.from_numpy(batch).to(device))     # (T, B, C)
    probs = logp.exp().permute(1, 0, 2).cpu().numpy()     # (B, T, C)
    out = []
    for p in probs:
        idx = p.argmax(axis=-1)
        text = decode_greedy(idx.tolist())
        nonblank = idx != BLANK
        conf = float(p.max(axis=-1)[nonblank].mean()) if nonblank.any() else 0.0
        out.append((text, conf))
    return out
