"""Dataset adapter: synth scenes -> (image tensor, normalized corner targets).

Geometric augmentation is deliberately *absent* here: the synth engine
already randomizes geometry with exact label propagation, which is strictly
better than augmenting after the fact (no interpolation drift in labels).
Only label-preserving photometric jitter is applied at load time.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from seer.localize.model import INPUT_SIZE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalize_image(rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> (3,H,W) float tensor, ImageNet-normalized."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class CornerDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train",
                 input_size: int = INPUT_SIZE, jitter: bool | None = None):
        self.root = Path(root)
        self.input_size = input_size
        self.jitter = split == "train" if jitter is None else jitter
        self.items = [
            json.loads(line)["id"]
            for line in (self.root / "index.jsonl").read_text().splitlines()
            if json.loads(line)["split"] == split
        ]
        if not self.items:
            raise FileNotFoundError(f"no '{split}' samples in {self.root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sid = self.items[idx]
        label = json.loads((self.root / "labels" / f"{sid}.json").read_text())
        img = cv2.cvtColor(cv2.imread(str(self.root / "scenes" / f"{sid}.jpg")),
                           cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        corners = np.array(label["corners"], np.float32)  # 4x2 scene px

        img = cv2.resize(img, (self.input_size, self.input_size),
                         interpolation=cv2.INTER_AREA)
        if self.jitter:
            img = self._photometric_jitter(img)

        # normalize corners to [-1, 1] (pixel-center convention matches DSNT)
        tx = (corners[:, 0] * 2 + 1) / w - 1
        ty = (corners[:, 1] * 2 + 1) / h - 1
        target = torch.from_numpy(np.stack([tx, ty], axis=1))
        return normalize_image(img), target

    @staticmethod
    def _photometric_jitter(img: np.ndarray) -> np.ndarray:
        r = random.Random()
        out = img.astype(np.float32)
        out *= r.uniform(0.85, 1.15)                      # brightness
        out = (out - 128) * r.uniform(0.85, 1.15) + 128   # contrast
        if r.random() < 0.2:                              # grayscale collapse
            g = out.mean(axis=2, keepdims=True)
            out = out * 0.3 + g * 0.7
        return np.clip(out, 0, 255).astype(np.uint8)


def denormalize_corners(coords: torch.Tensor, width: int, height: int) -> np.ndarray:
    """(...,4,2) in [-1,1] -> pixel coordinates in the original image."""
    c = coords.detach().cpu().numpy()
    out = np.empty_like(c)
    out[..., 0] = ((c[..., 0] + 1) * width - 1) / 2
    out[..., 1] = ((c[..., 1] + 1) * height - 1) / 2
    return out
