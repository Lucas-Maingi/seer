"""Field-crop dataset for CRNN training.

Crops value fields (and MRZ lines) out of *canonical* renders using the
synth engine's exact quads, with random padding jitter so the recognizer
tolerates the ROI slop that rectification error introduces at inference.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from seer.datautil import canonical_path
from seer.ocr.charset import encode
from seer.ocr.crnn import prepare_line


class FieldCropDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", jitter: bool | None = None):
        self.root = Path(root)
        self.jitter = split == "train" if jitter is None else jitter
        self.samples: list[tuple[str, dict]] = []
        for line in (self.root / "index.jsonl").read_text().splitlines():
            meta = json.loads(line)
            if meta["split"] != split:
                continue
            label = json.loads(
                (self.root / "labels" / f"{meta['id']}.json").read_text())
            for f in label["fields"]:
                if f["text"].strip():
                    self.samples.append((meta["id"], f))
        if not self.samples:
            raise FileNotFoundError(f"no '{split}' field crops in {self.root}")
        self._cache: tuple[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def _canonical(self, sid: str) -> np.ndarray:
        if self._cache and self._cache[0] == sid:
            return self._cache[1]
        img = cv2.imread(str(canonical_path(self.root, sid)), cv2.IMREAD_GRAYSCALE)
        self._cache = (sid, img)
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sid, f = self.samples[idx]
        img = self._canonical(sid)
        q = np.array(f["quad"], np.float32)
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        r = random.Random() if self.jitter else random.Random(0)
        pad = (r.randint(1, 6), r.randint(1, 6), r.randint(1, 6), r.randint(1, 6)) \
            if self.jitter else (3, 3, 3, 3)
        x0 = max(0, int(x0) - pad[0]); y0 = max(0, int(y0) - pad[1])
        x1 = min(img.shape[1], int(x1) + pad[2]); y1 = min(img.shape[0], int(y1) + pad[3])
        crop = img[y0:y1, x0:x1]
        if self.jitter:
            crop = self._degrade(crop, r)
        line = prepare_line(crop)
        target = torch.tensor(encode(f["text"]), dtype=torch.long)
        return torch.from_numpy(line)[None], target

    @staticmethod
    def _degrade(crop: np.ndarray, r: random.Random) -> np.ndarray:
        """Simulate what rectification hands the recognizer: slight blur,
        resampling softness, noise, contrast wobble."""
        out = crop.astype(np.float32)
        if r.random() < 0.5:
            out = cv2.GaussianBlur(out, (0, 0), r.uniform(0.4, 1.4))
        if r.random() < 0.3:  # down-up resample (rectification softness)
            s = r.uniform(0.5, 0.9)
            hh, ww = out.shape
            small = cv2.resize(out, (max(4, int(ww * s)), max(4, int(hh * s))))
            out = cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)
        out = (out - 128) * r.uniform(0.7, 1.2) + 128 + r.uniform(-20, 20)
        out += np.random.RandomState(r.getrandbits(31)).normal(0, r.uniform(0, 6), out.shape)
        return np.clip(out, 0, 255).astype(np.uint8)


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """CTC collate: stacked images + concatenated flat targets with lengths."""
    imgs = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return imgs, targets, lengths
