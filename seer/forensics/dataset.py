"""Forensics dataset: canonical renders + tamper masks from the synth engine."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from seer.datautil import canonical_path
from seer.forensics.model import INPUT_SIZE
from seer.forensics.signals import forensic_stack


class TamperDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.items: list[dict] = [
            meta for meta in map(json.loads,
                                 (self.root / "index.jsonl").read_text().splitlines())
            if meta["split"] == split
        ]
        if not self.items:
            raise FileNotFoundError(f"no '{split}' samples in {self.root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        meta = self.items[idx]
        sid = meta["id"]
        bgr = cv2.imread(str(canonical_path(self.root, sid)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(forensic_stack(rgb, INPUT_SIZE))

        h, w = rgb.shape[:2]
        mask_path = self.root / "masks" / f"{sid}.png"
        if meta["tampered"] and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((h, w), np.uint8)
        mask = cv2.resize(mask, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        mask_t = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return x, torch.tensor(float(meta["tampered"])), mask_t
