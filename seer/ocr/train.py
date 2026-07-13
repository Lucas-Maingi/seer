"""Train the CRNN field recognizer with CTC.

    python -m seer.ocr.train --data data/synth --epochs 40 --out weights/crnn.pt

Reports character error rate (CER, Levenshtein) and exact-match rate on the
validation field crops — CER is what drives downstream check-digit failures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from seer.ocr.charset import BLANK, decode_greedy
from seer.ocr.crnn import CRNN
from seer.ocr.dataset import FieldCropDataset, collate


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    from seer.ocr.charset import _IDX_TO_CHAR  # decode targets for CER
    model.eval()
    total_edits = total_chars = exact = count = 0
    for imgs, targets, lengths in loader:
        logp = model(imgs.to(device))
        preds = logp.argmax(dim=-1).permute(1, 0).cpu().numpy()  # (B, T)
        offset = 0
        for i, ln in enumerate(lengths.tolist()):
            gt = "".join(_IDX_TO_CHAR[t] for t in targets[offset:offset + ln].tolist())
            offset += ln
            hyp = decode_greedy(preds[i].tolist())
            total_edits += levenshtein(hyp, gt)
            total_chars += len(gt)
            exact += int(hyp == gt)
            count += 1
    return {"cer": total_edits / max(total_chars, 1),
            "exact": exact / max(count, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("weights/crnn.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(FieldCropDataset(args.data, "train"), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers, collate_fn=collate,
                          drop_last=True)
    val_dl = DataLoader(FieldCropDataset(args.data, "val"), batch_size=args.batch,
                        num_workers=args.workers, collate_fn=collate)

    model = CRNN().to(device)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))

    best_cer = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for imgs, targets, lengths in train_dl:
            imgs = imgs.to(device)
            opt.zero_grad(set_to_none=True)
            logp = model(imgs)  # (T, B, C)
            input_lengths = torch.full((imgs.size(0),), logp.size(0), dtype=torch.long)
            loss = ctc(logp, targets, input_lengths, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            running += float(loss.detach())

        metrics = evaluate(model, val_dl, device)
        print(f"epoch {epoch:3d}  ctc {running / max(len(train_dl), 1):.4f}  "
              f"val CER {metrics['cer']:.4f}  exact {metrics['exact']:.3f}")
        if metrics["cer"] < best_cer:
            best_cer = metrics["cer"]
            torch.save({"model": model.state_dict(), "metrics": metrics,
                        "epoch": epoch}, args.out)

    print(f"best val CER: {best_cer:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
