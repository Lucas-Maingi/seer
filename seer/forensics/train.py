"""Train the tamper model.

    python -m seer.forensics.train --data data/synth --out weights/tamper.pt

Reports document-level ROC AUC (the KPI for a review-queue trigger) and
heatmap IoU on tampered validation samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seer.forensics.dataset import TamperDataset
from seer.forensics.model import TamperLoss, TamperNet


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann–Whitney), no sklearn dependency."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    for s in np.unique(scores):
        idx = scores == s
        ranks[idx] = ranks[idx].mean()
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    labels, scores, ious = [], [], []
    for x, tampered, masks in loader:
        doc_logit, hm = model(x.to(device))
        probs = torch.sigmoid(doc_logit).cpu().numpy()
        scores.extend(probs.tolist())
        labels.extend(tampered.numpy().tolist())
        hm_prob = torch.sigmoid(hm).cpu()
        target = F.interpolate(masks, size=hm.shape[2:], mode="area") > 0.25
        for i in range(len(tampered)):
            if tampered[i] > 0.5:
                pred = hm_prob[i, 0] > 0.5
                gt = target[i, 0]
                union = (pred | gt).sum().item()
                if union:
                    ious.append(((pred & gt).sum().item()) / union)
    return {"auc": roc_auc(np.array(labels), np.array(scores)),
            "mask_iou": float(np.mean(ious)) if ious else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("weights/tamper.pt"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl = DataLoader(TamperDataset(args.data, "train"), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(TamperDataset(args.data, "val"), batch_size=args.batch,
                        num_workers=args.workers)

    model = TamperNet().to(device)
    criterion = TamperLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, tampered, masks in train_dl:
            x, tampered, masks = x.to(device), tampered.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            doc_logit, hm = model(x)
            losses = criterion(doc_logit, hm, tampered, masks)
            losses["loss"].backward()
            opt.step()
            running += float(losses["loss"].detach())
        sched.step()

        m = evaluate(model, val_dl, device)
        print(f"epoch {epoch:3d}  loss {running / max(len(train_dl), 1):.4f}  "
              f"val AUC {m['auc']:.4f}  mask IoU {m['mask_iou']:.3f}")
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save({"model": model.state_dict(), "metrics": m, "epoch": epoch},
                       args.out)

    print(f"best val AUC: {best_auc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
