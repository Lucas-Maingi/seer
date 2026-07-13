"""Train the corner localization model.

    python -m seer.localize.train --data data/synth --epochs 30 \
        --out weights/corner.pt

Reports validation mean corner error in *scene pixels* and mean quad IoU —
the numbers that actually predict whether rectified OCR will work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seer.localize.dataset import CornerDataset, denormalize_corners
from seer.localize.model import CornerLoss, CornerNet
from seer.localize.rectify import quad_iou


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader,
             device: torch.device, scene_size: tuple[int, int]) -> dict[str, float]:
    model.eval()
    errs, ious = [], []
    w, h = scene_size
    for x, target in loader:
        coords, _ = model(x.to(device))
        pred_px = denormalize_corners(coords, w, h)
        gt_px = denormalize_corners(target, w, h)
        errs.extend(np.linalg.norm(pred_px - gt_px, axis=-1).mean(axis=-1).tolist())
        ious.extend(quad_iou(p, g, (w, h)) for p, g in zip(pred_px, gt_px))
    return {"corner_err_px": float(np.mean(errs)), "quad_iou": float(np.mean(ious))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("weights/corner.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--js-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = CornerDataset(args.data, "train")
    val_ds = CornerDataset(args.data, "val")
    # scene size is constant across the synth set; read it once
    sample_label = json.loads(next((args.data / "labels").glob("*.json")).read_text())
    scene_size = tuple(sample_label["scene_size"])

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    model = CornerNet(pretrained=True).to(device)
    criterion = CornerLoss(js_weight=args.js_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_err = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "l1": 0.0, "js": 0.0}
        for x, target in train_dl:
            x, target = x.to(device), target.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                coords, logits = model(x)
                losses = criterion(coords, logits, target)
            scaler.scale(losses["loss"]).backward()
            scaler.step(opt)
            scaler.update()
            for k in running:
                running[k] += float(losses[k].detach())
        sched.step()

        n = max(len(train_dl), 1)
        metrics = evaluate(model, val_dl, device, scene_size)
        print(f"epoch {epoch:3d}  loss {running['loss']/n:.4f} "
              f"(l1 {running['l1']/n:.4f} js {running['js']/n:.4f})  "
              f"val corner_err {metrics['corner_err_px']:.2f}px "
              f"iou {metrics['quad_iou']:.4f}")

        if metrics["corner_err_px"] < best_err:
            best_err = metrics["corner_err_px"]
            torch.save({"model": model.state_dict(), "metrics": metrics,
                        "epoch": epoch}, args.out)

    print(f"best val corner error: {best_err:.2f}px -> {args.out}")


if __name__ == "__main__":
    main()
