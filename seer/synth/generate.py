"""Dataset generation CLI.

    python -m seer.synth.generate --out data/synth --count 5000 \
        [--faces path/to/synthetic_faces] [--tamper-frac 0.35] [--seed 7]

Emits, per sample i:

    scenes/{i:06d}.jpg      the "phone photo" (localization input)
    canonical/{i:06d}.jpg   the fronto-parallel document (OCR/forensics input;
                            --canonical-format png for lossless)
    masks/{i:06d}.png       tamper mask in canonical coords (tampered only)
    labels/{i:06d}.json     all ground truth
    index.jsonl             one summary line per sample, with train/val split

The split is a deterministic hash of the sample id, so regenerating with more
samples never moves an existing id across the split boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from seer.synth.compose import compose_scene
from seer.synth.face import make_provider
from seer.synth.identity import sample_persona
from seer.synth.tamper import apply_random_attack
from seer.synth.template import render_national_id, render_passport


def _split_of(sample_id: str, val_frac: float = 0.1) -> str:
    h = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_frac else "train"


def generate(out: Path, count: int, faces_dir: str | None,
             tamper_frac: float, passport_frac: float, seed: int,
             canonical_format: str = "jpg") -> None:
    rng = random.Random(seed)
    provider = make_provider(faces_dir)
    for sub in ("scenes", "canonical", "masks", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    index_path = out / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as index:
        for i in tqdm(range(count), desc="generating"):
            sid = f"{i:06d}"
            persona = sample_persona(rng)
            portrait = provider.sample(rng)

            if rng.random() < passport_frac:
                render = render_passport(persona, portrait, rng)
            else:
                render = render_national_id(persona, portrait, rng)

            tampered = rng.random() < tamper_frac
            attack = None
            if tampered:
                impostor = provider.sample(rng)
                result = apply_random_attack(render, impostor, rng)
                render, attack = result.render, result.attack
                cv2.imwrite(str(out / "masks" / f"{sid}.png"), result.mask)

            scene = compose_scene(render, rng)
            cv2.imwrite(str(out / "scenes" / f"{sid}.jpg"),
                        cv2.cvtColor(scene.image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            if canonical_format == "png":
                render.image.save(out / "canonical" / f"{sid}.png")
            else:
                # q95: visually lossless for OCR, ~6x smaller than PNG, and
                # matches reality — KYC uploads always carry a JPEG history
                render.image.save(out / "canonical" / f"{sid}.jpg",
                                  quality=95, subsampling=0)

            label = {
                "id": sid,
                "kind": render.kind,
                "split": _split_of(sid),
                "tampered": tampered,
                "attack": attack,
                "corners": scene.corners.tolist(),
                "homography": scene.homography.tolist(),
                "scene_size": list(scene.image.shape[1::-1]),
                "canonical_size": list(render.image.size),
                "photo_box": list(render.photo_box),
                "portrait_identity": render.portrait_identity,
                "mrz": list(render.mrz) if render.mrz else None,
                "fields": [
                    {"name": f.name, "text": f.text, "quad": f.quad}
                    for f in render.fields
                ],
            }
            (out / "labels" / f"{sid}.json").write_text(
                json.dumps(label), encoding="utf-8")
            index.write(json.dumps({
                "id": sid, "kind": render.kind, "split": label["split"],
                "tampered": tampered, "attack": attack,
            }) + "\n")

    print(f"wrote {count} samples to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--faces", type=str, default=None,
                    help="directory of synthetic face images (SFHQ etc.); "
                         "omit to use the procedural portrait generator")
    ap.add_argument("--tamper-frac", type=float, default=0.35)
    ap.add_argument("--passport-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--canonical-format", choices=["jpg", "png"], default="jpg",
                    help="jpg q95 is ~6x smaller and matches real upload "
                         "compression history; png for lossless")
    args = ap.parse_args()
    np.random.seed(args.seed)
    generate(args.out, args.count, args.faces,
             args.tamper_frac, args.passport_frac, args.seed,
             canonical_format=args.canonical_format)


if __name__ == "__main__":
    main()
