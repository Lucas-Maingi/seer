"""Forgery synthesis on canonical document renders.

Each attack mimics a documented fraud pattern and returns the tampered
document plus a per-pixel binary mask of the manipulated region — the label
the forensics model trains against. Attacks operate in *canonical* space,
before scene compositing, exactly where a real fraudster edits a scan.

Attacks:
- photo_swap    a different identity's portrait pasted over the original
                (the classic stolen-document attack)
- text_splice   one field re-rendered with a subtly wrong font/weight/color
                (date-of-birth and ID-number edits)
- copy_move     a region cloned elsewhere on the document (covering a stain,
                duplicating a stamp)
- recompress    a rectangular region round-tripped through aggressive JPEG
                (a paste from a screenshot; leaves a quality seam)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from seer.synth import fonts
from seer.synth.face import Portrait
from seer.synth.template import DocumentRender

ATTACKS = ("photo_swap", "text_splice", "copy_move", "recompress")


@dataclass
class TamperResult:
    render: DocumentRender      # image replaced by the tampered version
    attack: str
    mask: np.ndarray            # HxW uint8, 255 where manipulated


def _blank_mask(img: Image.Image) -> np.ndarray:
    return np.zeros((img.height, img.width), np.uint8)


def photo_swap(render: DocumentRender, impostor: Portrait,
               rng: random.Random) -> TamperResult:
    img = render.image.copy()
    x0, y0, x1, y1 = render.photo_box
    # inset slightly: fraudsters paste inside the frame line
    inset = rng.randint(2, 6)
    box = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
    from PIL import ImageOps
    p = ImageOps.fit(impostor.image, (box[2] - box[0], box[3] - box[1]), Image.BICUBIC)
    if render.kind == "national_id":
        g = ImageOps.grayscale(p)
        p = ImageOps.colorize(g, black=(40, 40, 45), white=(235, 235, 240))
    img.paste(p, (box[0], box[1]))
    mask = _blank_mask(img)
    mask[box[1]:box[3], box[0]:box[2]] = 255
    out = DocumentRender(kind=render.kind, image=img, fields=render.fields,
                         photo_box=render.photo_box,
                         portrait_identity=impostor.identity, mrz=render.mrz)
    return TamperResult(out, "photo_swap", mask)


def text_splice(render: DocumentRender, rng: random.Random) -> TamperResult:
    img = render.image.copy()
    d = ImageDraw.Draw(img)
    # target a value field, prefer numeric ones (that's what gets forged)
    numeric = [f for f in render.fields
               if f.name in ("id_number", "date_of_birth", "date_of_expiry",
                             "document_number", "serial_number")]
    tgt = rng.choice(numeric or render.fields)
    (x0, y0), _, (x1, y1), _ = tgt.quad
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    # sample the local paper color and paint over the original value
    patch = np.asarray(img)[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4]
    paper = tuple(int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
    d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=paper)

    # re-render a mutated value with a slightly wrong face/size — the
    # inconsistency a font-forensics pass should catch
    new_text = _mutate_text(tgt.text, rng)
    f = fonts.load(rng.choice(["sans", "serif", "sans_bold"]),
                   max(12, (y1 - y0) + rng.randint(-3, 2)), rng)
    d.text((x0 + rng.randint(-2, 2), y0 + rng.randint(-2, 2)), new_text,
           font=f, fill=(15, 15, 20))

    mask = _blank_mask(img)
    mask[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4] = 255
    out = DocumentRender(kind=render.kind, image=img, fields=render.fields,
                         photo_box=render.photo_box,
                         portrait_identity=render.portrait_identity, mrz=render.mrz)
    return TamperResult(out, "text_splice", mask)


def _mutate_text(text: str, rng: random.Random) -> str:
    chars = list(text)
    digit_idx = [i for i, c in enumerate(chars) if c.isdigit()]
    if digit_idx:
        i = rng.choice(digit_idx)
        chars[i] = rng.choice([c for c in "0123456789" if c != chars[i]])
    elif chars:
        i = rng.randrange(len(chars))
        if chars[i].isalpha():
            chars[i] = rng.choice([c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                   if c != chars[i].upper()])
    return "".join(chars)


def copy_move(render: DocumentRender, rng: random.Random) -> TamperResult:
    arr = np.asarray(render.image).copy()
    h, w = arr.shape[:2]
    pw, ph = rng.randint(w // 12, w // 5), rng.randint(h // 12, h // 5)
    sx, sy = rng.randint(0, w - pw), rng.randint(0, h - ph)
    dx, dy = rng.randint(0, w - pw), rng.randint(0, h - ph)
    arr[dy:dy + ph, dx:dx + pw] = arr[sy:sy + ph, sx:sx + pw]
    mask = np.zeros((h, w), np.uint8)
    mask[dy:dy + ph, dx:dx + pw] = 255
    out = DocumentRender(kind=render.kind, image=Image.fromarray(arr),
                         fields=render.fields, photo_box=render.photo_box,
                         portrait_identity=render.portrait_identity, mrz=render.mrz)
    return TamperResult(out, "copy_move", mask)


def recompress(render: DocumentRender, rng: random.Random) -> TamperResult:
    arr = np.asarray(render.image).copy()
    h, w = arr.shape[:2]
    pw, ph = rng.randint(w // 8, w // 3), rng.randint(h // 8, h // 3)
    x, y = rng.randint(0, w - pw), rng.randint(0, h - ph)
    region = cv2.cvtColor(arr[y:y + ph, x:x + pw], cv2.COLOR_RGB2BGR)
    q = rng.randint(25, 55)
    _, buf = cv2.imencode(".jpg", region, [cv2.IMWRITE_JPEG_QUALITY, q])
    arr[y:y + ph, x:x + pw] = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR),
                                           cv2.COLOR_BGR2RGB)
    mask = np.zeros((h, w), np.uint8)
    mask[y:y + ph, x:x + pw] = 255
    out = DocumentRender(kind=render.kind, image=Image.fromarray(arr),
                         fields=render.fields, photo_box=render.photo_box,
                         portrait_identity=render.portrait_identity, mrz=render.mrz)
    return TamperResult(out, "recompress", mask)


def apply_random_attack(render: DocumentRender, impostor: Portrait,
                        rng: random.Random) -> TamperResult:
    attack = rng.choice(ATTACKS)
    if attack == "photo_swap":
        return photo_swap(render, impostor, rng)
    if attack == "text_splice":
        return text_splice(render, rng)
    if attack == "copy_move":
        return copy_move(render, rng)
    return recompress(render, rng)
