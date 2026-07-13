"""Specimen document renderers with dense ground truth.

Renders two Kenyan document types at canonical (fronto-parallel) resolution:

- national ID card, 1000 x 631 px (85.6 x 54 mm aspect)
- passport data page (TD3), 1250 x 880 px

Every text field is recorded as (name, text, quad) in canonical coordinates,
the portrait box is recorded for the face stage, and passports carry a fully
check-digited MRZ composed by :mod:`seer.icao9303`. A diagonal SPECIMEN
watermark is baked into the artwork of both — it is part of the document,
not an overlay, so it survives every downstream augmentation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from seer.synth import fonts
from seer.synth.face import Portrait
from seer.synth.identity import Persona

ID_SIZE = (1000, 631)
PASSPORT_SIZE = (1250, 880)

Quad = list[list[float]]  # 4 x [x, y], clockwise from top-left


@dataclass
class FieldGT:
    name: str
    text: str
    quad: Quad


@dataclass
class DocumentRender:
    kind: str  # "national_id" | "passport"
    image: Image.Image
    fields: list[FieldGT] = field(default_factory=list)
    photo_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    portrait_identity: str = ""
    mrz: tuple[str, str] | None = None

    def corners(self) -> Quad:
        w, h = self.image.size
        return [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]


def _text_quad(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> Quad:
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=font)
    return [[float(x0), float(y0)], [float(x1), float(y0)],
            [float(x1), float(y1)], [float(x0), float(y1)]]


def _guilloche(size: tuple[int, int], rng: random.Random,
               palette: list[tuple[int, int, int]]) -> Image.Image:
    """Layered rose/Lissajous curve field — the visual language of security
    printing, drawn as decorative art only (no genuine pattern is copied)."""
    w, h = size
    img = Image.new("RGB", size, (248, 246, 240))
    d = ImageDraw.Draw(img, "RGBA")
    t = np.linspace(0, 2 * math.pi, 900)
    for _ in range(rng.randint(8, 14)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rx, ry = rng.uniform(w * 0.2, w * 0.7), rng.uniform(h * 0.2, h * 0.7)
        a, b = rng.randint(3, 9), rng.randint(2, 7)
        phase = rng.uniform(0, math.pi)
        xs = cx + rx * np.cos(a * t + phase) * np.cos(t)
        ys = cy + ry * np.sin(b * t) * np.sin(t + phase)
        color = palette[rng.randrange(len(palette))] + (rng.randint(14, 30),)
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=color, width=1)
    return img


def _specimen_watermark(img: Image.Image) -> None:
    """Bake a repeated diagonal SPECIMEN into the artwork."""
    w, h = img.size
    layer = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = fonts.load("sans_bold", max(28, w // 18))
    step_y = max(90, h // 4)
    for i, y in enumerate(range(0, h * 2, step_y)):
        for x in range(-w, w * 2, w // 2 + 120):
            d.text((x + (i % 2) * 160, y), "SPECIMEN", font=f, fill=(120, 120, 120, 46))
    layer = layer.rotate(18, resample=Image.BICUBIC)
    left = (layer.width - w) // 2
    top = (layer.height - h) // 2
    alpha = layer.crop((left, top, left + w, top + h)).split()[3]
    img.paste((120, 120, 120), (0, 0), alpha)


def _paste_portrait(img: Image.Image, portrait: Portrait,
                    box: tuple[int, int, int, int], grayish: bool) -> None:
    x0, y0, x1, y1 = box
    p = ImageOps.fit(portrait.image, (x1 - x0, y1 - y0), Image.BICUBIC)
    if grayish:
        g = ImageOps.grayscale(p)
        p = ImageOps.colorize(g, black=(40, 40, 45), white=(235, 235, 240))
    img.paste(p, (x0, y0))
    ImageDraw.Draw(img).rectangle(box, outline=(90, 90, 90), width=2)


def render_national_id(persona: Persona, portrait: Portrait,
                       rng: random.Random) -> DocumentRender:
    w, h = ID_SIZE
    palette = [(30, 110, 60), (150, 90, 40), (60, 60, 130)]
    img = _guilloche(ID_SIZE, rng, palette)
    d = ImageDraw.Draw(img)
    gt = DocumentRender(kind="national_id", image=img,
                        portrait_identity=portrait.identity)

    f_head = fonts.load("serif", 34)
    f_label = fonts.load("sans", 22)
    f_value = fonts.load("sans_bold", 30, rng)
    f_serial = fonts.load("mono", 30)

    d.text((w // 2, 28), "JAMHURI YA KENYA", font=f_head, fill=(20, 60, 30), anchor="mm")
    d.text((w // 2, 66), "REPUBLIC OF KENYA", font=f_head, fill=(20, 60, 30), anchor="mm")
    d.line([(40, 92), (w - 40, 92)], fill=(20, 60, 30), width=3)

    d.text((w - 500, 108), "SERIAL NUMBER:", font=f_serial, fill=(60, 30, 30))
    serial_xy = (w - 220, 108)
    d.text(serial_xy, persona.serial_number, font=f_serial, fill=(60, 30, 30))
    gt.fields.append(FieldGT("serial_number", persona.serial_number,
                             _text_quad(d, serial_xy, persona.serial_number, f_serial)))

    photo_box = (52, 160, 52 + 280, 160 + 350)
    _paste_portrait(img, portrait, photo_box, grayish=True)
    gt.photo_box = photo_box

    rows = [
        ("id_number", "ID NUMBER", persona.id_number),
        ("full_name", "FULL NAMES", persona.full_name.upper()),
        ("date_of_birth", "DATE OF BIRTH", persona.birth_date.strftime("%d.%m.%Y")),
        ("sex", "SEX", persona.sex),
        ("district_of_birth", "DISTRICT OF BIRTH", persona.district),
        ("place_of_issue", "PLACE OF ISSUE", persona.place_of_issue),
        ("date_of_issue", "DATE OF ISSUE", persona.date_of_issue.strftime("%d.%m.%Y")),
    ]
    x_label, y = 370, 150
    for name, label, value in rows:
        d.text((x_label, y), label, font=f_label, fill=(70, 70, 70))
        vy = y + 26
        d.text((x_label, vy), value, font=f_value, fill=(15, 15, 20))
        gt.fields.append(FieldGT(name, value, _text_quad(d, (x_label, vy), value, f_value)))
        y = vy + 40

    # signature strip + fingerprint box (drawn, not fielded)
    d.line([(70, 560), (300, 560)], fill=(30, 30, 60), width=2)
    d.text((90, 566), "HOLDER'S SIGNATURE", font=f_label, fill=(70, 70, 70))
    d.rectangle([w - 200, h - 190, w - 60, h - 50], outline=(90, 90, 90), width=2)

    _specimen_watermark(img)
    return gt


def render_passport(persona: Persona, portrait: Portrait,
                    rng: random.Random) -> DocumentRender:
    w, h = PASSPORT_SIZE
    palette = [(140, 40, 40), (60, 60, 130), (120, 100, 40)]
    img = _guilloche(PASSPORT_SIZE, rng, palette)
    d = ImageDraw.Draw(img)
    gt = DocumentRender(kind="passport", image=img,
                        portrait_identity=portrait.identity)

    f_head = fonts.load("serif", 36)
    f_label = fonts.load("sans", 20)
    f_value = fonts.load("sans_bold", 28, rng)
    f_mrz = fonts.load("mono", 34)

    d.text((w // 2, 34), "REPUBLIC OF KENYA", font=f_head, fill=(90, 20, 20), anchor="mm")
    d.text((w // 2, 76), "PASSPORT · PASIPOTI", font=f_head, fill=(90, 20, 20), anchor="mm")

    photo_box = (60, 130, 60 + 330, 130 + 430)
    _paste_portrait(img, portrait, photo_box, grayish=False)
    gt.photo_box = photo_box

    rows = [
        ("document_number", "PASSPORT NO.", persona.passport_number),
        ("surname", "SURNAME", persona.surname.upper()),
        ("given_names", "GIVEN NAMES", persona.given_names.upper()),
        ("nationality", "NATIONALITY", "KENYAN"),
        ("date_of_birth", "DATE OF BIRTH", persona.birth_date.strftime("%d %b %y").upper()),
        ("sex", "SEX", persona.sex),
        ("date_of_issue", "DATE OF ISSUE", persona.passport_issue.strftime("%d %b %y").upper()),
        ("date_of_expiry", "DATE OF EXPIRY", persona.passport_expiry.strftime("%d %b %y").upper()),
        ("place_of_birth", "PLACE OF BIRTH", persona.district),
    ]
    col_x = [440, 860]
    col_y = [130, 130]
    for i, (name, label, value) in enumerate(rows):
        c = 0 if i < 6 else 1
        x, y = col_x[c], col_y[c]
        d.text((x, y), label, font=f_label, fill=(80, 60, 60))
        vy = y + 24
        d.text((x, vy), value, font=f_value, fill=(15, 15, 25))
        gt.fields.append(FieldGT(name, value, _text_quad(d, (x, vy), value, f_value)))
        col_y[c] = vy + 42

    # machine readable zone
    line1, line2 = persona.td3().compose()
    gt.mrz = (line1, line2)
    mrz_top = h - 150
    d.rectangle([0, mrz_top - 18, w, h], fill=(252, 252, 250))
    for j, line in enumerate((line1, line2)):
        xy = (36, mrz_top + j * 62)
        d.text(xy, line, font=f_mrz, fill=(20, 20, 20))
        gt.fields.append(FieldGT(f"mrz_line{j + 1}", line, _text_quad(d, xy, line, f_mrz)))

    _specimen_watermark(img)
    return gt
