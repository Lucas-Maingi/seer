"""Canonical field ROIs and semantic validation.

After rectification every document sits in a canonical frame, so field
positions are template constants — this registry mirrors the geometry in
:mod:`seer.synth.template` (a consistency test keeps them from drifting).

Validation encodes Kenyan document conventions: an (old-format) national ID
number is 8 digits, card serials are 9 digits, passports are a letter + 'K'
+ 6 digits. A field that reads cleanly but fails its format check is a
strong review signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in canonical pixels

_ID_ROWS = ["id_number", "full_name", "date_of_birth", "sex",
            "district_of_birth", "place_of_issue", "date_of_issue"]
_PASSPORT_COL0 = ["document_number", "surname", "given_names",
                  "nationality", "date_of_birth", "sex"]
_PASSPORT_COL1 = ["date_of_issue", "date_of_expiry", "place_of_birth"]


def _id_rois() -> dict[str, Box]:
    rois: dict[str, Box] = {"serial_number": (770, 100, 990, 148)}
    for i, name in enumerate(_ID_ROWS):
        y = 150 + i * 66
        rois[name] = (364, y + 20, 995, y + 64)
    return rois


def _passport_rois() -> dict[str, Box]:
    rois: dict[str, Box] = {}
    for i, name in enumerate(_PASSPORT_COL0):
        y = 130 + i * 66
        rois[name] = (434, y + 18, 850, y + 62)
    for i, name in enumerate(_PASSPORT_COL1):
        y = 130 + i * 66
        rois[name] = (854, y + 18, 1240, y + 62)
    mrz_top = 880 - 150
    rois["mrz_line1"] = (28, mrz_top - 6, 1244, mrz_top + 52)
    rois["mrz_line2"] = (28, mrz_top + 56, 1244, mrz_top + 114)
    return rois


FIELD_ROIS: dict[str, dict[str, Box]] = {
    "national_id": _id_rois(),
    "passport": _passport_rois(),
}

_NAME_RE = re.compile(r"^[A-Z][A-Z' -]{1,60}$")
_PLACE_RE = _NAME_RE


def _valid_date(text: str, fmts: tuple[str, ...]) -> bool:
    for fmt in fmts:
        try:
            datetime.strptime(text.strip(), fmt)
            return True
        except ValueError:
            continue
    return False


VALIDATORS = {
    "id_number": lambda t: re.fullmatch(r"\d{8}", t) is not None,
    "serial_number": lambda t: re.fullmatch(r"\d{9}", t) is not None,
    "document_number": lambda t: re.fullmatch(r"[A-Z]K\d{6}", t) is not None,
    "sex": lambda t: t in ("M", "F"),
    "nationality": lambda t: t == "KENYAN",
    "full_name": lambda t: _NAME_RE.fullmatch(t) is not None,
    "surname": lambda t: _NAME_RE.fullmatch(t) is not None,
    "given_names": lambda t: _NAME_RE.fullmatch(t) is not None,
    "district_of_birth": lambda t: _PLACE_RE.fullmatch(t) is not None,
    "place_of_birth": lambda t: _PLACE_RE.fullmatch(t) is not None,
    "place_of_issue": lambda t: _PLACE_RE.fullmatch(t) is not None,
    "date_of_birth": lambda t: _valid_date(t, ("%d.%m.%Y", "%d %b %y")),
    "date_of_issue": lambda t: _valid_date(t, ("%d.%m.%Y", "%d %b %y")),
    "date_of_expiry": lambda t: _valid_date(t, ("%d %b %y",)),
}


@dataclass
class FieldRead:
    name: str
    text: str
    confidence: float
    format_valid: bool


def validate(name: str, text: str) -> bool:
    fn = VALIDATORS.get(name)
    return True if fn is None else bool(fn(text.strip()))
