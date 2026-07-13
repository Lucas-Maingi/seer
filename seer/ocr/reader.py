"""Document reader: rectified image -> validated fields (+ MRZ cross-check).

For passports the MRZ is parsed with ICAO 9303 check digits and then
cross-referenced against the visual inspection zone (VIZ). Agreement between
two independently read zones is the strongest OCR-level authenticity signal
a single image can give you: a fraudster who edits the printed date of birth
but not the MRZ (or vice versa) trips either a check digit or the VIZ/MRZ
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch

from seer.icao9303 import parse_td3
from seer.ocr.crnn import CRNN, recognize
from seer.ocr.fields import FIELD_ROIS, FieldRead, validate


@dataclass
class DocumentReadResult:
    kind: str
    fields: dict[str, FieldRead]
    mrz_valid: bool | None = None          # None for documents without MRZ
    mrz_checks: dict[str, bool] = field(default_factory=dict)
    viz_mrz_consistency: dict[str, bool] = field(default_factory=dict)

    @property
    def mean_confidence(self) -> float:
        if not self.fields:
            return 0.0
        return float(np.mean([f.confidence for f in self.fields.values()]))


def _to_mrz_date(viz_date: str) -> str | None:
    """'06 DEC 79' or '04.06.1996' -> 'YYMMDD'."""
    for fmt in ("%d %b %y", "%d.%m.%Y"):
        try:
            return datetime.strptime(viz_date.strip().title(), fmt.replace("%b", "%b")) \
                .strftime("%y%m%d")
        except ValueError:
            continue
    return None


class DocumentReader:
    def __init__(self, model: CRNN, device: torch.device | str = "cpu"):
        self.model = model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

    def read(self, canonical_gray: np.ndarray, kind: str) -> DocumentReadResult:
        rois = FIELD_ROIS[kind]
        names = list(rois)
        crops = [canonical_gray[y0:y1, x0:x1] for (x0, y0, x1, y1) in rois.values()]
        recognized = recognize(self.model, crops, self.device)

        fields: dict[str, FieldRead] = {}
        for name, (text, conf) in zip(names, recognized):
            text = text.strip()
            if not name.startswith("mrz_"):
                fields[name] = FieldRead(name, text, conf, validate(name, text))

        result = DocumentReadResult(kind=kind, fields=fields)
        if kind == "passport":
            self._attach_mrz(result, dict(zip(names, recognized)))
        return result

    def _attach_mrz(self, result: DocumentReadResult,
                    reads: dict[str, tuple[str, float]]) -> None:
        l1, c1 = reads.get("mrz_line1", ("", 0.0))
        l2, c2 = reads.get("mrz_line2", ("", 0.0))
        l1 = l1.replace(" ", "").ljust(44, "<")[:44]
        l2 = l2.replace(" ", "").ljust(44, "<")[:44]
        result.fields["mrz_line1"] = FieldRead("mrz_line1", l1, c1, True)
        result.fields["mrz_line2"] = FieldRead("mrz_line2", l2, c2, True)
        try:
            parsed = parse_td3(l1, l2)
        except ValueError:
            result.mrz_valid = False
            return
        result.mrz_checks = parsed.checks
        result.mrz_valid = parsed.all_valid

        viz = {k: v.text for k, v in result.fields.items()}
        cons: dict[str, bool] = {}
        if "document_number" in viz:
            cons["document_number"] = viz["document_number"] == parsed.document_number
        if "surname" in viz:
            cons["surname"] = viz["surname"].replace("'", "") == \
                parsed.surname.replace(" ", "").replace("'", "") or \
                viz["surname"] == parsed.surname
        if "sex" in viz:
            cons["sex"] = viz["sex"] == parsed.sex
        dob = _to_mrz_date(viz.get("date_of_birth", ""))
        if dob is not None:
            cons["date_of_birth"] = dob == parsed.birth_date
        exp = _to_mrz_date(viz.get("date_of_expiry", ""))
        if exp is not None:
            cons["date_of_expiry"] = exp == parsed.expiry_date
        result.viz_mrz_consistency = cons
