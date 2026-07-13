"""ICAO Doc 9303 machine-readable-zone codec.

Shared by the synthetic engine (which composes MRZs for specimen passports)
and the OCR stage (which parses and validates what the recognizer reads).
Keeping compose and parse in one module means the check-digit arithmetic can
never drift between the two sides.

Implements the TD3 (passport, 2 lines x 44 chars) format.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

MRZ_FILLER = "<"
_WEIGHTS = (7, 3, 1)


def _char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - ord("A") + 10
    if c == MRZ_FILLER:
        return 0
    raise ValueError(f"invalid MRZ character: {c!r}")


def check_digit(field: str) -> int:
    """ICAO 9303 check digit: weighted sum mod 10 with weights 7,3,1 repeating."""
    return sum(_char_value(c) * _WEIGHTS[i % 3] for i, c in enumerate(field)) % 10


def _pad(text: str, width: int) -> str:
    return text[:width].ljust(width, MRZ_FILLER)


def transliterate(name: str) -> str:
    """Fold a name into the MRZ alphabet (A-Z and filler)."""
    out = []
    for c in name.upper():
        if "A" <= c <= "Z":
            out.append(c)
        elif c in " -'":
            out.append(MRZ_FILLER)
        # anything else (accents already folded upstream) is dropped
    return "".join(out)


def _yymmdd(d: date) -> str:
    return d.strftime("%y%m%d")


@dataclass(frozen=True)
class TD3Data:
    surname: str
    given_names: str
    document_number: str
    nationality: str  # 3-letter code, e.g. KEN
    issuing_state: str
    birth_date: date
    expiry_date: date
    sex: str  # "M" | "F" | "<"
    personal_number: str = ""

    def compose(self) -> tuple[str, str]:
        """Render the two 44-character MRZ lines, computing all check digits."""
        primary = transliterate(self.surname)
        secondary = transliterate(self.given_names).replace(MRZ_FILLER, "<")
        name_field = _pad(f"{primary}<<{secondary}", 39)
        line1 = f"P<{_pad(self.issuing_state, 3)}{name_field}"

        doc = _pad(self.document_number, 9)
        dob = _yymmdd(self.birth_date)
        exp = _yymmdd(self.expiry_date)
        pers = _pad(self.personal_number, 14)
        composite_input = f"{doc}{check_digit(doc)}{dob}{check_digit(dob)}" \
                          f"{exp}{check_digit(exp)}{pers}{check_digit(pers)}"
        line2 = (
            f"{doc}{check_digit(doc)}{_pad(self.nationality, 3)}"
            f"{dob}{check_digit(dob)}{self.sex}"
            f"{exp}{check_digit(exp)}{pers}{check_digit(pers)}"
        )
        # composite check covers positions 1-10, 14-20 and 22-43 of line 2
        line2 += str(check_digit(composite_input))
        assert len(line1) == 44 and len(line2) == 44
        return line1, line2


@dataclass(frozen=True)
class TD3Parse:
    """Result of parsing a TD3 MRZ, with per-field checksum verdicts."""

    surname: str
    given_names: str
    document_number: str
    nationality: str
    issuing_state: str
    birth_date: str   # YYMMDD as printed; century resolution is policy, not codec
    expiry_date: str
    sex: str
    personal_number: str
    checks: dict[str, bool]

    @property
    def all_valid(self) -> bool:
        return all(self.checks.values())


def parse_td3(line1: str, line2: str) -> TD3Parse:
    """Parse and validate a TD3 MRZ. Raises ValueError on malformed structure;
    checksum failures are reported in ``checks`` rather than raised, because a
    failed check digit is a *signal* (bad OCR or tampering), not an error."""
    if len(line1) != 44 or len(line2) != 44:
        raise ValueError("TD3 lines must be exactly 44 characters")
    if not line1.startswith("P"):
        raise ValueError("not a TD3 passport MRZ")

    issuing = line1[2:5].replace(MRZ_FILLER, "")
    name_field = line1[5:44]
    primary, _, secondary = name_field.partition("<<")
    surname = primary.replace(MRZ_FILLER, " ").strip()
    given = secondary.replace(MRZ_FILLER, " ").strip()

    doc = line2[0:9]
    doc_cd = line2[9]
    nationality = line2[10:13].replace(MRZ_FILLER, "")
    dob = line2[13:19]
    dob_cd = line2[19]
    sex = line2[20]
    exp = line2[21:27]
    exp_cd = line2[27]
    pers = line2[28:42]
    pers_cd = line2[42]
    comp_cd = line2[43]

    def _ok(field: str, cd: str) -> bool:
        return cd.isdigit() and check_digit(field) == int(cd)

    composite_input = f"{doc}{doc_cd}{dob}{dob_cd}{exp}{exp_cd}{pers}{pers_cd}"
    checks = {
        "document_number": _ok(doc, doc_cd),
        "birth_date": _ok(dob, dob_cd),
        "expiry_date": _ok(exp, exp_cd),
        "personal_number": _ok(pers, pers_cd),
        "composite": _ok(composite_input, comp_cd),
    }
    return TD3Parse(
        surname=surname,
        given_names=given,
        document_number=doc.replace(MRZ_FILLER, ""),
        nationality=nationality,
        issuing_state=issuing,
        birth_date=dob,
        expiry_date=exp,
        sex=sex if sex != MRZ_FILLER else "",
        personal_number=pers.replace(MRZ_FILLER, ""),
        checks=checks,
    )
