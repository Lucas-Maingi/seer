"""Recognition alphabet and CTC codec.

Index 0 is reserved for the CTC blank. The alphabet covers everything the
specimen templates can emit: uppercase letters, digits, space, and the
punctuation that appears in dates, names and MRZ lines.
"""

from __future__ import annotations

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .'<-"
BLANK = 0

_CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}
_IDX_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}

NUM_CLASSES = len(ALPHABET) + 1  # + blank


def encode(text: str) -> list[int]:
    return [_CHAR_TO_IDX[c] for c in text.upper() if c in _CHAR_TO_IDX]


def decode_greedy(indices: list[int]) -> str:
    """Collapse repeats then drop blanks (standard CTC greedy decoding)."""
    out, prev = [], BLANK
    for i in indices:
        if i != prev and i != BLANK:
            out.append(_IDX_TO_CHAR[i])
        prev = i
    return "".join(out)
