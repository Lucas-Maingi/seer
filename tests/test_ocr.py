"""OCR stage invariants: codec, model shapes, ROI/template consistency."""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seer.ocr.charset import decode_greedy, encode  # noqa: E402
from seer.ocr.crnn import CRNN, LINE_WIDTH, TIME_STEPS, prepare_line  # noqa: E402
from seer.ocr.fields import FIELD_ROIS, validate  # noqa: E402


def test_ctc_codec_roundtrip():
    text = "MWIKALI KWAMBOKA 12345678 <<"
    ids = encode(text)
    # simulate a perfect CTC path: each char emitted once, blanks between
    path = []
    for i in ids:
        path.extend([0, i])
    assert decode_greedy(path) == text


def test_ctc_collapse_repeats():
    # 'AAB' as raw frame output must collapse to 'AB', but 'A blank A' stays 'AA'
    a, b = encode("A")[0], encode("B")[0]
    assert decode_greedy([a, a, b]) == "AB"
    assert decode_greedy([a, 0, a]) == "AA"


def test_crnn_time_axis_fits_mrz():
    model = CRNN().eval()
    x = torch.randn(2, 1, 32, LINE_WIDTH)
    with torch.no_grad():
        logp = model(x)
    assert logp.shape[0] == TIME_STEPS
    # CTC needs T >= 2*len+1 for a 44-char MRZ line
    assert TIME_STEPS >= 2 * 44 + 1
    # log-probs normalize
    assert torch.allclose(logp.exp().sum(-1), torch.ones(1), atol=1e-4)


def test_prepare_line_shapes_and_stats():
    crop = (np.random.rand(40, 700) * 255).astype(np.uint8)
    line = prepare_line(crop)
    assert line.shape == (32, LINE_WIDTH)
    assert abs(float(line.mean())) < 0.2


def test_field_rois_contain_rendered_quads():
    """The ROI registry must contain every quad the templates actually emit."""
    from seer.synth.face import ProceduralFaces
    from seer.synth.identity import sample_persona
    from seer.synth.template import render_national_id, render_passport

    for seed in range(4):
        rng = random.Random(seed)
        persona = sample_persona(rng)
        portrait = ProceduralFaces().sample(rng)
        for render in (render_national_id(persona, portrait, rng),
                       render_passport(persona, portrait, rng)):
            rois = FIELD_ROIS[render.kind]
            for f in render.fields:
                q = np.array(f.quad)
                x0, y0, x1, y1 = rois[f.name]
                assert q[:, 0].min() >= x0 and q[:, 0].max() <= x1, \
                    f"{render.kind}/{f.name} x out of ROI"
                assert q[:, 1].min() >= y0 and q[:, 1].max() <= y1, \
                    f"{render.kind}/{f.name} y out of ROI"


def test_validators():
    assert validate("id_number", "12345678")
    assert not validate("id_number", "1234567")
    assert validate("document_number", "AK012345")
    assert not validate("document_number", "XX012345")
    assert validate("date_of_birth", "04.06.1996")
    assert validate("date_of_birth", "06 DEC 79")
    assert not validate("date_of_birth", "31.02.1996")
    assert validate("sex", "F") and not validate("sex", "X")
    assert validate("full_name", "MWIKALI KWAMBOKA MUTUA")
    assert validate("surname", "NYONG'O")
