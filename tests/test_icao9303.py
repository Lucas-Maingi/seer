"""MRZ codec tests against the worked example in ICAO Doc 9303 Part 4."""

from datetime import date

from seer.icao9303 import TD3Data, check_digit, parse_td3

# Official specimen from ICAO Doc 9303 (Utopia / Anna Maria Eriksson)
ICAO_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
ICAO_LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_check_digit_icao_vectors():
    # per-field check digits from the published example
    assert check_digit("L898902C3") == 6
    assert check_digit("740812") == 2
    assert check_digit("120415") == 9


def test_parse_official_specimen():
    p = parse_td3(ICAO_LINE1, ICAO_LINE2)
    assert p.surname == "ERIKSSON"
    assert p.given_names == "ANNA MARIA"
    assert p.document_number == "L898902C3"
    assert p.nationality == "UTO"
    assert p.birth_date == "740812"
    assert p.sex == "F"
    assert p.all_valid


def test_compose_parse_roundtrip():
    data = TD3Data(
        surname="Maingi", given_names="Lucas Kamau",
        document_number="AK0123456", nationality="KEN", issuing_state="KEN",
        birth_date=date(1995, 3, 14), expiry_date=date(2033, 3, 13),
        sex="M", personal_number="23456789",
    )
    l1, l2 = data.compose()
    assert len(l1) == 44 and len(l2) == 44
    p = parse_td3(l1, l2)
    assert p.all_valid
    assert p.surname == "MAINGI"
    assert p.given_names == "LUCAS KAMAU"
    assert p.document_number == "AK0123456"
    assert p.personal_number == "23456789"
    assert p.birth_date == "950314"


def test_single_digit_corruption_is_detected():
    data = TD3Data(
        surname="Otieno", given_names="Grace",
        document_number="BK0555123", nationality="KEN", issuing_state="KEN",
        birth_date=date(1988, 11, 2), expiry_date=date(2031, 6, 30),
        sex="F", personal_number="30112233",
    )
    l1, l2 = data.compose()
    # corrupt one digit of the birth date the way a bad OCR read would
    bad = l2[:14] + ("0" if l2[14] != "0" else "1") + l2[15:]
    p = parse_td3(l1, bad)
    assert not p.checks["birth_date"]
    assert not p.all_valid
