"""Fictional Kenyan personas with realistically formatted identity fields.

Field formats follow publicly documented conventions (an old-format Kenyan
national ID number is 8 digits; passports are one letter + K + 6 digits,
e.g. "AK0123456"). Names are sampled from pools reflecting Kenya's major
naming traditions so the OCR model sees realistic character distributions,
combined with Faker for date/place variety. Every persona is fictional.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from faker import Faker

from seer.icao9303 import TD3Data

# Name pools spanning Kenya's naming traditions (Kikuyu, Luo, Luhya, Kalenjin,
# Kamba, Swahili/coastal, Somali). Deliberately broad so the recognizer does
# not overfit a narrow character n-gram distribution.
FIRST_NAMES_M = [
    "James", "John", "Peter", "David", "Brian", "Kevin", "Dennis", "Samuel",
    "Daniel", "Joseph", "Collins", "Victor", "Felix", "George", "Stephen",
    "Mohamed", "Abdi", "Hassan", "Omar", "Juma", "Baraka", "Kiprono",
    "Kipchoge", "Kiplagat", "Wafula", "Barasa", "Otieno", "Odhiambo",
    "Mwangi", "Kamau", "Njoroge", "Mutua", "Musyoka", "Gitonga",
]
FIRST_NAMES_F = [
    "Mary", "Grace", "Faith", "Mercy", "Joyce", "Esther", "Catherine",
    "Elizabeth", "Ann", "Susan", "Lucy", "Agnes", "Beatrice", "Naomi",
    "Amina", "Fatuma", "Halima", "Zainab", "Wanjiru", "Njeri", "Wambui",
    "Akinyi", "Atieno", "Awuor", "Nekesa", "Nafula", "Chebet", "Cherono",
    "Jepkorir", "Mwikali", "Ndinda", "Nyambura", "Moraa", "Kwamboka",
]
SURNAMES = [
    "Maingi", "Mwangi", "Kamau", "Njoroge", "Kariuki", "Waweru", "Githinji",
    "Odhiambo", "Otieno", "Owino", "Ochieng", "Onyango", "Okoth", "Oduya",
    "Wafula", "Wanyama", "Barasa", "Simiyu", "Wekesa", "Nabwera",
    "Kiprotich", "Kipkorir", "Cheruiyot", "Langat", "Rotich", "Sang",
    "Mutua", "Musyoka", "Kilonzo", "Mwendwa", "Nzomo", "Muema",
    "Hassan", "Abdullahi", "Mohamed", "Farah", "Hussein", "Ali",
    "Nyong'o", "Gathecha", "Wainaina", "Muriuki", "Gitau", "Ndegwa",
]
DISTRICTS = [
    "NAIROBI", "MOMBASA", "KISUMU", "NAKURU", "ELDORET", "THIKA", "NYERI",
    "MACHAKOS", "KAKAMEGA", "KERICHO", "EMBU", "MERU", "KITALE", "GARISSA",
    "KIAMBU", "MURANG'A", "BUNGOMA", "KISII", "NAROK", "KAJIADO",
]


@dataclass(frozen=True)
class Persona:
    surname: str
    given_names: str
    sex: str            # "M" | "F"
    birth_date: date
    id_number: str      # 8-digit national ID number
    serial_number: str  # card serial
    district: str
    place_of_issue: str
    date_of_issue: date
    passport_number: str
    passport_issue: date
    passport_expiry: date

    @property
    def full_name(self) -> str:
        return f"{self.given_names} {self.surname}"

    def td3(self) -> TD3Data:
        return TD3Data(
            surname=self.surname,
            given_names=self.given_names,
            document_number=self.passport_number,
            nationality="KEN",
            issuing_state="KEN",
            birth_date=self.birth_date,
            expiry_date=self.passport_expiry,
            sex=self.sex,
            personal_number=self.id_number,
        )


def sample_persona(rng: random.Random) -> Persona:
    fake = Faker()
    fake.seed_instance(rng.getrandbits(32))

    sex = rng.choice("MF")
    first = rng.choice(FIRST_NAMES_M if sex == "M" else FIRST_NAMES_F)
    # Kenyans commonly carry two given names; mix traditions freely.
    middle_pool = FIRST_NAMES_M if sex == "M" else FIRST_NAMES_F
    given = first if rng.random() < 0.3 else f"{first} {rng.choice(middle_pool)}"
    surname = rng.choice(SURNAMES)

    birth = fake.date_of_birth(minimum_age=18, maximum_age=75)
    # ID issued at age 18+, passport some years later, valid 10 years.
    issue = birth + timedelta(days=int(rng.uniform(18.2, 40) * 365.25))
    issue = min(issue, date(2025, 12, 31))
    p_issue = issue + timedelta(days=rng.randint(0, 3000))
    p_issue = min(p_issue, date(2025, 12, 31))

    return Persona(
        surname=surname,
        given_names=given,
        sex=sex,
        birth_date=birth,
        id_number=f"{rng.randint(10_000_000, 39_999_999)}",
        serial_number=f"{rng.randint(100_000_000, 999_999_999)}",
        district=rng.choice(DISTRICTS),
        place_of_issue=rng.choice(DISTRICTS),
        date_of_issue=issue,
        passport_number=f"{rng.choice('ABC')}K{rng.randint(0, 999_999):06d}",
        passport_issue=p_issue,
        passport_expiry=p_issue + timedelta(days=3652),
    )
