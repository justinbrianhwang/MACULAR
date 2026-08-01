"""PII value generators with disjoint families (proposal 14.4).

Families A / B / C / D / E draw from *disjoint* value pools so that:
  - train uses family A, val uses B, test uses C
  - full-name / address / id overlap between splits is zero by construction
    (PII-value-held-out and generator-held-out).

D and E exist for one specific experiment. Concept erasure fitted on family A
does not transfer to family B (measured: residual cross-covariance 1.5e-07 where
it was fit, 0.771 on validation; linear probe 0.932 vs a majority baseline of
0.847). The candidate explanation is that the erased subspace is family-specific.
D and E are additional TRAIN-side families — disjoint from B and C, so using them
leaks nothing — that let an eraser be fitted across several value distributions
at once to test whether family-diverse fitting restores transfer.

Disjointness is guaranteed by:
  - non-overlapping given-name and surname pools per family,
  - different phone prefixes and ID century codes per family,
  - non-overlapping street-name pools.

Everything is driven by a numpy RandomState for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass


# Per (family, language) given-name and surname pools. Small on purpose;
# expand later. The invariant that matters is: pools are disjoint across
# families, so no full name from A can appear in B or C.
_GIVEN = {
    ("A", "ko"): ["민준", "서연", "도윤", "지우", "하준", "수아"],
    ("B", "ko"): ["예준", "지호", "채원", "시우", "다은", "건우"],
    ("C", "ko"): ["윤서", "지안", "은우", "서준", "아린", "정우"],
    ("A", "en"): ["James", "Mary", "Robert", "Linda", "David", "Susan"],
    ("B", "en"): ["Oliver", "Emma", "Liam", "Ava", "Noah", "Sophia"],
    ("C", "en"): ["Ethan", "Chloe", "Mason", "Zoe", "Caleb", "Nora"],
    ("A", "ja"): ["ハルト", "ヒマリ", "ソウタ", "メイ", "ユウト", "ユイ"],
    ("B", "ja"): ["リク", "アオイ", "ハル", "サクラ", "カイト", "ミオ"],
    ("C", "ja"): ["レン", "ヒナ", "アサヒ", "ユナ", "タクミ", "リン"],
    ("D", "ko"): ["시윤", "하율", "주원", "소율", "이준", "예린"],
    ("E", "ko"): ["태윤", "나윤", "승우", "지윤", "현우", "가온"],
    ("D", "en"): ["Henry", "Grace", "Owen", "Lily", "Jack", "Ruby"],
    ("E", "en"): ["Felix", "Iris", "Victor", "Hazel", "Oscar", "Wren"],
    ("D", "ja"): ["ダイキ", "ノゾミ", "ケンタ", "アヤカ", "ショウ", "マナ"],
    ("E", "ja"): ["トオル", "サヤ", "ゲンキ", "ミサキ", "ジュン", "エリ"],
}
_SURNAME = {
    ("A", "ko"): ["김", "이", "박"],
    ("B", "ko"): ["최", "정", "강"],
    ("C", "ko"): ["조", "윤", "장"],
    ("A", "en"): ["Smith", "Johnson", "Williams"],
    ("B", "en"): ["Brown", "Jones", "Garcia"],
    ("C", "en"): ["Miller", "Davis", "Wilson"],
    ("A", "ja"): ["佐藤", "鈴木", "高橋"],
    ("B", "ja"): ["田中", "伊藤", "渡辺"],
    ("C", "ja"): ["山本", "中村", "小林"],
    ("D", "ko"): ["한", "오", "서"],
    ("E", "ko"): ["임", "신", "권"],
    ("D", "en"): ["Taylor", "Moore", "Clark"],
    ("E", "en"): ["Hughes", "Foster", "Bennett"],
    ("D", "ja"): ["加藤", "吉田", "山田"],
    ("E", "ja"): ["松本", "井上", "木村"],
}
_STREET = {
    "A": ["Maple", "Oak", "Cedar"],
    "B": ["Birch", "Pine", "Elm"],
    "C": ["Willow", "Aspen", "Spruce"],
    "D": ["Alder", "Hazel", "Juniper"],
    "E": ["Sycamore", "Poplar", "Chestnut"],
}
_ORG = {
    ("A", "ko"): ["가온병원", "한빛의료원"],
    ("B", "ko"): ["새봄병원", "온누리의료원"],
    ("C", "ko"): ["푸른병원", "미르의료원"],
    ("A", "en"): ["Gaon Hospital", "Hanbit Medical Center"],
    ("B", "en"): ["Saebom Hospital", "Onnuri Medical Center"],
    ("C", "en"): ["Pureun Hospital", "Mir Medical Center"],
    ("A", "ja"): ["ガオン病院", "ハンビット医療院"],
    ("B", "ja"): ["セボム病院", "オンヌリ医療院"],
    ("C", "ja"): ["プルン病院", "ミル医療院"],
    ("D", "ko"): ["하늘병원", "달빛의료원"],
    ("E", "ko"): ["빛솔병원", "너울의료원"],
    ("D", "en"): ["Haneul Hospital", "Dalbit Medical Center"],
    ("E", "en"): ["Bitsol Hospital", "Neoul Medical Center"],
    ("D", "ja"): ["ハヌル病院", "タルビット医療院"],
    ("E", "ja"): ["ピッソル病院", "ノウル医療院"],
}
# Disjoint phone prefixes and ID century digits keep numeric PII non-overlapping.
_PHONE_PREFIX = {"A": "010-2", "B": "010-5", "C": "010-8",
                 "D": "010-3", "E": "010-7"}
_ID_CENTURY = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}


@dataclass
class Family:
    name: str  # "A" | "B" | "C" | "D" | "E"

    def full_name(self, rng, lang: str) -> str:
        sur = rng.choice(_SURNAME[(self.name, lang)])
        giv = rng.choice(_GIVEN[(self.name, lang)])
        # Western order for en, family-name-first for ko/ja.
        return f"{giv} {sur}" if lang == "en" else f"{sur}{giv}"

    def provider_name(self, rng, lang: str) -> str:
        return self.full_name(rng, lang)

    def patient_id(self, rng, lang: str) -> str:
        return f"P{self.name}{int(rng.randint(100000, 999999))}"

    def national_id(self, rng, lang: str) -> str:
        yy = int(rng.randint(0, 100))
        mm = int(rng.randint(1, 13))
        dd = int(rng.randint(1, 29))
        tail = int(rng.randint(100000, 999999))
        return f"{yy:02d}{mm:02d}{dd:02d}-{_ID_CENTURY[self.name]}{tail}"

    def phone(self, rng, lang: str) -> str:
        return f"{_PHONE_PREFIX[self.name]}{int(rng.randint(100, 1000))}-{int(rng.randint(1000, 10000))}"

    def dob(self, rng, lang: str) -> str:
        y = int(rng.randint(1940, 2015))
        m = int(rng.randint(1, 13))
        d = int(rng.randint(1, 29))
        return f"{y:04d}-{m:02d}-{d:02d}"

    def email(self, rng, lang: str) -> str:
        user = f"user{self.name.lower()}{int(rng.randint(1000, 9999))}"
        return f"{user}@example-{self.name.lower()}.org"

    def address(self, rng, lang: str) -> str:
        num = int(rng.randint(1, 400))
        street = rng.choice(_STREET[self.name])
        if lang == "ko":
            return f"{street}로 {num}길 {int(rng.randint(1, 99))}"
        if lang == "ja":
            return f"{street}町{num}-{int(rng.randint(1, 99))}"
        return f"{num} {street} St."

    def organization(self, rng, lang: str) -> str:
        return rng.choice(_ORG[(self.name, lang)])


FAMILY_FOR_SPLIT = {"train": Family("A"), "val": Family("B"), "test": Family("C")}

# Train-side families usable for fitting without touching val (B) or test (C).
TRAIN_SIDE_FAMILIES = ["A", "D", "E"]


def all_families():
    """Every family whose pools are fully populated, for validation in tests."""
    return sorted({fam for fam, _lang in _GIVEN})
