"""Realistic clinical content for synthetic documents (realism upgrade #1).

Uses real, public medical terminology and plausible value distributions — no
patient data. Lab tests carry real LOINC codes with realistic reference ranges
and sampling distributions; medications carry real RxNorm-style names with
typical forms/doses/frequencies. This makes MACULAR-MedDoc credible without any
PHI.

Sources for the vocabularies (public):
  - LOINC (loinc.org) for laboratory observation codes.
  - RxNorm (nlm.nih.gov/research/umls/rxnorm) for medication naming.
Only code/name identifiers and typical ranges are used; no proprietary tables
are redistributed.
"""

from __future__ import annotations

# name, LOINC code, unit, ref_low, ref_high, sampling mean, sampling sd
LOINC_PANEL = [
    ("Glucose",      "2345-7", "mg/dL", 70, 110, 98, 22),
    ("Hemoglobin",   "718-7",  "g/dL",  13.0, 17.0, 14.2, 1.6),
    ("Cholesterol",  "2093-3", "mg/dL", 0, 200, 190, 38),
    ("Creatinine",   "2160-0", "mg/dL", 0.6, 1.2, 0.95, 0.25),
    ("Potassium",    "2823-3", "mmol/L", 3.5, 5.1, 4.2, 0.4),
    ("Sodium",       "2951-2", "mmol/L", 135, 145, 140, 3),
    ("ALT",          "1742-6", "U/L",   7, 56, 30, 16),
    ("Platelets",    "777-3",  "10^3/uL", 150, 400, 265, 60),
    ("TSH",          "3016-3", "mIU/L", 0.4, 4.0, 2.1, 1.1),
    ("HbA1c",        "4548-4", "%",     4.0, 5.6, 5.6, 0.9),
]

# drug, RxNorm CUI, [forms], [doses], [frequencies], [durations]
MED_LIST = [
    ("Metformin",     "6809",  ["tablet"], ["500 mg", "850 mg", "1000 mg"],
     ["once daily", "twice daily"], ["30 days", "90 days"]),
    ("Atorvastatin",  "83367", ["tablet"], ["10 mg", "20 mg", "40 mg"],
     ["once daily", "at bedtime"], ["30 days", "90 days"]),
    ("Amlodipine",    "17767", ["tablet"], ["5 mg", "10 mg"],
     ["once daily"], ["30 days", "90 days"]),
    ("Lisinopril",    "29046", ["tablet"], ["5 mg", "10 mg", "20 mg"],
     ["once daily"], ["30 days"]),
    ("Levothyroxine", "10582", ["tablet"], ["50 mcg", "75 mcg", "100 mcg"],
     ["once daily", "in the morning"], ["90 days"]),
    ("Amoxicillin",   "723",   ["capsule"], ["250 mg", "500 mg"],
     ["three times daily"], ["7 days", "10 days"]),
    ("Omeprazole",    "7646",  ["capsule"], ["20 mg", "40 mg"],
     ["once daily", "before breakfast"], ["14 days", "30 days"]),
    ("Ibuprofen",     "5640",  ["tablet"], ["200 mg", "400 mg", "600 mg"],
     ["as needed", "three times daily"], ["5 days", "10 days"]),
]


def sample_lab(rng, panel_item) -> str:
    """Draw a realistic result value for a LOINC panel item."""
    _, _, _unit, low, high, mean, sd = panel_item
    val = float(rng.normal(mean, sd))
    # keep physiologically sane and occasionally out of range
    lo_clip = max(0.0, low - 3 * (high - low))
    hi_clip = high + 3 * (high - low)
    val = min(max(val, lo_clip), hi_clip)
    decimals = 0 if high >= 100 else 1
    return f"{round(val, decimals):g}"
