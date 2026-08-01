"""Synthetic medical document generation.

Produces, per document:
  - an exact ground-truth ``Document`` (candidates, bboxes, PII spans, FHIR paths)
  - a rendered page image (PNG)

Ground truth is created *by construction*, so it never depends on OCR. The
pipeline downstream (shortcut audit, metrics) uses the labels, so it runs on
any machine regardless of installed fonts. Rendering degrades gracefully when a
CJK-capable font is unavailable (glyphs may not display, but geometry/labels
stay exact).

Supports the shortcut-resistance controls from proposal 14.4:
  - ``counterfactual_layout``: shuffle field order so PII is not at a fixed
    position, and randomly swap the label/value column.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..schema import BBox, Candidate, Document
from .pii_generators import Family, FAMILY_FOR_SPLIT
from .clinical_content import LOINC_PANEL, MED_LIST, sample_lab

PAGE_W, PAGE_H = 1000, 1400

# Document types the synthetic generator can produce.
DOC_TYPES = ("laboratory_report", "prescription")

# Fields rendered in the patient block: (label, pii_type, generator method).
_PATIENT_FIELDS = [
    ("Name", "PATIENT_NAME", "full_name"),
    ("Patient ID", "PATIENT_ID", "patient_id"),
    ("National ID", "NATIONAL_ID", "national_id"),
    ("DOB", "DOB", "dob"),
    ("Phone", "PHONE", "phone"),
    ("Address", "ADDRESS", "address"),
    ("Email", "EMAIL", "email"),
]

# Non-PII decoy values placed in the SAME value column as PII, only under the
# counterfactual layout. Their purpose (proposal 14.4) is to break the
# "block_type==value and/or fixed position => PII" shortcut: after mixing these
# in, geometry alone can no longer separate PII from non-PII.
_DECOY_FIELDS = [
    ("Department", ["Internal Medicine", "Cardiology", "Neurology", "Oncology"]),
    ("Room", ["302", "415", "108", "227"]),
    ("Visit Type", ["Outpatient", "Inpatient", "Emergency"]),
    ("Plan Code", ["PLN-A1", "PLN-B7", "PLN-C3", "PLN-D9"]),
    ("Ward", ["3F-West", "5F-East", "2F-North"]),
    ("Bed", ["B-12", "B-03", "B-21", "B-07"]),
    ("Clinic", ["C1", "C2", "C3", "C4"]),
    ("Attending Code", ["ATT-11", "ATT-42", "ATT-88", "ATT-56"]),
]

_TABLE_XS = [80, 380, 560, 720]


def _table_header(headers, ty):
    cands = []
    for h, x in zip(headers, _TABLE_XS):
        cands.append(Candidate(text=h, bbox=_norm(x, ty, x + 170, ty + 34),
                               block_type="table_header", pii_type=None))
    return cands


def _lab_table(rng, ty):
    """Realistic lab result rows with real LOINC codes/units/ranges."""
    cands = _table_header(["Test", "Result", "Unit", "Reference"], ty)
    ty += 44
    n = int(rng.randint(3, min(6, len(LOINC_PANEL)) + 1))
    idx = rng.choice(len(LOINC_PANEL), size=n, replace=False)
    for i in idx:
        item = LOINC_PANEL[i]
        name, loinc, unit, low, high = item[0], item[1], item[2], item[3], item[4]
        value = sample_lab(rng, item)
        ref = f"{low:g}-{high:g}"
        cells = [
            (name, "LAB_NAME", f"Observation.code[loinc={loinc}]"),
            (value, "LAB_VALUE", "Observation.valueQuantity.value"),
            (unit, "UNIT", "Observation.valueQuantity.unit"),
            (ref, "REF_RANGE", "Observation.referenceRange"),
        ]
        for (txt, ctype, fp), x in zip(cells, _TABLE_XS):
            cands.append(Candidate(text=str(txt),
                                   bbox=_norm(x, ty, x + 170, ty + 34),
                                   block_type="table_cell", pii_type=None,
                                   clinical_type=ctype, fhir_path=fp))
        ty += 40
    return cands, ty


def _prescription_table(rng, ty):
    """Realistic medication rows with RxNorm-style drugs/doses/frequencies."""
    cands = _table_header(["Medication", "Dose", "Frequency", "Duration"], ty)
    ty += 44
    n = int(rng.randint(2, min(5, len(MED_LIST)) + 1))
    idx = rng.choice(len(MED_LIST), size=n, replace=False)
    for i in idx:
        drug, cui, forms, doses, freqs, durs = MED_LIST[i]
        cells = [
            (f"{drug} ({rng.choice(forms)})", "MED_NAME",
             f"MedicationRequest.medication[rxnorm={cui}]"),
            (str(rng.choice(doses)), "DOSE",
             "MedicationRequest.dosageInstruction.doseAndRate"),
            (str(rng.choice(freqs)), "FREQUENCY",
             "MedicationRequest.dosageInstruction.timing"),
            (str(rng.choice(durs)), "DURATION",
             "MedicationRequest.dispenseRequest.expectedSupplyDuration"),
        ]
        for (txt, ctype, fp), x in zip(cells, _TABLE_XS):
            cands.append(Candidate(text=str(txt),
                                   bbox=_norm(x, ty, x + 190, ty + 34),
                                   block_type="table_cell", pii_type=None,
                                   clinical_type=ctype, fhir_path=fp))
        ty += 40
    return cands, ty


def _clinical_value_items(rng, doc_type) -> list[tuple[str, str, str]]:
    """Flat list of (text, clinical_type, fhir_path) clinical VALUES — no
    positions, no headers. Used to fill the counterfactual pool with non-PII
    values that are geometrically interchangeable with PII."""
    items: list[tuple[str, str, str]] = []
    if doc_type == "prescription":
        idx = rng.choice(len(MED_LIST), size=int(rng.randint(2, 5)), replace=False)
        for i in idx:
            drug, cui, forms, doses, freqs, durs = MED_LIST[i]
            items += [
                (f"{drug} ({rng.choice(forms)})", "MED_NAME",
                 f"MedicationRequest.medication[rxnorm={cui}]"),
                (str(rng.choice(doses)), "DOSE",
                 "MedicationRequest.dosageInstruction.doseAndRate"),
                (str(rng.choice(freqs)), "FREQUENCY",
                 "MedicationRequest.dosageInstruction.timing"),
            ]
    else:
        idx = rng.choice(len(LOINC_PANEL), size=int(rng.randint(3, 6)), replace=False)
        for i in idx:
            item = LOINC_PANEL[i]
            items += [
                (item[0], "LAB_NAME", f"Observation.code[loinc={item[1]}]"),
                (sample_lab(rng, item), "LAB_VALUE", "Observation.valueQuantity.value"),
                (item[2], "UNIT", "Observation.valueQuantity.unit"),
            ]
    return items


def _norm(x0, y0, x1, y1) -> BBox:
    return BBox(x0 / PAGE_W, y0 / PAGE_H, x1 / PAGE_W, y1 / PAGE_H)


def _counterfactual_body(rng, family, language, doc_type) -> list[Candidate]:
    """Shortcut-hardened layout (proposal 14.4).

    PII fields, non-PII decoys, and clinical values are merged into ONE pool,
    shuffled across the whole page, each value given a RANDOM block_type. This
    delocalizes PII and decorrelates it from both position and block_type, so a
    coordinate-only classifier drops toward the base rate (no positional/
    structural shortcut left to exploit).
    """
    pool: list[dict] = []
    for label, pii_type, method in _PATIENT_FIELDS:
        pool.append({"pii": pii_type, "text": str(getattr(family, method)(rng, language)),
                     "ctype": None, "fhir": None, "label": label})
    pool.append({"pii": "PROVIDER_NAME", "text": family.provider_name(rng, language),
                 "ctype": None, "fhir": None, "label": "Provider"})
    for label, vals in _DECOY_FIELDS:
        pool.append({"pii": None, "text": str(rng.choice(vals)),
                     "ctype": None, "fhir": None, "label": label})
    for text, ctype, fhir in _clinical_value_items(rng, doc_type):
        pool.append({"pii": None, "text": str(text), "ctype": ctype,
                     "fhir": fhir, "label": ctype or "Field"})

    rng.shuffle(pool)
    cands: list[Candidate] = []
    y, row_h = 180, 42
    for it in pool:
        if y > PAGE_H - 70:
            break
        swap = bool(rng.randint(0, 2))
        lx, vx = (80, 380) if not swap else (520, 80)
        cands.append(Candidate(text=it["label"],
                               bbox=_norm(lx, y, lx + 220, y + 34),
                               block_type="label", pii_type=None))
        vblock = str(rng.choice(["value", "table_cell"]))
        cands.append(Candidate(text=it["text"],
                               bbox=_norm(vx, y, vx + 300, y + 34),
                               block_type=vblock, pii_type=it["pii"],
                               clinical_type=it["ctype"], fhir_path=it["fhir"],
                               redaction_action="pseudonymize" if it["pii"] else None))
        y += row_h
    return cands


def build_document(
    rng: np.random.RandomState,
    family: Family,
    language: str,
    doc_id: str,
    split: str,
    doc_type: str = "laboratory_report",
    counterfactual_layout: bool = False,
) -> Document:
    cands: list[Candidate] = []
    title_text = {"laboratory_report": "Laboratory Report",
                  "prescription": "Prescription"}.get(doc_type, "Medical Report")

    # Title
    cands.append(
        Candidate(
            text=f"{family.organization(rng, language)} - {title_text}",
            bbox=_norm(80, 60, 920, 110),
            block_type="title",
            pii_type=None,
            clinical_type=None,
        )
    )
    # Organization also appears as a PII-relevant identifier in header corner.
    cands.append(
        Candidate(
            text=family.organization(rng, language),
            bbox=_norm(700, 120, 920, 155),
            block_type="value",
            pii_type="ADDRESS" if False else None,  # org name kept non-PII here
            clinical_type=None,
        )
    )

    if counterfactual_layout:
        # Shortcut-hardened layout: PII delocalized and block_type randomized.
        cands.extend(_counterfactual_body(rng, family, language, doc_type))
        return Document(
            doc_id=doc_id, language=language, doc_type=doc_type,
            width=PAGE_W, height=PAGE_H, candidates=cands, split=split,
            generator_family=family.name,
        )

    # --- realistic (default) layout: clean patient block + table + signature.
    # PII sits in the value column, giving a deliberate positional shortcut that
    # the coordinate-only audit is meant to DETECT (counterfactual removes it).
    y = 190
    row_h = 46
    for label, pii_type, method in _PATIENT_FIELDS:
        value = str(getattr(family, method)(rng, language))
        cands.append(Candidate(text=label, bbox=_norm(80, y, 320, y + 36),
                               block_type="label", pii_type=None))
        cands.append(Candidate(text=value, bbox=_norm(360, y, 660, y + 36),
                               block_type="value", pii_type=pii_type,
                               redaction_action="pseudonymize"))
        y += row_h

    # Clinical table (content depends on document type) ---------------------
    ty = y + 40
    if doc_type == "prescription":
        table_cands, ty = _prescription_table(rng, ty)
    else:
        table_cands, ty = _lab_table(rng, ty)
    cands.extend(table_cands)

    # Provider signature ----------------------------------------------------
    cands.append(Candidate(text="Reviewed by", bbox=_norm(80, ty + 60, 320, ty + 96),
                           block_type="label", pii_type=None))
    cands.append(Candidate(text=family.provider_name(rng, language),
                           bbox=_norm(360, ty + 60, 660, ty + 96),
                           block_type="value", pii_type="PROVIDER_NAME",
                           redaction_action="pseudonymize"))

    return Document(
        doc_id=doc_id,
        language=language,
        doc_type=doc_type,
        width=PAGE_W,
        height=PAGE_H,
        candidates=cands,
        split=split,
        generator_family=family.name,
    )


# --- rendering -------------------------------------------------------------
#
# A SINGLE font for all languages is fragile: a font that covers Latin+Japanese
# (e.g. MS Gothic) has no Hangul, so Korean renders as tofu boxes and any OCR of
# it is garbage. We therefore pick a font PER LANGUAGE and verify at runtime
# that it actually renders that script's glyphs (not .notdef boxes).

# Pan-CJK fonts (cover Latin + Hangul + Japanese) are tried first for CJK langs.
_PAN_CJK = [
    "C:/Windows/Fonts/malgun.ttf",                                   # Windows
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",        # Linux
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",                    # macOS (ko)
]
_LANG_FONTS = {
    "ko": _PAN_CJK + [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ],
    "ja": [
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
    ] + _PAN_CJK + [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ],
    "en": ["C:/Windows/Fonts/arial.ttf"] + _PAN_CJK,
    "es": ["C:/Windows/Fonts/arial.ttf"] + _PAN_CJK,
}
# Distinct sample glyphs per script; if a font renders them all identically it
# lacks the script (they collapse to the same tofu box).
_COVERAGE_SAMPLE = {
    "ko": ["가", "한", "글"],
    "ja": ["佐", "鈴", "渡"],
    "en": ["A", "B", "C"],
    "es": ["A", "B", "C"],
}

_font_cache: dict = {}
FONT_WARNINGS: set = set()   # languages for which no covering font was found


def _font_covers(font, sample_chars) -> bool:
    import numpy as np
    bitmaps = set()
    for ch in sample_chars:
        im = Image.new("L", (40, 40), 0)
        ImageDraw.Draw(im).text((2, 2), ch, fill=255, font=font)
        bitmaps.add(np.asarray(im).tobytes())
    return len(bitmaps) > 1


def _load_font(size: int, language: str = "en"):
    key = (language, size)
    if key in _font_cache:
        return _font_cache[key]
    sample = _COVERAGE_SAMPLE.get(language, _COVERAGE_SAMPLE["en"])
    for path in _LANG_FONTS.get(language, _PAN_CJK):
        if not os.path.exists(path):
            continue
        try:
            f = ImageFont.truetype(path, size)
        except Exception:
            continue
        if _font_covers(f, sample):
            _font_cache[key] = f
            return f
    # No covering font: glyphs for this language will not render. Record it so
    # the OCR baseline result flags these languages instead of scoring garbage.
    FONT_WARNINGS.add(language)
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def render_document(doc: Document, add_noise: bool = True,
                    rng: Optional[np.random.RandomState] = None) -> Image.Image:
    img = Image.new("RGB", (doc.width, doc.height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(22, doc.language)
    title_font = _load_font(28, doc.language)
    for c in doc.candidates:
        x0 = int(c.bbox.x0 * doc.width)
        y0 = int(c.bbox.y0 * doc.height)
        f = title_font if c.block_type == "title" else font
        try:
            draw.text((x0, y0), c.text, fill="black", font=f)
        except Exception:
            draw.text((x0, y0), "?", fill="black", font=f)
    if add_noise and rng is not None:
        img = _apply_noise(img, rng)
    return img


def _apply_noise(img: Image.Image, rng: np.random.RandomState) -> Image.Image:
    # Light, cheap scan simulation: slight rotation + gaussian noise.
    angle = float(rng.uniform(-1.5, 1.5))
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor="white")
    arr = np.asarray(img).astype(np.int16)
    arr += rng.normal(0, 6, arr.shape).astype(np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# --- dataset driver --------------------------------------------------------

def generate_dataset(
    out_dir: str,
    n_per_split: int,
    languages: list[str],
    seed: int = 0,
    counterfactual_layout: bool = False,
    render_images: bool = True,
    doc_types: Optional[list[str]] = None,
    families: Optional[dict] = None,
) -> dict:
    """Generate train/val/test with disjoint PII families and write to disk.

    Returns a manifest dict. Images go to ``<out_dir>/images``; labels to
    ``<out_dir>/<split>.jsonl``. ``doc_types`` mixes document types evenly
    (default: all of ``DOC_TYPES``).

    ``families`` overrides the split -> family mapping, e.g.
    ``{"train": Family("D")}`` to emit an extra train-side value distribution.
    Only ever map a split to a family that is disjoint from the val/test
    families, or the held-out-PII protocol is broken.
    """
    from ..schema import write_jsonl

    doc_types = list(doc_types) if doc_types else list(DOC_TYPES)
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    if render_images:
        os.makedirs(img_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    manifest = {"out_dir": out_dir, "splits": {}, "languages": languages,
                "doc_types": doc_types,
                "counterfactual_layout": counterfactual_layout}

    split_families = dict(families) if families else dict(FAMILY_FOR_SPLIT)
    for split, family in split_families.items():
        docs = []
        for i in range(n_per_split):
            lang = languages[int(rng.randint(0, len(languages)))]
            dtype = doc_types[int(rng.randint(0, len(doc_types)))]
            doc_id = f"{split}-{family.name}-{i:05d}"
            doc = build_document(
                rng, family, lang, doc_id, split,
                doc_type=dtype,
                counterfactual_layout=counterfactual_layout,
            )
            if render_images:
                img = render_document(doc, add_noise=True, rng=rng)
                img_path = os.path.join(img_dir, f"{doc_id}.png")
                img.save(img_path)
                doc.image_path = os.path.relpath(img_path, out_dir)
            docs.append(doc)
        path = os.path.join(out_dir, f"{split}.jsonl")
        write_jsonl(docs, path)
        manifest["splits"][split] = {"n": len(docs), "family": family.name,
                                     "labels": os.path.basename(path)}
    # Languages whose glyphs could not be rendered (no covering font installed).
    # OCR/metrics on these languages would be meaningless — surface it loudly.
    if FONT_WARNINGS:
        manifest["font_warnings"] = sorted(FONT_WARNINGS)
        manifest["font_warning_note"] = (
            "No glyph-covering font found for these languages; their rendered "
            "text is unreadable (tofu). Install a CJK font (e.g. Noto Sans CJK "
            "or Windows malgun.ttf) before trusting OCR results for them."
        )
    return manifest
