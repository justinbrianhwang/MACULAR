"""Map FUNSD form annotations onto the privacy protocol, for a REAL-SCAN arm.

Everything in the erasure comparison so far runs on synthetic documents, because
no public scanned corpus carries PII annotations — a gap all four literature
reviews independently confirmed. FUNSD is real scanned paper with real handwriting
and scanner noise, and although it has no PII labels its annotation scheme
carries the same structural distinction the protocol needs:

    answer   the filled-in VALUE of a field — names, dates, numbers, signatures.
             This is what a redaction mechanism has to remove, and what an
             inversion attacker wants back.
    question the printed field LABEL, and
    header   the form's structure. These must survive redaction: they are the
             utility that makes the redacted document still worth processing.

So we map ``answer`` to the sensitive class and the rest to the utility classes.
This is a *stand-in* for PII, not PII: FUNSD answers are ordinary business-form
values, and the corpus is released for research use. The claim it supports is
narrow and worth having anyway — that the mechanism ranking measured on synthetic
pages also holds on real scanned pages with real OCR-grade noise.

What it cannot support: the held-out-PII-value-family protocol. FUNSD has one
document population, so the erasure transfer question (§4 of FINDINGS) stays a
synthetic-data result. Say so rather than blurring it.
"""

from __future__ import annotations

from ..schema import Document

# FUNSD block_type (already mapped from its labels) -> our two roles.
SENSITIVE_BLOCK = "value"          # FUNSD "answer"
_UTILITY_BLOCKS = {
    "label": "LAB_NAME",           # FUNSD "question": the printed field name
    "title": "MED_NAME",           # FUNSD "header": section structure
    "table_cell": "UNIT",          # FUNSD "other"
}
SENSITIVE_PII_TYPE = "PATIENT_NAME"   # must be a member of schema.PII_TYPES


def _check_vocabularies():
    """Fail loudly if the labels we assign are not in the canonical vocabularies.

    An earlier version used the type name "NAME", which is not in PII_TYPES, so
    every region silently mapped to the NON_PII index. The run completed and
    produced a full results table in which every probe scored 1.000 against a
    majority baseline of 1.000 — a table that looks like data and means nothing.
    """
    from ..schema import PII_TYPES
    from ..models.features import CLINICAL_TYPES

    if SENSITIVE_PII_TYPE not in PII_TYPES:
        raise ValueError(
            f"SENSITIVE_PII_TYPE {SENSITIVE_PII_TYPE!r} is not in PII_TYPES "
            f"{PII_TYPES}; it would silently map to NON_PII.")
    bad = [v for v in _UTILITY_BLOCKS.values() if v not in CLINICAL_TYPES]
    if bad:
        raise ValueError(f"utility classes not in CLINICAL_TYPES: {bad}")


def to_privacy_documents(docs: list[Document]) -> list[Document]:
    """Annotate in place-ish: return docs with pii_type / clinical_type set.

    ``answer`` regions become the sensitive class; every other region gets a
    utility class so the clinical head has something real to preserve. Empty
    regions are left alone — an empty string is neither sensitive nor useful.
    """
    _check_vocabularies()
    out = []
    for doc in docs:
        for c in doc.candidates:
            if not (c.text or "").strip():
                c.pii_type, c.clinical_type = None, "NONE"
                continue
            if c.block_type == SENSITIVE_BLOCK:
                c.pii_type, c.clinical_type = SENSITIVE_PII_TYPE, "NONE"
            else:
                c.pii_type = None
                c.clinical_type = _UTILITY_BLOCKS.get(c.block_type, "NONE")
        out.append(doc)
    return out


def stats(docs: list[Document]) -> dict:
    n_sens = sum(1 for d in docs for c in d.candidates if c.pii_type)
    n_util = sum(1 for d in docs for c in d.candidates
                 if c.clinical_type and c.clinical_type != "NONE")
    n_all = sum(len(d.candidates) for d in docs)
    return {"documents": len(docs), "regions": n_all,
            "sensitive_regions": n_sens, "utility_regions": n_util,
            "sensitive_rate": n_sens / max(1, n_all)}
