"""Shared converter for FUNSD-style annotations -> MACULAR Documents.

FUNSD and XFUND both annotate a page as a list of "form" items, each with a
pixel bbox, transcribed text, and a label in {header, question, answer, other}.
We map those to our schema (bbox normalized by page size, label -> block_type,
transcription -> candidate text). There are no PII labels in these datasets, so
``pii_type`` is always None (the OCR baseline and layout stats still apply).
"""

from __future__ import annotations

from ..schema import BBox, Candidate, Document

# FUNSD/XFUND label -> MACULAR block_type
_LABEL_TO_BLOCK = {
    "header": "title",
    "question": "label",
    "answer": "value",
    "other": "table_cell",
}


def form_items_to_document(
    items: list[dict],
    page_w: int,
    page_h: int,
    doc_id: str,
    language: str,
    split: str,
    image_path: str | None = None,
) -> Document:
    """Convert one page's FUNSD-style ``form`` items into a Document.

    Each item must have ``box`` = [x0, y0, x1, y1] in pixels, ``text``, and
    ``label``. Empty-text items are skipped.
    """
    cands: list[Candidate] = []
    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        box = it.get("box") or it.get("bbox")
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        # guard against inverted or out-of-range boxes
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        bbox = BBox(
            max(0.0, x0 / page_w), max(0.0, y0 / page_h),
            min(1.0, x1 / page_w), min(1.0, y1 / page_h),
        )
        block = _LABEL_TO_BLOCK.get(it.get("label", "other"), "table_cell")
        cands.append(Candidate(text=text, bbox=bbox, block_type=block,
                               pii_type=None))
    return Document(
        doc_id=doc_id, language=language, doc_type="form",
        width=page_w, height=page_h, candidates=cands, split=split,
        generator_family="real", image_path=image_path,
    )
