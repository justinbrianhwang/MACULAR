"""Shared data contract for MACULAR.

Every module (data generation, baselines, metrics, and eventually the model
core) reads and writes these types. Ground-truth labels are produced *by
construction* during synthetic generation, so bboxes and PII spans are exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# --- PII / clinical / block vocabularies -----------------------------------

PII_TYPES = (
    "PATIENT_NAME",
    "PATIENT_ID",
    "DOB",
    "PHONE",
    "ADDRESS",
    "EMAIL",
    "PROVIDER_NAME",
    "NATIONAL_ID",
)

NON_PII = "NON_PII"

BLOCK_TYPES = ("title", "label", "value", "table_header", "table_cell", "stamp")

REDACTION_ACTIONS = ("remove", "mask", "pseudonymize", "retain")


@dataclass
class BBox:
    """Normalized (0..1) page coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    def iou(self, other: "BBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Candidate:
    """One region proposal / token span on a page."""

    text: str
    bbox: BBox
    block_type: str
    pii_type: Optional[str] = None            # None => not PII
    clinical_type: Optional[str] = None       # e.g. LAB_NAME, LAB_VALUE, UNIT
    fhir_path: Optional[str] = None           # e.g. Observation.valueQuantity.value
    redaction_action: Optional[str] = None

    @property
    def is_pii(self) -> bool:
        return self.pii_type is not None


@dataclass
class Document:
    doc_id: str
    language: str                # ko | en | ja | es
    doc_type: str                # laboratory_report | prescription | ...
    width: int
    height: int
    candidates: list[Candidate] = field(default_factory=list)
    split: Optional[str] = None          # train | val | test
    generator_family: Optional[str] = None
    image_path: Optional[str] = None

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidates"] = [
            {**asdict(c), "bbox": c.bbox.as_list()} for c in self.candidates
        ]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Document":
        cands = []
        for c in d.get("candidates", []):
            b = c["bbox"]
            cands.append(
                Candidate(
                    text=c["text"],
                    bbox=BBox(*b),
                    block_type=c["block_type"],
                    pii_type=c.get("pii_type"),
                    clinical_type=c.get("clinical_type"),
                    fhir_path=c.get("fhir_path"),
                    redaction_action=c.get("redaction_action"),
                )
            )
        return Document(
            doc_id=d["doc_id"],
            language=d["language"],
            doc_type=d["doc_type"],
            width=d["width"],
            height=d["height"],
            candidates=cands,
            split=d.get("split"),
            generator_family=d.get("generator_family"),
            image_path=d.get("image_path"),
        )


def write_jsonl(docs: list[Document], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(Document.from_dict(json.loads(line)))
    return docs
