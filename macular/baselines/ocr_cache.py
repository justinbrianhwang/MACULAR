"""Decouple OCR from training by caching recognized text to disk.

Why this exists (proposal 8, "권장 환경 분리"): PaddlePaddle and PyTorch cannot
share a process on Windows — whichever loads first breaks the other's DLLs
(observed: `import paddleocr` pulls in torch, and torch's shm.dll then fails to
load once paddle is resident). The proposal already prescribes running OCR as a
separate service from the model core; this module is that boundary.

Flow:
  1. In an OCR-only environment:  build_cache(...) -> ocr_cache_<engine>.json
  2. In the torch environment:    documents_to_batch(source="cache", ...)

The cache also makes experiments cheap: OCR runs once per engine, then any
number of training configurations reuse it.
"""

from __future__ import annotations

import json
import os

CACHE_VERSION = 1


def cache_path(data_dir: str, engine: str) -> str:
    return os.path.join(data_dir, f"ocr_cache_{engine}.json")


def build_cache(docs, data_dir: str, engine, max_docs: int | None = None,
                progress_every: int = 25) -> dict:
    """Run ``engine`` over every region of every doc and write a text cache.

    Returns a manifest. The cache maps doc_id -> list of recognized strings,
    aligned with ``doc.candidates`` order.
    """
    from PIL import Image
    from .ocr import _crop

    subset = docs[:max_docs] if max_docs else docs
    out: dict[str, list[str]] = {}
    n_regions = 0
    skipped_langs: dict[str, str] = {}

    for i, doc in enumerate(subset):
        if not engine.supports(doc.language):
            skipped_langs[doc.language] = f"engine cannot handle '{doc.language}'"
            continue
        if not doc.image_path:
            continue
        img_path = os.path.join(data_dir, doc.image_path)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)
        texts: list[str] = []
        for c in doc.candidates:
            crop = _crop(img, c.bbox, doc.width, doc.height)
            if crop is None or not c.text.strip():
                texts.append(c.text)
                continue
            try:
                texts.append(engine.recognize(crop, doc.language) or "")
            except Exception as e:
                # Fail loudly: a cache full of "" looks like a successful run but
                # silently destroys every downstream experiment.
                raise RuntimeError(
                    f"OCR engine '{engine.name}' failed on {doc.doc_id} "
                    f"region {len(texts)}: {type(e).__name__}: {e}") from e
            n_regions += 1
        out[doc.doc_id] = texts
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[ocr_cache] {i + 1}/{len(subset)} docs", flush=True)

    # Sanity gate: an engine that "succeeds" but returns nothing everywhere is
    # broken (e.g. misconfigured GPU libs). Refuse to write such a cache.
    values = [s for texts in out.values() for s in texts]
    non_empty = sum(1 for s in values if s.strip())
    if values and non_empty / len(values) < 0.05:
        raise RuntimeError(
            f"OCR engine '{engine.name}' produced empty text for "
            f"{100 * (1 - non_empty / len(values)):.1f}% of regions — refusing "
            f"to write a useless cache. Check the engine installation.")

    path = cache_path(data_dir, engine.name)
    payload = {"version": CACHE_VERSION, "engine": engine.name,
               "n_documents": len(out), "n_regions": n_regions,
               "non_empty_regions": non_empty,
               "skipped_languages": skipped_langs, "texts": out}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return {"cache_path": path, "engine": engine.name,
            "n_documents": len(out), "n_regions": n_regions,
            "skipped_languages": skipped_langs}


def load_cache(data_dir: str, engine: str) -> dict[str, list[str]]:
    path = cache_path(data_dir, engine)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no OCR cache for engine '{engine}' at {path}. Build it first: "
            f"macular run ocr_cache --config <cfg>  (with ocr_engine: {engine})")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("texts", {})
