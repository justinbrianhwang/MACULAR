"""Bridge: real documents -> per-region features -> model-core tensors.

Turns our Document schema (synthetic jsonl or real FUNSD/XFUND) into the
(region_feats, boxes, pii_labels, clinical_labels, mask) batch the model core
consumes. Region features come from an OCR TOOL:

  source="gt"   use the annotated/ground-truth region text (clean; upper bound)
  source="ocr"  run an OCR engine on the region crop (realistic: OCR errors
                propagate into the features, exactly the failure mode the
                proposal cares about)

Feature vector per region = deterministic char-code histogram of the text
(captures script/charset — Hangul vs Latin differ) + block-type one-hot + box.
This is a CPU-runnable stand-in for a VLM's visual region embedding; the real
backbone (PaddleOCR-VL / Qwen-VL) drops into the same slot on GPU.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ..schema import PII_TYPES, NON_PII, BLOCK_TYPES, Document

# label vocabularies (index 0 is the "negative" class in both)
PII_TO_IDX = {NON_PII: 0, **{t: i + 1 for i, t in enumerate(PII_TYPES)}}
CLINICAL_TYPES = ["NONE", "LAB_NAME", "LAB_VALUE", "UNIT", "REF_RANGE",
                  "MED_NAME", "DOSE", "FREQUENCY", "DURATION"]
CLIN_TO_IDX = {t: i for i, t in enumerate(CLINICAL_TYPES)}

N_PII_CLASSES = len(PII_TO_IDX)          # 9
N_CLINICAL = len(CLINICAL_TYPES)         # 9
CHAR_DIM = 54
FEATURE_DIM = CHAR_DIM + len(BLOCK_TYPES) + 4   # char hist + block one-hot + box


def char_features(text: str, dim: int = CHAR_DIM) -> torch.Tensor:
    """Deterministic char-code histogram (normalized). ord() is stable across
    runs, unlike Python's salted hash()."""
    v = torch.zeros(dim)
    for ch in text:
        v[ord(ch) % dim] += 1.0
    n = max(1, len(text))
    return v / n


def corrupt_text(text: str, cer: float, rng) -> str:
    """Simulate OCR errors at a target character error rate.

    Each character is independently substituted / deleted / a neighbour inserted
    with probability ``cer``. Substitutions draw from a confusable pool of the
    same script, so the corruption looks like recognition error rather than
    random bytes. Used to measure how OCR quality propagates downstream
    (proposal 2: OCR errors cascade into PII detection and clinical fields).
    """
    if cer <= 0 or not text:
        return text
    out = []
    for ch in text:
        r = rng.random()
        if r >= cer:
            out.append(ch)
            continue
        op = rng.randint(0, 3)
        if op == 0:      # substitute with a nearby codepoint (same script-ish)
            out.append(chr(max(32, ord(ch) + int(rng.randint(-2, 3)) or 1)))
        elif op == 1:    # deletion
            continue
        else:            # insertion (duplicate-ish)
            out.append(ch)
            out.append(ch)
    return "".join(out)


def _region_feature(text: str, block_type: str, box) -> torch.Tensor:
    block = torch.zeros(len(BLOCK_TYPES))
    if block_type in BLOCK_TYPES:
        block[BLOCK_TYPES.index(block_type)] = 1.0
    box_t = torch.tensor(box.as_list(), dtype=torch.float32)
    return torch.cat([char_features(text), block, box_t])


def documents_to_batch(docs: list[Document], max_regions: int = 48,
                       source: str = "gt", engine=None, data_dir: str = "",
                       cer: float = 0.0, seed: int = 0, cache_engine: str = ""):
    """Build padded tensors from Documents.

    source="gt"     annotated text (clean upper bound)
    source="noisy"  annotated text corrupted at rate ``cer`` (controlled OCR-error
                    simulation; no engine needed — CPU reproducible)
    source="ocr"    run ``engine`` on the region crop (real OCR errors)
    source="cache"  read text produced earlier by ``cache_engine`` (see
                    baselines/ocr_cache.py — required for PaddleOCR, which
                    cannot share a process with torch)

    Returns (feats, boxes, pii_labels, clinical_labels, mask).
    """
    rng = np.random.RandomState(seed)
    cache = None
    if source == "cache":
        from ..baselines.ocr_cache import load_cache
        cache = load_cache(data_dir, cache_engine)
    B = len(docs)
    feats = torch.zeros(B, max_regions, FEATURE_DIM)
    boxes = torch.zeros(B, max_regions, 4)
    pii = torch.zeros(B, max_regions, dtype=torch.long)
    clin = torch.zeros(B, max_regions, dtype=torch.long)
    mask = torch.zeros(B, max_regions, dtype=torch.bool)

    img_cache = {}
    for b, doc in enumerate(docs):
        img = None
        if source == "ocr" and engine is not None and doc.image_path:
            p = os.path.join(data_dir, doc.image_path)
            if os.path.exists(p):
                from PIL import Image
                img = img_cache.get(p) or Image.open(p)
                img_cache[p] = img
        cached = cache.get(doc.doc_id) if cache is not None else None
        for n, c in enumerate(doc.candidates[:max_regions]):
            text = c.text
            if cached is not None and n < len(cached):
                text = cached[n]
            elif source == "ocr" and img is not None and engine is not None:
                from ..baselines.ocr import _crop
                crop = _crop(img, c.bbox, doc.width, doc.height)
                if crop is not None and engine.supports(doc.language):
                    text = engine.recognize(crop, doc.language) or c.text
            elif source == "noisy":
                text = corrupt_text(text, cer, rng)
            feats[b, n] = _region_feature(text, c.block_type, c.bbox)
            boxes[b, n] = torch.tensor(c.bbox.as_list())
            pii[b, n] = PII_TO_IDX.get(c.pii_type or NON_PII, 0)
            clin[b, n] = CLIN_TO_IDX.get(c.clinical_type or "NONE", 0)
            mask[b, n] = True
    return feats, boxes, pii, clin, mask


def config_for_features(d: int = 128, d_in: int = FEATURE_DIM, **kw):
    """A MacularConfig whose input dims match this featurizer."""
    from .core import MacularConfig
    return MacularConfig(d_in=d_in, d=d, n_pii_classes=N_PII_CLASSES,
                         n_clinical=N_CLINICAL, **kw)


def documents_to_vlm_batch(docs: list[Document], backbone, data_dir: str,
                           max_regions: int = 48, cache: dict | None = None,
                           normalize: bool = True, stats: dict | None = None):
    """Per-region features from a REAL VLM vision tower (proposal 11.2).

    One backbone forward per page; regions are ROI-pooled from that single grid.
    Labels/boxes/mask match ``documents_to_batch`` so the model core is unchanged
    — only the feature source differs (this is the A2 track; RQ6 compares it to
    the text-only A1 features).
    """
    import os
    from PIL import Image

    B = len(docs)
    boxes = torch.zeros(B, max_regions, 4)
    pii = torch.zeros(B, max_regions, dtype=torch.long)
    clin = torch.zeros(B, max_regions, dtype=torch.long)
    mask = torch.zeros(B, max_regions, dtype=torch.bool)
    feats = None

    for b, doc in enumerate(docs):
        cands = doc.candidates[:max_regions]
        bx = torch.tensor([c.bbox.as_list() for c in cands], dtype=torch.float32)
        key = doc.doc_id
        if cache is not None and key in cache:
            f = cache[key]
        else:
            img = Image.open(os.path.join(data_dir, doc.image_path)).convert("RGB")
            f = backbone.encode_page(img, bx)
            if cache is not None:
                cache[key] = f
        if feats is None:
            feats = torch.zeros(B, max_regions, f.shape[-1])
        n = len(cands)
        feats[b, :n] = f[:n]
        boxes[b, :n] = bx[:n]
        for i, c in enumerate(cands):
            pii[b, i] = PII_TO_IDX.get(c.pii_type or NON_PII, 0)
            clin[b, i] = CLIN_TO_IDX.get(c.clinical_type or "NONE", 0)
            mask[b, i] = True

    if normalize and feats is not None:
        # Standardize per feature dim. Raw vision-tower activations have a large
        # and uneven scale; feeding them unnormalized makes the linear projector
        # train far worse than the small hand-built text features, which would
        # look like "the backbone does not help" when it is really an
        # optimization artifact. Train statistics are reused for val via `stats`.
        valid = feats[mask]
        if stats is not None and "mean" in stats:
            mu, sd = stats["mean"], stats["std"]
        else:
            mu = valid.mean(0)
            sd = valid.std(0).clamp(min=1e-6)
            if stats is not None:
                stats["mean"], stats["std"] = mu, sd
        feats = (feats - mu) / sd
        feats = feats * mask.unsqueeze(-1)      # keep padding at zero
    return feats, boxes, pii, clin, mask
