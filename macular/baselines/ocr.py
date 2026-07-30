"""Region-level, per-language OCR baseline with pluggable engines.

Tesseract is a weak floor for CJK — the proposal (25.x tool list) targets
PaddleOCR-VL for exactly this reason. This module lets you swap the recognition
engine and compare them on the SAME synthetic set, per language:

    tesseract   weak floor, CPU, needs language packs
    easyocr     much stronger CJK, pip-only, uses GPU if present
    (paddleocr-vl: the real target, GPU + heavier setup — future engine)

Measurement design (unchanged across engines):
  - OCR each candidate REGION crop, not the whole page (removes reading-order /
    whitespace confounds).
  - Report CER/WER PER LANGUAGE, never blended (proposal 4.4 / 18.1).
  - WER is null for scripts without word spaces (ko/ja/zh) — whitespace WER is
    meaningless there.
  - Languages the engine can't handle are SKIPPED and reported, not scored as
    garbage.
"""

from __future__ import annotations

import os

from ..schema import Document
from ..evaluation.metrics import cer, wer

# MACULAR language code -> Tesseract language pack code.
LANG_MAP = {"en": "eng", "ko": "kor", "ja": "jpn", "es": "spa"}

# Scripts without whitespace word delimiters: whitespace WER is meaningless
# ("김민준" vs "김 민 준" inflates WER past 1). Report CER only for these.
NO_WORD_SEGMENTATION = {"ko", "ja", "zh"}

CROP_PAD_FRAC = 0.004
CROP_UPSCALE = 3


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _normalize(s: str) -> str:
    return " ".join(s.split())


def _crop(img, bbox, page_w, page_h):
    from PIL import Image
    pad_x = CROP_PAD_FRAC * page_w
    pad_y = CROP_PAD_FRAC * page_h
    x0 = max(0, int(bbox.x0 * page_w - pad_x))
    y0 = max(0, int(bbox.y0 * page_h - pad_y))
    x1 = min(page_w, int(bbox.x1 * page_w + pad_x))
    y1 = min(page_h, int(bbox.y1 * page_h + pad_y))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img.crop((x0, y0, x1, y1)).convert("L")
    if CROP_UPSCALE > 1:
        crop = crop.resize(
            (crop.width * CROP_UPSCALE, crop.height * CROP_UPSCALE),
            resample=Image.LANCZOS,
        )
    return crop


# --- engines ---------------------------------------------------------------

class TesseractEngine:
    name = "tesseract"

    def available(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def _installed(self) -> set:
        try:
            import pytesseract
            return set(pytesseract.get_languages(config=""))
        except Exception:
            return set()

    def supports(self, mac_lang: str) -> bool:
        return LANG_MAP.get(mac_lang) in self._installed()

    def recognize(self, crop, mac_lang: str) -> str:
        import pytesseract
        return pytesseract.image_to_string(
            crop, lang=LANG_MAP[mac_lang], config="--psm 7"
        )


class EasyOCREngine:
    name = "easyocr"
    # EasyOCR requires English to be combinable with most non-Latin scripts.
    _READER_LANGS = {"en": ["en"], "ko": ["ko", "en"], "ja": ["ja", "en"],
                     "es": ["es"]}

    def __init__(self):
        self._readers: dict = {}

    def available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except Exception:
            return False

    def supports(self, mac_lang: str) -> bool:
        return mac_lang in self._READER_LANGS

    def _reader(self, mac_lang: str):
        if mac_lang not in self._readers:
            import easyocr
            self._readers[mac_lang] = easyocr.Reader(
                self._READER_LANGS[mac_lang], gpu=_cuda()
            )
        return self._readers[mac_lang]

    def recognize(self, crop, mac_lang: str) -> str:
        import numpy as np
        parts = self._reader(mac_lang).readtext(np.array(crop), detail=0)
        sep = "" if mac_lang in NO_WORD_SEGMENTATION else " "
        return sep.join(parts)


def _extract_paddle_text(res) -> str:
    """PP-OCR return shapes vary across versions. Pull out recognized strings
    defensively: find (text, confidence) tuples and join their text."""
    out = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            if (len(x) == 2 and isinstance(x[0], str)
                    and isinstance(x[1], (int, float))):
                out.append(x[0])
                return
            for e in x:
                walk(e)
        elif isinstance(x, dict):
            # newer predict() API: {'rec_texts': [...]} or similar
            for k in ("rec_texts", "rec_text", "text"):
                if k in x:
                    v = x[k]
                    out.extend(v if isinstance(v, list) else [v])
                    return
            for v in x.values():
                walk(v)

    walk(res)
    return " ".join(t for t in out if t)


def _paddle_gpu() -> bool:
    try:
        import paddle
        return bool(paddle.device.is_compiled_with_cuda()
                    and paddle.device.cuda.device_count() > 0)
    except Exception:
        return False


class PaddleOCREngine:
    """PP-OCR (classic PaddleOCR). Strong multilingual incl. CJK, GPU-capable.
    Install: pip install -e ".[paddleocr]"  plus paddlepaddle (see README).

    Robust to PaddleOCR 2.x vs 3.x: the constructor and inference call signatures
    changed across versions, so we try several and keep the first that works.
    GPU is detected via paddle itself (not torch)."""
    name = "paddleocr"
    _LANG = {"en": "en", "ko": "korean", "ja": "japan", "es": "es"}

    def __init__(self):
        self._ocr: dict = {}

    def available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
            return True
        except Exception:
            return False

    def supports(self, mac_lang: str) -> bool:
        return mac_lang in self._LANG

    def _engine(self, mac_lang: str):
        if mac_lang not in self._ocr:
            from paddleocr import PaddleOCR
            lang = self._LANG[mac_lang]
            gpu = _paddle_gpu()
            # Try newest->oldest kwarg shapes; keep the first that constructs.
            attempts = [
                dict(lang=lang, use_angle_cls=True, use_gpu=gpu, show_log=False),
                dict(lang=lang, use_angle_cls=True),
                dict(lang=lang, device="gpu" if gpu else "cpu"),
                dict(lang=lang),
            ]
            last_err = None
            built = None
            for kw in attempts:
                try:
                    built = PaddleOCR(**kw)
                    break
                except Exception as e:  # TypeError on removed kwargs, etc.
                    last_err = e
            if built is None:
                raise RuntimeError(
                    f"Could not construct PaddleOCR for lang={lang}; installed "
                    f"version may differ. Last error: {last_err}")
            self._ocr[mac_lang] = built
        return self._ocr[mac_lang]

    def recognize(self, crop, mac_lang: str) -> str:
        import numpy as np
        arr = np.array(crop.convert("RGB"))
        eng = self._engine(mac_lang)
        # recognition-only where supported (we already have the region box);
        # fall back through signature changes across versions.
        last_err = None
        for call in (
            lambda: eng.ocr(arr, det=False, cls=False),
            lambda: eng.ocr(arr, cls=False),
            lambda: eng.predict(arr),
            lambda: eng.ocr(arr),
        ):
            try:
                return _extract_paddle_text(call())
            except TypeError as e:
                last_err = e          # signature mismatch -> try the next shape
                continue
            except Exception as e:
                # A real runtime failure (missing cuDNN, OOM, ...). Do NOT keep
                # trying other signatures and do NOT return "" — a silently
                # empty result would be cached as if OCR had succeeded.
                raise RuntimeError(
                    f"PaddleOCR recognition failed: {type(e).__name__}: {e}"
                ) from e
        raise RuntimeError(
            f"No compatible PaddleOCR call signature; last error: {last_err}")


class PaddleOCRVLEngine:
    """PaddleOCR-VL-1.6 (the proposal's target OCR). EXPERIMENTAL: heavy GPU
    setup (paddlepaddle 3.2.1+ / transformers custom code). Skips gracefully if
    it can't load. Set env MACULAR_PADDLEVL_DIR to a local model dir, or it
    tries the hub id. See the PaddleOCR-VL model card for exact usage."""
    name = "paddleocr_vl"
    _MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"

    def __init__(self):
        self._pipe = None
        self._loaded = False

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            return False

    def supports(self, mac_lang: str) -> bool:
        return mac_lang in {"en", "ko", "ja", "es"}

    def _load(self):
        if self._loaded:
            return self._pipe
        self._loaded = True
        try:
            import os
            from transformers import AutoModelForCausalLM, AutoProcessor
            src = os.environ.get("MACULAR_PADDLEVL_DIR", self._MODEL)
            proc = AutoProcessor.from_pretrained(src, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                src, trust_remote_code=True, device_map="auto")
            self._pipe = (proc, model)
        except Exception:
            self._pipe = None   # unavailable at runtime -> recognize returns ""
        return self._pipe

    def recognize(self, crop, mac_lang: str) -> str:
        pipe = self._load()
        if pipe is None:
            return ""
        try:
            proc, model = pipe
            inputs = proc(images=crop.convert("RGB"),
                          text="OCR:", return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=64)
            return proc.batch_decode(out, skip_special_tokens=True)[0].strip()
        except Exception:
            return ""


ENGINES = {
    "tesseract": TesseractEngine,
    "easyocr": EasyOCREngine,
    "paddleocr": PaddleOCREngine,
    "paddleocr_vl": PaddleOCRVLEngine,
}


class _InjectedEngine:
    """Wraps a test-provided ocr_fn(crop, tess_lang) -> str."""
    name = "injected"

    def __init__(self, ocr_fn):
        self._fn = ocr_fn

    def available(self) -> bool:
        return True

    def supports(self, mac_lang: str) -> bool:
        return True

    def recognize(self, crop, mac_lang: str) -> str:
        return self._fn(crop, LANG_MAP.get(mac_lang, mac_lang))


# --- tesseract-facing helpers (used by the env probe) ----------------------

def available() -> bool:
    return TesseractEngine().available()


def installed_languages() -> list[str]:
    return sorted(TesseractEngine()._installed())


# --- aggregation -----------------------------------------------------------

def aggregate(records: list[tuple[str, str, str]]) -> dict:
    """Aggregate (language, reference, hypothesis) into per-language and macro
    CER/WER. Pure function — unit-tested without any engine installed."""
    by_lang: dict[str, dict] = {}
    for lang, ref, hyp in records:
        ref_n, hyp_n = _normalize(ref), _normalize(hyp)
        d = by_lang.setdefault(lang, {"cer": [], "wer": [], "n": 0})
        d["cer"].append(cer(ref_n, hyp_n))
        if lang not in NO_WORD_SEGMENTATION:
            d["wer"].append(wer(ref_n, hyp_n))
        d["n"] += 1

    per_language = {}
    for lang, d in by_lang.items():
        n = d["n"]
        wer_mean = (sum(d["wer"]) / len(d["wer"])) if d["wer"] else None
        per_language[lang] = {
            "n_regions": n,
            "cer_mean": sum(d["cer"]) / n if n else None,
            "wer_mean": wer_mean,
        }
    cer_langs = [v for v in per_language.values() if v["cer_mean"] is not None]
    wer_langs = [v for v in per_language.values() if v["wer_mean"] is not None]
    macro = {
        "cer_macro_over_languages": (
            sum(v["cer_mean"] for v in cer_langs) / len(cer_langs)
            if cer_langs else None
        ),
        "wer_macro_over_languages": (
            sum(v["wer_mean"] for v in wer_langs) / len(wer_langs)
            if wer_langs else None
        ),
    }
    return {"per_language": per_language, "macro": macro}


# --- runner ----------------------------------------------------------------

def run(docs, data_dir: str, engine: str = "tesseract", ocr_fn=None) -> dict:
    """Region-level, per-language OCR baseline with the chosen engine.

    ocr_fn(crop, tess_lang) -> str can be injected for testing; otherwise the
    named engine ('tesseract' or 'easyocr') is used.
    """
    if ocr_fn is not None:
        eng = _InjectedEngine(ocr_fn)
    else:
        cls = ENGINES.get(engine)
        if cls is None:
            return {
                "experiment": "ocr_baseline",
                "error": f"unknown engine '{engine}'",
                "available_engines": sorted(ENGINES),
            }
        eng = cls()
        if not eng.available():
            return {
                "experiment": "ocr_baseline",
                "engine": engine,
                "skipped": True,
                "reason": f"engine '{engine}' is not installed",
            }

    from PIL import Image

    records: list[tuple[str, str, str]] = []
    skipped_langs: dict[str, str] = {}
    n_docs_used = 0

    for doc in docs:
        if doc.language not in LANG_MAP:
            skipped_langs[doc.language] = "no LANG_MAP entry"
            continue
        if not eng.supports(doc.language):
            skipped_langs[doc.language] = f"engine '{eng.name}' cannot handle it"
            continue
        if not doc.image_path:
            continue
        img_path = os.path.join(data_dir, doc.image_path)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path)
        used_any = False
        for c in doc.candidates:
            if not c.text.strip():
                continue
            crop = _crop(img, c.bbox, doc.width, doc.height)
            if crop is None:
                continue
            records.append((doc.language, c.text, eng.recognize(crop, doc.language)))
            used_any = True
        if used_any:
            n_docs_used += 1

    if not records:
        return {
            "experiment": "ocr_baseline",
            "engine": eng.name,
            "skipped": True,
            "reason": "no OCR-able regions (missing images or unsupported languages)",
            "skipped_languages": skipped_langs,
        }

    result = aggregate(records)
    result.update(
        {
            "experiment": "ocr_baseline",
            "engine": eng.name,
            "level": "region",
            "n_documents": n_docs_used,
            "n_regions": len(records),
            "skipped_languages": skipped_langs,
            "note": (
                "Region-level CER per language (WER null for CJK). Tesseract is "
                "a weak floor for CJK; try engine=easyocr, target is PaddleOCR-VL."
            ),
        }
    )
    return result
