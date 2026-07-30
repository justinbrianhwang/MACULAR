"""Backward-compatibility shim. The OCR baseline now lives in ``ocr`` and
supports multiple engines (tesseract, easyocr). Import from ``macular.baselines.ocr``.
"""

from .ocr import (  # noqa: F401
    run,
    aggregate,
    available,
    installed_languages,
    LANG_MAP,
    NO_WORD_SEGMENTATION,
    ENGINES,
)
