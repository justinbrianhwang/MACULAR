"""XFUND adapter: multilingual real scanned forms (JA, ES, ZH, FR, IT, DE, PT).

XFUND (Xu et al., 2021) extends FUNSD to 7 languages with the same annotation
shape. Public research dataset (https://github.com/doc-analysis/XFUND). No PHI.
We use it for real scanned Japanese/Spanish forms to compare OCR engines on
genuine non-English images.

Per language + split, XFUND ships a JSON (``{lang}.{split}.json``) and a zip of
images (``{lang}.{split}.zip``). Language codes: zh, ja, es, fr, it, de, pt.

Usage:  macular run fetch_xfund --config configs/xfund.yaml
Then:   macular run ocr_baseline --config configs/xfund.yaml
"""

from __future__ import annotations

import json
import os
import zipfile

from ..schema import write_jsonl
from .funsd_format import form_items_to_document

_BASE = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"

# XFUND language code -> MACULAR language code (only those we score here).
_LANG_MAP = {"ja": "ja", "es": "es"}


def _download(url: str, dest: str) -> str:
    import urllib.request
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)
    return dest


def fetch_and_convert(raw_dir: str, out_dir: str,
                      langs: list[str], split: str = "val") -> dict:
    """Download the given XFUND languages for one split and convert to jsonl.

    XFUND's public split is train/val; we default to ``val``.
    """
    from PIL import Image

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    img_out = os.path.join(out_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    manifest = {"dataset": "XFUND", "source": _BASE,
                "license": "research use; see XFUND repository",
                "out_dir": out_dir, "split": split, "splits": {}}

    all_docs = []
    for lang in langs:
        if lang not in _LANG_MAP:
            continue
        json_url = f"{_BASE}/{lang}.{split}.json"
        zip_url = f"{_BASE}/{lang}.{split}.zip"
        json_path = _download(json_url, os.path.join(raw_dir, f"{lang}.{split}.json"))
        zip_path = _download(zip_url, os.path.join(raw_dir, f"{lang}.{split}.zip"))
        img_dir = os.path.join(raw_dir, f"{lang}.{split}")
        os.makedirs(img_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(img_dir)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data.get("documents", []):
            img_meta = doc.get("img", {})
            fname = img_meta.get("fname")
            w = img_meta.get("width")
            h = img_meta.get("height")
            src = _find_image(img_dir, fname)
            if src is None:
                continue
            if not (w and h):
                with Image.open(src) as im:
                    w, h = im.size
            rel = os.path.join("images", f"{lang}-{fname}")
            import shutil
            shutil.copy(src, os.path.join(out_dir, rel))
            all_docs.append(form_items_to_document(
                doc.get("document", []), w, h,
                doc_id=f"xfund-{lang}-{fname}",
                language=_LANG_MAP[lang], split=split, image_path=rel))

    # XFUND has no separate test split here; expose everything as the test set
    # so ocr_baseline (which reads test.jsonl) runs on it.
    path = os.path.join(out_dir, "test.jsonl")
    write_jsonl(all_docs, path)
    manifest["splits"]["test"] = {"n": len(all_docs),
                                  "labels": "test.jsonl", "langs": langs}
    return manifest


def _find_image(root: str, fname: str | None):
    if fname:
        for dp, _, fns in os.walk(root):
            if fname in fns:
                return os.path.join(dp, fname)
    return None
