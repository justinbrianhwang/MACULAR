"""FUNSD adapter: download + convert to MACULAR Documents.

FUNSD (Jaume et al., 2019) is 199 real, noisy, scanned English forms with
word/entity boxes, transcriptions, and labels. Public research dataset
(https://guillaumejaume.github.io/FUNSD/). No PHI.

Usage (via the runner):  macular run fetch_funsd --config configs/funsd.yaml
Then:                     macular run ocr_baseline --config configs/funsd.yaml
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile

from ..schema import write_jsonl
from .funsd_format import form_items_to_document

FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"

# (split_dir_in_zip, our_split_name)
_SPLITS = [("training_data", "train"), ("testing_data", "test")]


def ensure_funsd(dest: str) -> str:
    """Download and extract FUNSD into ``dest`` (idempotent). Returns the path
    to the extracted ``dataset`` directory."""
    os.makedirs(dest, exist_ok=True)
    dataset_dir = os.path.join(dest, "dataset")
    if os.path.isdir(os.path.join(dataset_dir, "training_data")):
        return dataset_dir
    import urllib.request
    zip_path = os.path.join(dest, "funsd.zip")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(FUNSD_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    if not os.path.isdir(dataset_dir):
        # some mirrors nest differently; find the folder that has training_data
        for root, dirs, _ in os.walk(dest):
            if "training_data" in dirs:
                return root
    return dataset_dir


def convert_funsd(dataset_dir: str, out_dir: str, copy_images: bool = True) -> dict:
    """Convert extracted FUNSD into ``out_dir`` as MACULAR jsonl + images."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    img_out = os.path.join(out_dir, "images")
    if copy_images:
        os.makedirs(img_out, exist_ok=True)

    manifest = {"dataset": "FUNSD", "source": FUNSD_URL,
                "license": "research use; see FUNSD dataset page",
                "out_dir": out_dir, "splits": {}}

    for zip_split, our_split in _SPLITS:
        ann_dir = os.path.join(dataset_dir, zip_split, "annotations")
        img_dir = os.path.join(dataset_dir, zip_split, "images")
        if not os.path.isdir(ann_dir):
            continue
        docs = []
        for fn in sorted(os.listdir(ann_dir)):
            if not fn.endswith(".json"):
                continue
            stem = os.path.splitext(fn)[0]
            img_src = os.path.join(img_dir, stem + ".png")
            if not os.path.exists(img_src):
                continue
            with Image.open(img_src) as im:
                w, h = im.size
            with open(os.path.join(ann_dir, fn), "r", encoding="utf-8") as f:
                items = json.load(f).get("form", [])
            rel_img = None
            if copy_images:
                shutil.copy(img_src, os.path.join(img_out, stem + ".png"))
                rel_img = os.path.join("images", stem + ".png")
            docs.append(form_items_to_document(
                items, w, h, doc_id=f"funsd-{our_split}-{stem}",
                language="en", split=our_split, image_path=rel_img))
        path = os.path.join(out_dir, f"{our_split}.jsonl")
        write_jsonl(docs, path)
        manifest["splits"][our_split] = {"n": len(docs),
                                         "labels": os.path.basename(path)}
    return manifest


def fetch_and_convert(raw_dir: str, out_dir: str) -> dict:
    dataset_dir = ensure_funsd(raw_dir)
    return convert_funsd(dataset_dir, out_dir)
