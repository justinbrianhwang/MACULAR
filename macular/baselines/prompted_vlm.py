"""Zero-shot prompted-VLM baseline for sensitive-region detection.

The comparison so far is between redaction *mechanisms* sitting on top of a
trained head. A reviewer in 2026 will ask the obvious question first: why train
anything at all — can a capable VLM just be asked which regions are sensitive?

This runs that baseline honestly. The model sees the page and one region's
bounding box and text, and answers yes/no. Scored with the same P/R/F1 as the
trained head so the numbers sit in one table.

Two things to keep straight when reading the result:

* This baseline solves DETECTION only. It says nothing about representation
  leakage, which is what the erasure comparison measures — a prompted model that
  detects PII perfectly still hands downstream a representation that an attacker
  can invert. Detection and redaction are different problems and the paper must
  not let them blur.
* Generative scoring is slow: one forward pass per region rather than one per
  page. Region budgets here are small on purpose.
"""

from __future__ import annotations

import torch

PROMPT = (
    "You are auditing a scanned form for personal data.\n"
    "Region text: {text!r}\n"
    "Does this region contain personal identifying information "
    "(a person's name, date of birth, phone number, address, email, or an "
    "identifier such as a patient or national ID)?\n"
    "Answer with exactly one word: yes or no."
)


def _answer_is_yes(text: str) -> bool:
    t = text.strip().lower()
    # Take the first word the model actually commits to; models often continue
    # past the answer, and "no" is a prefix of nothing useful here.
    for token in t.replace(".", " ").replace(",", " ").split():
        if token.startswith("yes"):
            return True
        if token.startswith("no"):
            return False
    return False


@torch.no_grad()
def classify_regions(backbone, image, texts, max_new_tokens: int = 4):
    """Ask the model, per region, whether it holds PII. Returns list[bool]."""
    proc, model = backbone._processor, backbone._model
    dev = next(model.parameters()).device
    out = []
    for text in texts:
        msg = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT.format(text=text)}]}]
        try:
            prompt = proc.apply_chat_template(msg, add_generation_prompt=True,
                                              tokenize=False)
        except Exception:
            prompt = PROMPT.format(text=text)
        inputs = proc(images=image, text=prompt, return_tensors="pt").to(dev)
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
        reply = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]
        out.append(_answer_is_yes(reply))
    return out


def run(docs, data_dir, backbone, max_docs=30, max_regions=24):
    """Score the prompted baseline against gold PII labels."""
    import os

    from PIL import Image

    tp = fp = fn = tn = 0
    n_regions = 0
    for doc in docs[:max_docs]:
        cands = doc.candidates[:max_regions]
        if not cands:
            continue
        img = Image.open(os.path.join(data_dir, doc.image_path)).convert("RGB")
        preds = classify_regions(backbone, img, [c.text or "" for c in cands])
        for c, p in zip(cands, preds):
            gold = bool(c.pii_type)
            tp += p and gold
            fp += p and not gold
            fn += (not p) and gold
            tn += (not p) and not gold
            n_regions += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n_documents": min(len(docs), max_docs), "n_regions": n_regions,
        "precision": prec, "recall": rec, "f1": f1,
        "positive_rate_gold": (tp + fn) / max(1, n_regions),
        "positive_rate_pred": (tp + fp) / max(1, n_regions),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "note": ("Detection only. A model that detects PII perfectly still hands "
                 "downstream a representation an attacker can invert, which is "
                 "what the erasure comparison measures."),
    }
