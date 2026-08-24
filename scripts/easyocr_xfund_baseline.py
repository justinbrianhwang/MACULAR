"""Classic OCR engine (EasyOCR) on the IDENTICAL 1,198 XFUND ja/es eval crops.

Reviewer response: the adaptation tables compare a VLM against itself; this
puts a conventional OCR engine on the same crops (same halving, interleave,
doc/region caps and 512px crop cap as finetune_and_measure) so the
"adaptation vs. model swap vs. classic engine" question has one eval set.

Usage: python scripts/easyocr_xfund_baseline.py  -> results/easyocr_xfund_eval_half.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macular.baselines.ocr import EasyOCREngine
from macular.evaluation.metrics import cer as _cer
from macular.models.ocr_adapt import (_regions, halve_by_language,
                                      interleave_by_language)
from macular.schema import read_jsonl

DATA = "data/xfund"
docs = read_jsonl(os.path.join(DATA, "test.jsonl"))
_, evald = halve_by_language(docs)
evald = interleave_by_language(evald)
items = list(_regions(evald, DATA, max_docs=50, max_regions=24))
assert len(items) == 1198, len(items)

eng = EasyOCREngine()
pairs, per_lang = [], {}
for i, (crop, gold, lang) in enumerate(items):
    pred = eng.recognize(crop, lang)
    pairs.append((lang, pred, gold))
    d = per_lang.setdefault(lang, {"num": 0.0, "den": 0, "em": 0, "n": 0})
    w = max(1, len(gold))
    d["num"] += _cer(pred, gold) * w
    d["den"] += w
    d["em"] += int(pred.strip() == gold.strip())
    d["n"] += 1
    if i % 100 == 0:
        print(f"{i}/{len(items)}", flush=True)

out = {"engine": "easyocr", "n_eval_regions": len(items),
       "cer": {l: {"cer": d["num"] / d["den"], "exact_match": d["em"] / d["n"],
                   "n_regions": d["n"]} for l, d in per_lang.items()},
       "eval_pairs": pairs,
       "note": "Same eval half / crops as results/ocr_adapt_xfund*.json."}
json.dump(out, open("results/easyocr_xfund_eval_half.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print({l: (round(v["cer"], 3), round(v["exact_match"], 3)) for l, v in out["cer"].items()})
