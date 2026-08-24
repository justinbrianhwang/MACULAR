"""Build Dataset.zip for the Zenodo release: per-run outputs, split manifests,
configs, FINDINGS and a README. Images and feature caches are excluded
(regenerable from the repo; XFUND/FUNSD are public).

Usage: python scripts/package_dataset.py  -> Dataset.zip (+ Dataset/ staging dir)
"""
import glob
import json
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Dataset")

README = """# MACULAR — per-run outputs, split manifests and configurations

Companion data for *MACULAR: Domain-Adapted OCR for Multilingual Medical
Documents — Transfer, Reproducibility, and the Limits of Representation-Level
Redaction* (Hwang & Shin, Yonsei University). Code:
https://github.com/justinbrianhwang/MACULAR

No real patient data is included. Synthetic forms carry generated values from
disjoint PII value families; real-scan manifests reference the public FUNSD and
XFUND corpora by document id only (download them with `macular run
fetch_funsd` / `fetch_xfund`).

## Layout

- `results/ocr_adapt*.json` — every adaptation run (paper §4). Each file holds
  the baseline (`cer_before`, `eval_pairs_before`) and, per seed,
  `cer_after`, `loss_history` and `eval_pairs` = list of
  `[language, prediction, gold]` for every evaluation region — the data behind
  all gate analyses. `note` states the split regime of the run.
- `results/easyocr_xfund_eval_half.json` — EasyOCR on the identical XFUND crops.
- `results/gate_reviewer_response.json` — document-level / sibling / prospective
  gate analysis (paper Table 3).
- `results/ocr_propagation_*.json` — OCR error cascade, 3 seeds, both layouts
  (paper §5).
- `results/erasure_comparison_*.json`, `leace_*.json`,
  `strong_attack_paddleocr_vl.json` — redaction mechanisms under attack
  (paper §6), incl. the stronger inverter and probe-selectivity control.
- `results/xfund_template_audit.json` — near-duplicate template audit.
- `results/backbone_gate.json`, `ablation.json`, `lora_ablation*.json`,
  `engine_downstream.json` — backbone contribution and ablations.
- `manifests/<corpus>/*.jsonl` — document/region manifests (gold text, boxes,
  PII and clinical labels, generator family). Images are not included: run
  `macular run data_gen --config configs/<...>.yaml` to re-render the
  synthetic corpora bit-identically (seeded), or fetch FUNSD/XFUND.
- `configs/` — every experiment configuration used.
- `FINDINGS.md` — the project's running findings log, including retractions.

## Reproducing a number

    pip install -e ".[model]"
    python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund.yaml --out results
    python scripts/gate_reviewer_response.py

Environment: transformers 5.14, peft 0.20, PyTorch 2.x, bf16, one RTX 5090.
Individual runs are not bit-reproducible on non-deterministic kernels (paper
§4.4); `configs/ocr_adapt_xfund_det.yaml` shows the deterministic setting.

## License

Data and outputs: CC BY 4.0. Code: see repository.
"""


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "results"))
    for f in glob.glob(os.path.join(ROOT, "results", "*.json")):
        shutil.copy(f, os.path.join(OUT, "results"))
    shutil.copytree(os.path.join(ROOT, "configs"), os.path.join(OUT, "configs"))
    for corpus in ("meddoc_cf_engines", "meddoc_cf", "meddoc_default",
                   "meddoc_cf_famD", "meddoc_cf_famE", "xfund", "xfund_cjk", "funsd"):
        src = os.path.join(ROOT, "data", corpus)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(OUT, "manifests", corpus)
        os.makedirs(dst)
        for f in glob.glob(os.path.join(src, "*.jsonl")):
            shutil.copy(f, dst)
    shutil.copy(os.path.join(ROOT, "FINDINGS.md"), OUT)
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(README)
    zpath = os.path.join(ROOT, "Dataset.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, files in os.walk(OUT):
            for f in files:
                p = os.path.join(d, f)
                z.write(p, os.path.relpath(p, ROOT))
    print(zpath, round(os.path.getsize(zpath) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
