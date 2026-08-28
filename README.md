# MACULAR

**Domain-Adapted OCR for Multilingual Medical Documents — Transfer, Reproducibility, and the Limits of Representation-Level Redaction**

Sunjun Hwang · Huiseung Shin — Yonsei University

[![Data DOI](https://img.shields.io/badge/data-10.5281%2Fzenodo.22078532-blue)](https://doi.org/10.5281/zenodo.22078532)

MACULAR is a research codebase for three coupled problems in medical-document
OCR, evaluated on a controlled synthetic medical-form corpus (Korean, Japanese,
English; held-out PII value families) and on real scanned forms (FUNSD, XFUND):

1. **Domain adaptation of compact vision–language OCR models** (Qwen2-VL-2B,
   Qwen2.5-VL-3B) with LoRA and its successors (DoRA, rsLoRA, PiSSA, VeRA, full
   fine-tuning), including a reproducibility study — identical seed, code and
   hardware can produce a 6× CER spread — traced to GPU kernel
   non-determinism with a deterministic-kernel control, and a sized,
   prospectively validated post-training **evaluation gate**.
2. **The OCR error cascade** into downstream extraction, measurable only after
   geometric shortcut learning is controlled with counterfactual layouts.
3. **Representation-level redaction under attack**: structural masking, a
   differentiable gate with adversarial training, and closed-form linear
   concept erasure (LEACE), attacked with fresh linear/MLP probes and
   embedding inversion at the mechanism output and after contextualisation,
   under held-out PII value families, on three VLM backbones.

Every number in the paper is regenerable from `results/*.json`; per-run
outputs (including every per-region prediction) are archived at
**https://doi.org/10.5281/zenodo.22078532**. `FINDINGS.md` is the running
findings log, including all retracted intermediate conclusions.

> No real patient data is used anywhere. Synthetic PII values are drawn from
> disjoint value families; the real-scan corpora are public research datasets.

---

## Headline findings

| Pillar | Result |
|---|---|
| Adaptation | LoRA improves CER in every pooled real-scan run (ja median 0.846 → 0.151 pooled / 0.114 in the ja+es configuration), but a newer backbone reaches the same level zero-shot; adaptation on top of the 3B still helps (0.118 → 0.047–0.075). |
| Reproducibility | Same seed, same code, same GPU: after-adaptation CER 0.099 vs 0.630. Deterministic kernels make the pair bit-identical, yet bad basins persist under determinism (seed 3 EM 0.274). |
| Evaluation gate | ~100 regions (~70 s) separate every ΔCER ≥ 0.10 divergence in ≥ 99.5 % of region- and document-level resamples; a fixed rule applied prospectively to 34 later-trained language cells agrees with the full evaluation in 32. |
| PEFT variants | The method ranking at the shared lr 1e-4 is largely a learning-rate artefact; at a matched 3e-5 the methods sit within ~0.04 CER and none diverged. Full fine-tuning is worse than every PEFT cell. |
| Transfer | Purely synthetic training recovers ~94 % of the in-domain CER gain on real Japanese scans and improves Spanish, which is absent from synthetic training. |
| Cascade | Downstream F1 is flat in OCR quality on default layouts (the model reads geometry) and falls 0.075–0.094 on counterfactual layouts, 3/3 seeds. |
| Redaction | Learned gating is dominated everywhere; LEACE keeps only 19–67 % of its linear protection under value shift; structural masking (driven by the model's own PII detector, recall 0.98 at threshold) is the only mechanism at or near the majority-baseline floor after contextualisation, and its post-graph balanced accuracy collapses to 0.16–0.20 (chance 0.125). A wider, longer-trained, beam-searched inverter does not change the ordering. |

---

## Repository layout

```
macular/
  data/          synthetic medical-form generator (seeded, bit-reproducible), PII value families
  realdata/      FUNSD / XFUND fetch + conversion to the common region schema
  models/
    ocr_adapt.py     LoRA / DoRA / rsLoRA / PiSSA / VeRA / full-FT adaptation + paired CER eval
    core.py          MACULAR core: projector, redaction gate / hard mask / LEACE, relation graph
    train.py         core training, OCR error cascade, erasure comparison, backbone contribution
    features.py      text-hash and VLM region features, controlled OCR corruption
    vlm_backbone.py  frozen VLM feature extraction (PaddleOCR-VL, Qwen2-VL, Ministral-3, ...)
  privacy/       LEACE, linear/MLP probes, GRU inversion attack, prior floor
  baselines/     EasyOCR / Tesseract region-level baselines, coordinate-only shortcut audit
  evaluation/    CER / WER / exact match
  runner.py      `python -m macular.runner run <experiment> --config <yaml>`
configs/         one YAML per experiment reported in the paper
scripts/         analyses that produce the paper's tables and figures
results/         JSON outputs (archived at the Zenodo DOI; gitignored here)
tests/           unit tests (`pytest tests`)
FINDINGS.md      running findings log incl. retractions
```

---

## Setup

```bash
conda env create -f environment.yml        # or: pip install -e ".[model,easyocr]"
conda activate macular
pip install -r requirements-lock.txt       # exact versions used for the paper
```

Paper environment: transformers 5.14.1, peft 0.20.0, PyTorch 2.11.0+cu128,
EasyOCR 1.7.2, bf16, one RTX 5090 (32 GB).

## Reproducing the paper

```bash
# corpora
python -m macular.runner run data_gen  --config configs/cascade_cf_gen.yaml --out results
python -m macular.runner run fetch_funsd --config configs/funsd.yaml --out results
python -m macular.runner run fetch_xfund --config configs/xfund.yaml --out results

# §4 adaptation (each YAML = one table row; ~50 min per seed on a 5090)
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund.yaml --out results
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund_det.yaml --out results   # deterministic kernels
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund_r16_lr3e5.yaml --out results

# §4 gate analyses and model-swap comparison
python scripts/gate_reviewer_response.py
python scripts/easyocr_xfund_baseline.py

# §5 OCR error cascade (CPU)
python -m macular.runner run ocr_propagation --config configs/cascade_cf_ep80.yaml --out results

# §6 redaction under attack
python -m macular.runner run erasure_comparison --config configs/erasure_comparison.yaml --out results
python scripts/strong_attack.py            # stronger inverter + probe selectivity
python scripts/detector_probe_metrics.py   # detector operating point, balanced accuracy

# figures
python scripts/paper_figures.py
```

Individual training runs are **not** bit-reproducible on non-deterministic GPU
kernels (paper §4.4); `configs/ocr_adapt_xfund_det.yaml` shows the
deterministic setting. Report distributions over seeds, never a single run.

---

## Data

- **Synthetic medical forms**: laboratory reports and prescriptions in ko/ja/en,
  rendered with pan-CJK fonts and a light scan simulation. Every region carries
  gold text, language, clinical type and PII type. PII values come from
  disjoint families A–E (disjoint name, street, organisation, phone-prefix and
  ID pools), assigned per split, so no evaluation value ever occurs in training.
- **FUNSD** (real scanned English forms) and **XFUND** (ja/es/zh): public
  corpora, used as sources of real region crops for recognition and of real
  region features for the redaction attacks. XFUND self-splits are
  language-stratified document-level halves; a perceptual-hash audit found no
  meaningful template sharing across halves.

Manifests for every corpus are in the Zenodo archive; images re-render from
the seeded generator or are fetched from the public sources.

---

## Citation

```bibtex
@article{hwang2026macular,
  title   = {MACULAR: Domain-Adapted OCR for Multilingual Medical Documents ---
             Transfer, Reproducibility, and the Limits of Representation-Level Redaction},
  author  = {Hwang, Sunjun and Shin, Huiseung},
  year    = {2026},
  note    = {Data and per-run outputs: \url{https://doi.org/10.5281/zenodo.22078532}}
}
```

## License

Code: MIT (see `LICENSE`). Data and per-run outputs: CC BY 4.0 (Zenodo).
