# MACULAR

**Domain-Adapted OCR for Multilingual Medical Documents — Transfer, Reproducibility, and the Limits of Representation-Level Redaction**

Sunjun Hwang (Division of Software, Yonsei University) · Huiseung Shin (Division of Computer and Telecommunications Engineering, Yonsei University)

[![Data DOI](https://img.shields.io/badge/data-10.5281%2Fzenodo.22078532-blue)](https://doi.org/10.5281/zenodo.22078532)
[![License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

## Contents

1. [What this is](#what-this-is)
2. [Headline findings](#headline-findings)
3. [Method details](#method-details)
   - [Adaptation (paper §4)](#adaptation-paper-4)
   - [Evaluation gate](#evaluation-gate)
   - [OCR error cascade (paper §5)](#ocr-error-cascade-paper-5)
   - [Redaction under attack (paper §6)](#redaction-under-attack-paper-6)
4. [Corpora](#corpora)
5. [Repository layout](#repository-layout)
6. [Setup](#setup)
7. [Reproducing every table and figure](#reproducing-every-table-and-figure)
8. [Experiment index (config → result → paper)](#experiment-index)
9. [Reproducibility notes and known pitfalls](#reproducibility-notes-and-known-pitfalls)
10. [Retracted intermediate conclusions](#retracted-intermediate-conclusions)
11. [Data archive](#data-archive)
12. [Citation](#citation)
13. [License](#license)

---

## What this is

Medical forms concentrate three difficulties that general document-OCR
benchmarks do not: multilingual content including CJK scripts, downstream
clinical tasks that inherit every recognition error, and text whose most
information-dense regions are exactly the ones that must not leak. MACULAR is
the research code for one paper that studies all three on a controlled
synthetic medical-form corpus (Korean, Japanese, English; held-out PII value
families) and on two real scanned-form corpora (FUNSD, XFUND):

1. **Domain adaptation of compact vision–language OCR models** — Qwen2-VL-2B and
   Qwen2.5-VL-3B with LoRA and its successors (DoRA, rsLoRA, PiSSA, VeRA), a
   full-fine-tuning control, an lr-matched sweep, synthetic→real transfer, and a
   reproducibility study: identical seed, code and hardware produced a 6× CER
   spread, isolated to GPU kernel non-determinism by a deterministic-kernel
   control, with bad basins that persist under determinism. From this we size
   and prospectively validate a post-training **evaluation gate**.
2. **The OCR error cascade** into downstream extraction — measurable only once
   geometric shortcut learning is defeated with counterfactual layouts.
3. **Representation-level redaction under attack** — structural masking, a
   differentiable gate with adversarial training, and closed-form linear concept
   erasure (LEACE), attacked with fresh linear/MLP probes and embedding
   inversion at the mechanism output *and* after contextualisation, under
   held-out PII value families, on three VLM backbones (PaddleOCR-VL-0.9B,
   Qwen2-VL-2B, Ministral-3-8B) and real scans.

Every number in the paper is regenerable from `results/*.json`; per-run outputs
(including every per-region prediction/gold pair) are archived at
**https://doi.org/10.5281/zenodo.22078532**. `FINDINGS.md` is the project's
running findings log, including all retracted intermediate conclusions.

> **No real patient data is used anywhere.** Synthetic PII values are generated
> from disjoint value families; the real-scan corpora (FUNSD, XFUND) are public
> research datasets of general administrative forms. Korean results are
> synthetic-only because no public Korean scanned medical corpus exists — we
> state this as a field gap, not a solved problem.

---

## Headline findings

| Pillar | Result |
|---|---|
| **Adaptation** | LoRA improves CER in every pooled real-scan run (ja median 0.846 → 0.151 pooled over both co-training configurations, 0.114 in ja+es). Most of that is recovery from a junk-output baseline: Qwen2.5-VL-3B reaches the same level *zero-shot* (0.118); adaptation on top of the 3B still helps in 2/3 seeds (→ 0.047 / 0.075). EasyOCR on the identical crops: ja 0.213 / es 0.208. |
| **Reproducibility** | Same seed, same code, same GPU: after-adaptation CER 0.099 vs 0.630, invisible in training loss. With deterministic kernels the pair is bit-identical (1,198/1,198 predictions) — yet seed 3 lands in a bad basin reproducibly (ja EM 0.274). Determinism makes bad draws repeatable, not gone. |
| **Evaluation gate** | ~100 regions (~70 s of inference) separate every ΔCER ≥ 0.10 divergence in ≥ 99.5 % of region-level *and* document-level resamples; a 0.04 margin is not separable; a runaway-generation case (79 % of excess CER in 10 regions) needs per-region CER clipping / an EM criterion. A rule fixed in advance and applied to 34 later-trained language cells agrees with the full evaluation in 32. |
| **PEFT variants** | At the shared lr 1e-4 the variant ranking spans 10× (rsLoRA r8 best, rsLoRA r16 model-destroying). At a matched 3e-5 all methods sit within ~0.04 CER, Spanish EM is above baseline everywhere, and none of 15 runs diverged: the ranking was a learning-rate artefact; the effective update size is what matters. Full fine-tuning (1.54B params, lr 1e-5) is worse than every PEFT cell. |
| **Transfer** | Training on purely rendered forms recovers ~94 % of the in-domain CER gain on real Japanese scans and improves Spanish, which is absent from synthetic training; one seed shows a *language-selective* divergence (es only) that a macro-averaged gate would pass. |
| **Cascade** | Downstream PII F1 is flat in OCR quality on default layouts (0.948 → 0.944: the model reads geometry) and falls 0.075–0.094 on counterfactual layouts, 3/3 seeds. |
| **Redaction** | Learned gating is dominated on every backbone and destabilises training. LEACE is exact where fitted but keeps only 19–67 % of its linear protection under value-family shift. Structural masking — driven by the model's own PII detector (recall 0.98 at the deployed threshold, on the held-out family), not an oracle — is the only mechanism at or near the majority-baseline floor after contextualisation; its post-graph balanced accuracy collapses to 0.16–0.20 (chance 0.111). A wider, longer-trained, beam-searched inverter and a probe-selectivity control leave the ordering unchanged. |

---

## Method details

### Adaptation (paper §4)

| | |
|---|---|
| Backbone | `Qwen/Qwen2-VL-2B-Instruct` (second backbone `Qwen/Qwen2.5-VL-3B-Instruct`), `AutoModelForImageTextToText`, bf16 |
| Input | one region crop per sample, chat-templated; prompt *"Transcribe the text in this image exactly. Output only the text."*; loss on target tokens only |
| Crop cap | longest side ≤ 512 px (`MAX_CROP_SIDE`). Load-bearing: uncapped real-scan crops become thousands of visual tokens under dynamic resolution |
| LoRA | r = 16, α = 32 (α/r = 2 held fixed in the rank ablation), dropout 0.05, no bias; targets discovered by name on **both** towers: `q/k/v/o_proj, qkv, proj, fc1, fc2, gate/up/down_proj` → 29.0M params (14.5M at r = 8) |
| Variants | `use_dora`, `use_rslora`, `init_lora_weights="pissa_niter_16"` (Linear-only targets), `VeraConfig(r=256)` (1.05M params); adapter cast to base dtype after injection |
| Full-FT control | language tower unfrozen (1.54B), vision frozen, gradient checkpointing, lr 1e-5 |
| Optimiser | AdamW, lr 1e-4 (default) / 3e-5 (lr-matched sweep), constant, no warm-up, 1 region per step, grad-clip 1.0, 2 epochs (1 in the budget ablation) |
| Seeds | fresh base-model reload per seed (peft injects in place); 3 seeds per cell, 7-run pools for the default configuration |
| Data | XFUND: language-stratified document-level halves, 50 train docs × ≤ 24 regions = 1,200 train / 1,198 eval regions. Synthetic: family A train / family C eval, 1,440 / 960 regions |
| Metrics | greedy decoding, `max_new_tokens=48`; per-language length-weighted CER and exact match; WER never reported for ko/ja/zh |
| Determinism control | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`, memory-efficient + flash SDPA disabled (`deterministic: true` in the YAML) |

### Evaluation gate

Every run saves `(language, prediction, gold)` for all 1,198 evaluation regions
(`eval_pairs` in the result JSON), which makes the gate analysis a pure
resampling exercise:

- **Sizing** (`scripts/eval_gate_analysis.py`, `scripts/gate_reviewer_response.py`):
  paired bootstrap over gate sets of *n* regions (same regions for both
  adapters), region-level and document-level (whole pages drawn until *n*
  regions accumulate), for subtle-vs-baseline, gross-vs-sibling and
  beats-baseline-but-loses-to-sibling cases.
- **Decision rule** (fixed before the later runs existed): per-language
  100-region gate; reject if gate CER exceeds a ceiling of 1.5 × the
  per-language median of the 7-run ja+es default pool (medians 0.114 / 0.299 →
  ceilings 0.171 / 0.449). Applied prospectively to the 17 adapters trained
  afterwards: 3 rejects, 31 accepts, 32/34 agreeing with the full evaluation.
- **Recommendation**: clip per-region CER at 1 and gate on exact match as well,
  because runaway generations concentrate CER mass in a few regions.

### OCR error cascade (paper §5)

A small core model (`macular/models/core.py`: linear projector → redaction
mechanism → 2-layer Transformer relation graph, d = 128, 4 heads) is trained on
per-region text-hash features under controlled character corruption at CER
∈ {0, 0.1, 0.2, 0.3, 0.5} (`features.corrupt_text`, confusable-pool
substitutions), on default layouts and on **counterfactual** layouts where
region positions are shuffled against content. A coordinate-only audit
classifier (`baselines/coordinate_only.py`) certifies the shortcut is gone
(F1 ≈ 1.0 default → 0.43 counterfactual). 3 seeds, 400 docs, 80 epochs.

### Redaction under attack (paper §6)

| | |
|---|---|
| Features | frozen per-region VLM features: one backbone forward per page, ROI-pooled per region, standardised with train statistics (`features.documents_to_vlm_batch`) |
| Backbones | PaddleOCR-VL-0.9B, Qwen2-VL-2B, Ministral-3-8B (`models/vlm_backbone.py`); FUNSD real-scan arm with PaddleOCR-VL |
| Mechanisms | `none`; `hard_mask` — regions flagged by the model's **own** PII head (threshold 0.5, gradient cut) are replaced by a detached probability-weighted *type embedding* (type retained by design, value dropped); `gate` — differentiable soft mask trained with a gradient-reversal adversary; `leace` — closed-form affine eraser fitted on train pooled features (`privacy/leace.py`) |
| Surfaces | `z_safe` (mechanism output) and `z_ctx` (after the relation graph — what is actually shipped downstream) |
| Attacks | fresh linear probe, fresh 2-layer MLP probe (256 hidden, 300 epochs; PII type, 9 classes, majority 0.847); GRU inversion decoder (h = 512, 400 epochs, teacher-forced, greedy) and a weaker per-position linear decoder, scored by exact match and CER on PII regions; **prior floor** from a decoder trained on shuffled feature/text pairs (EM 0.000) |
| Stronger attacker | `scripts/strong_attack.py`: GRU h = 1024, 1,200 epochs, beam k = 5; Hewitt–Liang selectivity control (random label per distinct text) |
| Detector metrics | `scripts/detector_probe_metrics.py`: is-PII precision/recall at the deployed threshold, balanced accuracy and per-class recall of the MLP probe |
| Utility | macro-F1 of a downstream clinical-type head (`LAB_NAME, LAB_VALUE, UNIT, REF_RANGE, MED_NAME, DOSE, FREQUENCY, DURATION`) on the same representation the attacker sees |
| Protocol | 80 epochs, lr 3e-3, 120 documents × ≤ 48 regions, 3 seeds; LEACE fitted on family A, evaluated on family B; residual covariance checked on validation |

---

## Corpora

| Corpus | Split / size | Role |
|---|---|---|
| Synthetic medical forms (`data/meddoc_cf_engines`, counterfactual layout) | 150 / 150 / 150 docs, ~8,100 regions per split; families A / B / C | redaction study, engine comparison |
| Synthetic (`data/meddoc_default`, `data/meddoc_cf`) | 500 / 500 / 500 docs each | cascade study (default vs counterfactual layouts) |
| Synthetic families D, E (`data/meddoc_cf_famD/E`) | 150 docs each | multi-family LEACE fitting |
| FUNSD (`data/funsd`) | 149 train / 50 test docs, 9,529 regions | real-scan redaction arm (answer → sensitive, question/header → utility) |
| XFUND ja+es (`data/xfund`) | 100 docs, 6,796 regions | adaptation, transfer, gate, model-swap |
| XFUND ja+zh (`data/xfund_cjk`) | 100 docs, 7,012 regions | co-training partner comparison |

Synthetic forms: two document types (laboratory report, prescription), pan-CJK
fonts (Noto Sans CJK / Nanum Gothic fallbacks), light scan simulation (small
rotation + Gaussian noise). Every region carries gold text, language, block
type, clinical type and PII type (`PATIENT_NAME, PATIENT_ID, DOB, PHONE,
ADDRESS, EMAIL, PROVIDER_NAME, NATIONAL_ID`, or `NON_PII`). PII values come
from families A–E with disjoint given-name, surname, street, organisation and
phone-prefix pools and disjoint ID century digits, assigned per split, so an
evaluation-time PII value never occurred in training. The generator is seeded
and re-renders bit-identically.

---

## Repository layout

```
macular/
  data/
    generate.py        synthetic form generator (seeded), scan simulation, counterfactual layouts
    pii_generators.py  PII value families A–E, disjoint pools
    clinical_content.py
  realdata/            FUNSD / XFUND fetch + conversion; FUNSD privacy annotation
  models/
    ocr_adapt.py       LoRA / DoRA / rsLoRA / PiSSA / VeRA / full-FT adaptation, paired CER eval,
                       language-stratified halving + interleaving, per-region pair saving
    core.py            MacularModel: projector, RedactionGate (soft / hard), LEACE hook,
                       RelationGraph (Transformer encoder), clinical student/teacher heads
    train.py           core training, ocr_error_propagation, engine_downstream_comparison,
                       erasure_comparison, backbone_contribution, LoRA ablation
    features.py        text-hash region features, controlled corruption, VLM ROI features
    vlm_backbone.py    frozen VLM feature extraction for several backbone families
  privacy/
    leace.py           closed-form LEACE eraser + residual-covariance diagnostic
    probes.py          linear / MLP probes with majority-baseline reporting
    inversion.py       GRU + linear inversion attacks, prior floor
  baselines/           EasyOCR / Tesseract region-level baselines, OCR cache, coordinate-only audit
  evaluation/          CER / WER / exact match
  runner.py            experiment dispatcher
configs/               one YAML per experiment reported in the paper (see index below)
scripts/               analyses producing the paper's tables/figures
tests/                 pytest suite (55 tests)
FINDINGS.md            running findings log incl. retractions and the two review rounds
requirements-lock.txt  exact package versions used for the paper
```

---

## Setup

```bash
conda env create -f environment.yml     # or: pip install -e ".[model,easyocr]"
conda activate macular
pip install -r requirements-lock.txt    # exact versions used for the paper
pytest tests                            # 55 tests
```

Paper environment: transformers 5.14.1, peft 0.20.0, PyTorch 2.11.0+cu128,
EasyOCR 1.7.2, bf16, one NVIDIA RTX 5090 (32 GB). Adaptation runs take ~50 min
per seed; the redaction attack battery runs for tens of minutes on CPU per
backbone; the cascade is CPU-only (minutes).

---

## Reproducing every table and figure

```bash
# ---- corpora -------------------------------------------------------------
python -m macular.runner run data_gen    --config configs/counterfactual.yaml     --out results
python -m macular.runner run data_gen    --config configs/cascade_default_gen.yaml --out results
python -m macular.runner run data_gen    --config configs/cascade_cf_gen.yaml      --out results
python -m macular.runner run fetch_funsd --config configs/funsd.yaml               --out results
python -m macular.runner run fetch_xfund --config configs/xfund.yaml               --out results
python -m macular.runner run fetch_xfund --config configs/xfund_cjk.yaml           --out results

# ---- §4 adaptation (Tables 1, 2, 4, 5; Figs 1, 5) --------------------------
python -m macular.runner run ocr_adapt --config configs/ocr_adapt.yaml                 --out results  # synthetic
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund.yaml           --out results  # ja+es, seeds 0-2
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund_repro.yaml     --out results  # seeds 0,3,4,5 (0 repeats)
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund_det.yaml       --out results  # deterministic kernels
python -m macular.runner run ocr_adapt --config configs/ocr_adapt_xfund_cjk.yaml       --out results  # ja+zh
# ablations / variants: _ep1, _r8, _r32, _dora, _dora_r8, _rslora_r8(_more), _rslora_r16,
# _pissa_r8, _vera, *_lr3e5, _fullft, _qwen25; transfer: ocr_adapt_syn2real_matched, ocr_adapt_real2syn

# ---- §4 gate, model swap (Table 3, Table 5, Fig 2) -------------------------
python scripts/eval_gate_analysis.py
python scripts/gate_reviewer_response.py
python scripts/easyocr_xfund_baseline.py
python scripts/xfund_template_audit.py

# ---- §5 cascade (Fig 3) -----------------------------------------------------
python -m macular.runner run ocr_propagation --config configs/cascade_default_ep80.yaml --out results
python -m macular.runner run ocr_propagation --config configs/cascade_cf_ep80.yaml      --out results

# ---- §6 redaction (Tables 6, 7; Fig 4) -------------------------------------
python -m macular.runner run erasure_comparison --config configs/erasure_comparison.yaml           --out results
python -m macular.runner run erasure_comparison --config configs/erasure_comparison_qwen.yaml      --out results
python -m macular.runner run erasure_comparison --config configs/erasure_comparison_ministral.yaml --out results
python -m macular.runner run erasure_comparison --config configs/erasure_funsd.yaml                --out results
python scripts/leace_transfer_control.py
python scripts/leace_fix_sweep.py
python scripts/leace_multifamily.py
python scripts/strong_attack.py
python scripts/detector_probe_metrics.py

# ---- figures ----------------------------------------------------------------
python scripts/paper_figures.py       # -> paper/figures/*.pdf
```

---

## Experiment index

| Config | Experiment | Result file | Paper |
|---|---|---|---|
| `ocr_adapt.yaml` | synthetic ko/ja/en, family A → C, LoRA r16 | `ocr_adapt.json` | Table 1 |
| `ocr_adapt_xfund.yaml`, `_repro.yaml` | XFUND ja+es default pool (7 runs, seed 0 twice) | `ocr_adapt_xfund*.json` | Table 2, Fig 1 |
| `ocr_adapt_xfund_cjk.yaml`, `_cjk_more.yaml` | XFUND ja+zh pool (7 runs) | `ocr_adapt_xfund_cjk*.json` | Table 2 |
| `ocr_adapt_xfund_det.yaml` | deterministic kernels, seeds 0,0,3,4 | `ocr_adapt_xfund_det.json` | §4.4 |
| `ocr_adapt_xfund_{ep1,r8,r32}.yaml` | epoch / rank ablation | same-named JSON | Table 4, Fig 5 |
| `ocr_adapt_xfund_{dora,dora_r8,rslora_r8,rslora_r8_more,rslora_r16,pissa_r8,vera}.yaml` | PEFT variants at lr 1e-4 | same-named JSON | Table 4 |
| `ocr_adapt_xfund_{r16,r8,dora,pissa_r8,rslora_r16}_lr3e5.yaml` | lr-matched sweep at 3e-5 | same-named JSON | Table 4, §4.6 |
| `ocr_adapt_xfund_fullft.yaml` | full fine-tuning control | `ocr_adapt_xfund_fullft.json` | Table 4 |
| `ocr_adapt_xfund_qwen25.yaml` | second backbone Qwen2.5-VL-3B | `ocr_adapt_xfund_qwen25.json` | §4.7, Table 5 |
| `ocr_adapt_syn2real_matched.yaml`, `ocr_adapt_real2syn.yaml` | cross-corpus transfer | same-named JSON | §4.9 |
| `scripts/gate_reviewer_response.py` | doc-level bootstrap, sibling cases, prospective rule | `gate_reviewer_response.json` | Table 3, Fig 2 |
| `scripts/easyocr_xfund_baseline.py` | EasyOCR on the identical crops | `easyocr_xfund_eval_half.json` | Table 5 |
| `scripts/xfund_template_audit.py` | cross-half template near-duplicates | `xfund_template_audit.json` | §3, §7 |
| `cascade_{default,cf}_ep80.yaml` | OCR error cascade, 3 seeds | `ocr_propagation_*_ep80.json` | §5, Fig 3 |
| `erasure_comparison*.yaml`, `erasure_funsd.yaml` | redaction mechanisms under attack | `erasure_comparison_*.json` | Table 6, Fig 4 |
| `scripts/leace_*.py` | LEACE value-shift, remedies, multi-family fitting | `leace_*.json` | §6.4 |
| `scripts/strong_attack.py` | stronger inverter + probe selectivity | `strong_attack_paddleocr_vl.json` | Table 7, §6.3 |
| `scripts/detector_probe_metrics.py` | detector operating point, balanced accuracy | `detector_probe_metrics.json` | §6.1–6.2 |
| `backbone_gate.yaml`, `ablation.yaml`, `lora_ablation.yaml`, `engine_downstream.yaml` | backbone contribution, core ablations, engine → downstream | same-named JSON | FINDINGS §1, §4 |

---

## Reproducibility notes and known pitfalls

- **Runs are not bit-reproducible on default GPU kernels.** Memory-efficient
  SDPA attention is non-deterministic; an identical seed produced CER 0.099 and
  0.630. Use `deterministic: true` (see `ocr_adapt_xfund_det.yaml`) for
  bit-identical repeats — and expect bad basins to *repeat* rather than vanish.
  Report distributions over seeds, never a single run.
- **Attention kernel moves the zero-shot baseline** by up to ~0.03 CER
  (ja 0.846 → 0.875 under the math kernel). Compare baselines only within one
  kernel configuration.
- **Language-grouped JSONL.** XFUND's converted file is grouped by language; a
  flat head-slice or flat halving silently becomes monolingual.
  `halve_by_language()` and `interleave_by_language()` in `ocr_adapt.py` exist
  for that reason and are applied in every branch of the runner.
- **Crop cap.** `MAX_CROP_SIDE = 512` is applied identically to baseline and
  adapted evaluation; changing it changes every number.
- **Full fine-tuning has no seed-dependent randomness** in this pipeline (no
  adapter init, no dropout in the base model); nominal seeds give identical
  results within one process.
- **Length-weighted CER is fragile to runaway generations**: a handful of
  regions with per-region CER > 1 can dominate a language's score. Gate on
  exact match as well, or clip per-region CER at 1.
- **PaddleOCR-VL as a recogniser** is not run: its remote generation code
  targets transformers 4.x and fails on 5.14 (documented in `ocr_adapt.py`).
  It is used as a frozen feature backbone only.

---

## Retracted intermediate conclusions

`FINDINGS.md` records every conclusion the project stated and later withdrew,
each from single-run reads or a metric that did not measure what it claimed —
among them "real-scan adaptation teaches reading", "it is format control, not
reading", "9/9 cells improve" (8/9), "rsLoRA r8 is the best method" (a
learning-rate artefact), cosine-similarity leakage, and a Frobenius-norm
"transfer quality" predictor that failed on the third backbone. The paper
reports the distribution; a run is an anecdote.

---

## Data archive

Zenodo record **10.5281/zenodo.22078532** contains `results/*.json` (every
adaptation run with per-region `eval_pairs`, the cascade curves, the redaction
attack results, gate analyses, audits), document/region manifests for every
corpus, all configuration files, `FINDINGS.md`, and a README. Images are not
included: synthetic corpora re-render bit-identically from the seeded
generator; FUNSD and XFUND are fetched from their public sources.

---

## Citation

```bibtex
@article{hwang2026macular,
  title   = {MACULAR: Domain-Adapted OCR for Multilingual Medical Documents ---
             Transfer, Reproducibility, and the Limits of Representation-Level Redaction},
  author  = {Hwang, Sunjun and Shin, Huiseung},
  year    = {2026},
  note    = {Code: \url{https://github.com/justinbrianhwang/MACULAR};
             data: \url{https://doi.org/10.5281/zenodo.22078532}}
}
```

## License

- **Code**: MIT — see [`LICENSE`](LICENSE).
- **Data and per-run outputs** (Zenodo archive): CC BY 4.0.
- FUNSD and XFUND remain under their own licenses; this repository redistributes
  neither, only fetch/convert scripts and document-id manifests.
