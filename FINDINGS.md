# MACULAR — measured findings

Running record of what the experiments actually showed, including the parts that
contradict the proposal. Numbers here are the ones to quote; anything not in this
file has either been superseded or was never reproduced.

Rule used throughout: **a single-seed number is not a result.** Several claims in
this project's history were retracted for exactly that reason, and they are kept
below rather than deleted, because knowing which measurements were unstable is
part of the finding.

---

## 1. What holds up

### 1.1 VLM backbone features contribute (RQ6, gate #3) — PASSED

Replacing hand-crafted region features with real VLM ROI-pooled features improves
PII detection average precision across every backbone tested (3 seeds each,
criterion: delta must exceed the max per-seed std).

| Backbone | ΔAP vs text-only features |
|---|---:|
| Ministral-3 | **+0.377** |
| Qwen3-VL-8B | +0.152 |
| Qwen2-VL-2B | + |
| Llama-3.2-Vision | + |
| PaddleOCR-VL-0.9B | + |

Two false negatives were found and fixed while establishing this, both of which
would have killed the project:
- Unnormalized features gave Δ −0.127 ("the backbone is useless") until per-dim
  standardization with train-fitted statistics was applied → +0.036.
- Forcing a square patch grid gave Ministral Δ −0.243 until the grid was derived
  from image size and `patch_size` (100×72 = 7200 patches) → **+0.377**.

Also: F1 disagreed with AP in sign (Qwen-8B ΔF1 −0.067 vs ΔAP +0.152) because
runs land on different operating points. AP/AUC are the reportable metrics here.

### 1.2 OCR error cascade is real, but only on shortcut-controlled data

| CER | default layout F1 | counterfactual F1 |
|---|---:|---:|
| 0.00 | 0.941 | **0.648** |
| 0.50 | 0.941 | **0.530** |
| drop | **0.000 (flat)** | −0.117 |

The flat left column is the important part: on default layouts the model answers
from geometry and never reads the text, so OCR quality cannot matter. Any OCR
sensitivity claim must be made on counterfactual data.

### 1.3 Shortcut control works

A coordinate-only audit classifier (geometry, no text) reaches F1 0.432 with
precision ≈ base rate on counterfactual layouts, versus ~1.00 on default layouts.
Confirmed independently at 500 documents on a second machine (0.464 / 0.302).

### 1.4 Real scanned data runs end to end

FUNSD: 149 train / 50 test documents, 9,529 regions converted and processed.
(License note: FUNSD's terms are research/non-commercial — fine for publication,
not for shipping in a permissively-licensed core.)

---

## 2. What failed

### 2.1 The differentiable redaction gate — no advantage over a hard mask

Frozen backbone, Ministral, counterfactual data, 3 seeds:

| Variant | identity leakage | clinical macro-F1 |
|---|---:|---:|
| full | +0.0103 | 0.779 |
| hard_mask | +0.0044 | 0.731 |
| no_consistency | +0.0095 | **0.953** |
| no_adversary | +0.0109 | 0.782 |
| no_gate | +0.0042 | 0.933 |
| *(raw, unprotected)* | *+0.0091* | — |

The gate + consistency + adversary cost a large amount of clinical utility and
bought no measurable privacy. Note also that leakage is tiny everywhere (~0.01):
there was little identity information to leak in the first place.

### 2.2 The LoRA follow-up — measured nothing at all

Hypothesis for 2.1: a frozen tower can only be reweighted, not reshaped, so the
gate had no way to strip content. Making the vision tower trainable (LoRA, 3.85M
params) was supposed to give it that ability.

3 seeds × 3 variants:

| Variant | leak reduction (mean ± std) | clinical F1 | collapsed runs |
|---|---:|---:|---|
| full | −0.0020 ± 0.0066 | 0.61 ± 0.54 | seed 0 |
| no_gate | −0.0458 ± 0.0457 | 0.64 ± 0.56 | seed 1 |
| no_adversary | −0.0261 ± 0.0617 | 0.52 ± 0.32 | — |

**std > mean in every row, and 2 of 9 runs collapsed outright** (PII AP 0.15 —
below base rate — and clinical macro-F1 exactly 0.0, while other seeds of the
same config reached 0.99).

Two independent defects, both since fixed:
- **Training instability.** Single-batch steps on 24 documents produced occasional
  huge gradients that destroyed the adapter in one update. `clip_grad_norm_(1.0)`
  added; collapsed runs are now flagged rather than silently averaged.
- **Metric noise.** The leakage metric was a continuous cosine similarity, which
  drifts with training noise. Two runs of an *identical* config disagreed in sign
  (+0.0224 vs −0.0163). No seed count fixes a metric with this property.

### 2.3 Claims retracted during this project

Kept deliberately — each was stated confidently and was wrong.

| Claim made | Why it was wrong |
|---|---|
| "val F1 0.94 shows genuine generalization" | The model answered from geometry; only counterfactual data tests reading. |
| "The gate provides zero protection" (leakage 0.733 safe vs 0.731 raw) | The metric measured PII *type*, which the gate preserves on purpose. Type-mean alone scored 0.737 — higher than both. Replaced by `identity_leakage = cosine − type baseline`. |
| "LoRA cuts leakage 74%" | Attributed to the gate, but hard masking achieved the same reduction. Not a gate effect. |
| "The GRL adversary costs ≈0.08 clinical macro-F1" | Single seed. With 3 seeds the variants are not separable. |
| "The gate is what costs utility" (frozen) | `full` vs `no_adversary` differ only by the adversary; on that single-seed read the adversary was the cost. Neither attribution survived seeding. |

---

## 3. Diagnosis, from the literature

Four independent deep-research passes (ChatGPT, Gemini, Kimi, Mistral) converged
on the same reading, and it matches our numbers:

1. **Adversarial/GRL attribute removal is a canonical failure mode.** It hides an
   attribute from the training-time adversary without removing it; a fresh
   post-hoc probe recovers it (Elazar & Goldberg, EMNLP 2018; Gonen & Goldberg,
   NAACL 2019, "Lipstick on a Pig"). Counter-nuance to cite for balance: Barrett
   et al., EMNLP-IJCNLP 2019.
2. **The successor with an actual guarantee is closed-form linear concept
   erasure**: INLP (ACL 2020) → RLACE (ICML 2022) → **LEACE** (NeurIPS 2023) →
   QLEACE/SPLINCE (2025). LEACE is a single affine map — no training loop — and
   provably prevents *all linear classifiers* from recovering the concept.
3. **Cosine similarity is not an accepted attack.** The standard is embedding
   inversion: Vec2Text (EMNLP 2023; 92% exact match on 32-token text, demonstrated
   on clinical notes), GEIA (Findings ACL 2023), and — closest to our threat model,
   since the attacker holds *vision* features — CapRecover (ACM MM 2025).
   Membership inference should be reported as TPR@low-FPR (LiRA, S&P 2022).
4. **No published application of LEACE-class erasure to per-region document/VLM
   features.** All four searches failed to find one. That absence is the opening.
5. **No public Korean or Japanese scanned medical form corpus exists.** Closest
   available: CHIP2022 MedOCR (Chinese, real de-identified invoices and discharge
   summaries, CC-BY-NC-SA → excluded from core), a 238-image Chinese lab-report
   set (license unstated), and SynthDoG (synthetic, en/zh/ja/ko, MIT).

**The trap to avoid:** linear erasure guards *linear* readouts only. Published
stress tests recover much of an "erased" concept with a nonlinear probe. Swapping
in LEACE and measuring with a linear probe would repeat the gate's mistake at a
more sophisticated level — the guarantee restated as if it were evidence.

---

## 4. Current experiment

`macular run erasure_comparison` — four mechanisms (`none`, `hard_mask`, `gate`,
`leace`) on identical frozen features, attacked three ways: fresh linear probe,
fresh nonlinear MLP probe, and an inversion attack scored by exact-match and CER.
Discrete metrics throughout, 3 seeds.

Decision rule fixed in advance:
- `hard_mask` is the **floor**. A mechanism that does not beat it is not worth its
  complexity, regardless of how good its theory is.
- A drop in linear-probe accuracy **without** a drop in MLP-probe accuracy is not
  a privacy result.

Results: see §5 (appended when the run completes).
