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

Two independent defects:
- **Training instability.** Single-batch steps on 24 documents produced occasional
  huge gradients that destroyed the adapter in one update. `clip_grad_norm_(1.0)`
  added; collapsed runs are now flagged rather than silently averaged.
- **Metric noise.** The leakage metric was a continuous cosine similarity, which
  drifts with training noise. Two runs of an *identical* config disagreed in sign
  (+0.0224 vs −0.0163). No seed count fixes a metric with this property. This one
  is not fixable by tuning; it is why §4 exists.

### 2.2a After gradient clipping — and a fourth strike against the gate

Re-running the same sweep with clipping (`results/lora_ablation_clipped.json`):
**0 of 9 runs collapse**, against 2 of 9 before, and PII average precision never
falls below 0.833. The fix works.

What it exposes is more interesting than what it fixed. Clinical macro-F1 per
seed:

| Variant | seed 0 | seed 1 | seed 2 | std |
|---|---:|---:|---:|---:|
| full (gate + adversary) | 0.264 | 0.987 | 0.516 | 0.367 |
| no_adversary (gate only) | 0.202 | 0.975 | 0.370 | 0.406 |
| **no_gate (hard mask)** | **0.988** | **0.982** | **0.974** | **0.007** |

Every variant containing the differentiable gate is unstable; the one without it
is not, by a factor of ~50 in standard deviation. The mechanism explains it: the
gate makes `z_safe` a function of the PII head, so early in training — when that
head is still random — the clinical student is fed a randomly corrupted
representation, and on a 24-document budget it sometimes never recovers. That is
a property of coupling the two heads, not a bug.

So the gate costs a third thing beyond utility and privacy: **optimisation
stability**. Leakage numbers from this sweep remain unusable (std 0.013–0.095);
the privacy question belongs to §4.

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

Each attack is run twice: on `z_safe` (the mechanism's own output) and on
`z_ctx_safe` (after the relation graph — the representation actually handed
downstream). LEACE's linear guarantee provably survives the linear projector
(`Cov(W r(x), z) = W Cov(r(x), z) = 0`) but **not** the graph's nonlinearity, so
the post-graph numbers are the deployed risk. Both are reported.

Decision rule fixed in advance:
- `hard_mask` is the **floor**. A mechanism that does not beat it is not worth its
  complexity, regardless of how good its theory is.
- A drop in linear-probe accuracy **without** a drop in MLP-probe accuracy is not
  a privacy result.
- The LEACE residual covariance is checked on **val**, not just on the fitting
  split. A guarantee that holds only where it was fit is not a guarantee.

### 4.1 Results (frozen PaddleOCR-VL, counterfactual data, 3 seeds, n_pii = 863)

Attacks on `z_safe` (the mechanism's own output):

| Mechanism | clinical F1 | linear probe | **nonlinear probe** | inversion EM | inversion CER |
|---|---:|---:|---:|---:|---:|
| none (unprotected) | 0.976 | 0.967 | 0.982 | 0.094 | 0.389 |
| **hard_mask** | 0.947 | 0.853 | 0.966 | **0.014** | **0.510** |
| gate | 0.917 | 0.850 | 0.976 | 0.025 | 0.490 |
| leace | 0.942 | 0.916 | 0.942 | 0.085 | 0.456 |

Same attacks after the relation graph — **the representation actually handed
downstream**:

| Mechanism | ctx linear | ctx nonlinear | ctx EM | ctx CER |
|---|---:|---:|---:|---:|
| none | 0.958 | 0.975 | 0.046 | 0.454 |
| **hard_mask** | **0.849** | **0.843** | **0.002** | **1.080** |
| gate | 0.919 | 0.937 | 0.007 | 0.705 |
| leace | 0.871 | 0.917 | 0.049 | 0.549 |

Majority baseline 0.847. Seed-to-seed std: 0.002–0.023 — **smaller than every
difference in the tables.** The discrete metrics fixed the reproducibility
problem that made §2.2 unusable.

### 4.1b All three backbones (same data, same protocol, 3 seeds each)

Post-relation-graph numbers — the deployed surface. Majority baseline 0.847.

| Backbone | Mechanism | clinical F1 | ctx linear | ctx nonlinear | ctx CER |
|---|---|---:|---:|---:|---:|
| PaddleOCR-VL 0.9B | none | 0.976 | 0.958 | 0.975 | 0.454 |
| | hard_mask | 0.947 | 0.849 | **0.843** | **1.080** |
| | gate | 0.917 | 0.919 | 0.937 | 0.705 |
| | leace | 0.942 | 0.871 | 0.917 | 0.549 |
| Qwen2-VL-2B | none | 0.899 | 0.956 | 0.968 | 0.549 |
| | hard_mask | 0.778 | 0.848 | **0.859** | **0.949** |
| | gate | 0.826 | 0.923 | 0.929 | 0.743 |
| | leace | **0.890** | 0.846 | 0.860 | 0.844 |
| Ministral-3 | none | 0.944 | 0.953 | 0.975 | 0.493 |
| | hard_mask | 0.731 | 0.876 | **0.879** | **0.915** |
| | gate | 0.779 | 0.946 | 0.959 | 0.581 |
| | leace | **0.897** | 0.877 | 0.896 | 0.623 |

### 4.1c Value reconstruction, all four datasets on identical metrics

Inversion exact-match, best of the two decoders, minus the shuffled-control prior
floor (which came out 0.000 everywhere, so the columns are equal).

| Mechanism | PaddleOCR-VL | Qwen2-VL-2B | Ministral-3 | FUNSD (real scans) |
|---|---:|---:|---:|---:|
| none | 0.157 | 0.013 | 0.083 | 0.109 |
| **hard_mask** | **0.013** | **0.002** | **0.020** | **0.008** |
| gate | 0.022 | 0.015 | 0.023 | 0.011 |
| leace | 0.144 | 0.032 | 0.058 | 0.101 |

Structural masking has the lowest recovery on **all four**. LEACE leaves the
literal value nearly as recoverable as no protection at all on three of four
(0.144 vs 0.157; 0.058 vs 0.083; 0.101 vs 0.109) — including the real scans.

### 4.2 What this says

**1. The differentiable gate is dominated on both backbones.** On PaddleOCR-VL
it has lower utility than hard masking (0.917 vs 0.947) *and* more leakage on
every measure. On Qwen2-VL-2B it is beaten by LEACE on both axes at once (utility
0.826 vs 0.890, post-graph nonlinear leakage 0.929 vs 0.860). There is no
backbone and no operating point at which the gate is the right choice. Proposal
gate #4 fails, and the measurement is now stable enough to say so.

**2. Structural masking is never worse on privacy; erasure can be far cheaper in
utility — by a backbone-dependent amount.** Across all three backbones,
hard masking's post-graph nonlinear leakage is ≤ LEACE's (0.843 vs 0.917;
0.859 vs 0.860; 0.879 vs 0.896) and its inversion CER is the highest (1.080 /
0.949 / 0.915). But its utility cost swings enormously by backbone, and LEACE's
does not:

| Backbone | hard_mask F1 | leace F1 | utility gap |
|---|---:|---:|---:|
| PaddleOCR-VL | 0.947 | 0.942 | −0.005 |
| Qwen2-VL-2B | 0.778 | 0.890 | **+0.112** |
| Ministral-3 | 0.731 | 0.897 | **+0.166** |

So the honest statement is a trade-off, not a winner. Structural masking is
strictly safer against reconstruction; closed-form erasure buys back large
amounts of clinical utility on two of three backbones for a small increase in
probe leakage (+0.001 to +0.074) and a much larger loss in value protection.

**2a. A predictor I proposed does not survive.** After two backbones it looked
like the ranking tracked erasure transfer quality, measured by the residual
cross-covariance on validation (0.771 for PaddleOCR-VL where hard masking won,
0.480 for Qwen where LEACE won). Ministral-3 breaks it: the *lowest* residual
covariance of the three (0.463) with the *worst* cross-family linear probe
(0.944 vs 0.932 and 0.889).

The reason is a measurement error worth recording: **the residual covariance is a
Frobenius norm on unnormalised features, so it is not comparable across
backbones** — only within one. The probe accuracy is the comparable quantity.
Reported as a cross-backbone predictor it would have been wrong, and two data
points were not enough to notice.

**3. LEACE's guarantee does not transfer on any backbone.** Fitted on family A
and applied to family B, the linear probe on the erased validation features —
before the projector, before any training — stays far above the 0.847 majority
baseline on all three: **0.932** (PaddleOCR-VL), **0.889** (Qwen2-VL-2B),
**0.944** (Ministral-3), against raw values of 0.964 / 0.975 / 0.967. The
residual cross-covariance is ~1e-07 where the eraser is fitted and 0.46–0.77 on
validation. The theorem holds exactly where it is fitted (§3a) and nowhere else.

The reason is a property of our protocol, not an accident: **train and val use
disjoint PII generator families by design** (family A vs B, with disjoint name,
address, organisation, phone-prefix and ID pools). The concept subspace carrying
family A's PII is not the one carrying family B's. This is exactly the
deployment condition — new patients, unseen name distributions — so a guarantee
fit on the training distribution is vacuous where it matters. `scripts/
leace_transfer_control.py` fits an eraser on validation itself to confirm the
implementation is sound and the failure is transfer, not a bug.

**3a. The transfer failure is real, not an implementation bug** (`results/
leace_transfer_control.json`). Fitting the eraser on validation and probing
validation drives the linear probe to **exactly** the majority baseline —
0.847 vs 0.847, residual covariance 1.3e-07. The implementation does what the
theorem says. Side by side on the same validation regions:

| Eraser | linear probe | nonlinear probe |
|---|---:|---:|
| none (raw) | 0.964 | 0.974 |
| fit on family A, applied to B | 0.932 | 0.958 |
| fit on family B, applied to B | **0.847** (= majority) | 0.912 |

Two limits, both quantified: LEACE is **exactly right in-distribution and only
there** (+0.085 linear leakage once the PII values change), and even in the best
case a **nonlinear attacker keeps +0.065 over chance** — the caveat the erasure
literature states, now measured on document VLM region features.

**3b. Cheap remedies do not rescue it** (`results/leace_fix_sweep.json`).
Coarsening the concept to binary is-PII (0.956), heavier covariance shrinkage
(0.928 at eps=0.1), and truncating the concept subspace to its top 1/2/4
directions (0.956/0.956/0.952) are all no better than the 0.932 baseline, and
most are worse. The failure is not a hyperparameter.

**3c. Nor does fitting across several PII families** (`results/
leace_multifamily.json`). Two additional train-side value families (D, E, pools
disjoint from val and test) were generated so the eraser could be fitted on
several value distributions at once:

| Eraser fitted on | regions | linear probe | residual cov on val |
|---|---:|---:|---:|
| A | 5,738 | 0.932 | 0.771 |
| A + D | 11,458 | 0.927 | 0.700 |
| A + D + E | 17,176 | 0.925 | 0.663 |
| *(control: fitted on val itself)* | — | *0.847 = majority* | *1.3e-07* |

Tripling the fitting data and the family count closes 0.007 of the 0.085 gap.
The trend is in the right direction and far too shallow to matter: reaching the
in-distribution floor this way would take an implausible number of families. So
the transfer failure is **intrinsic**, not a data-budget problem — which makes
this a complete negative result rather than an open question.

**4. Measuring only the mechanism's output would have been wrong.** Pre-graph,
hard masking looks partial (nonlinear probe 0.966). Post-graph it is complete
(0.843). The relation graph does not restore what was structurally removed —
but it does leave the gate's and LEACE's soft residue readable. Any privacy
number reported on `z_safe` alone describes a representation nobody deploys.

### 4.3 Consequence for the paper

Four claims hold on **all three** backbones, with seed-to-seed std well below
every difference:

1. **Learned differentiable redaction (gate + GRL adversary) is dominated
   everywhere.** Not "did not help in our setup" — on every backbone some simpler
   mechanism beats it on both axes at once. On Ministral-3 the gate is the worst
   protected mechanism measured (ctx nonlinear 0.959 against an unprotected
   0.975) while also costing utility. It additionally destabilises training
   (§2.2a): clinical-F1 std 0.37–0.41 with the gate versus 0.007 without it.
2. **Closed-form linear erasure does not transfer to unseen PII values on any
   backbone** (cross-family linear probe 0.889–0.944 against a 0.847 majority),
   and neither cheaper concepts, shrinkage, rank truncation, nor fitting across
   three value families repairs it.
3. **No mechanism defeats a nonlinear attacker at the mechanism's own output.**
   Contextual mixing then pushes hard masking and LEACE near chance while leaving
   the gate readable.
4. **Structural masking is never worse on privacy, and its utility cost is what
   varies** — from −0.005 to +0.166 clinical F1 against LEACE depending on the
   backbone. Choosing between them is a per-backbone empirical question, and we
   could not find a cross-backbone quantity that predicts it (§4.2a).

The transferable contribution is the evaluation protocol that produced all of
this: discrete attacks (probe accuracy + inversion exact-match/CER), applied both
at the mechanism and after contextualisation, under **held-out PII value
families**. Without held-out families the erasure transfer failure is invisible;
without the post-graph surface, hard masking looks partial when it is complete;
without discrete metrics the whole comparison is noise (§2.2). All four
literature reviews report this protocol does not exist for document/VLM region
features.

**Caveats to carry:** three backbones, 120 documents per split, and frozen
features throughout. §5 adds a real-scan arm.

---

## 4b. Domain adaptation actually improves CJK OCR

The one experiment here that *improves* OCR rather than measuring it. LoRA
fine-tune Qwen2-VL-2B on medical-form region crops (family A), evaluate on
family C — disjoint name, address, organisation, phone-prefix and ID pools, so a
gain cannot be memorisation of the values. 1,440 training regions, 960 evaluation
regions, 29M LoRA parameters, 2 epochs.

| Language | CER before → after | exact match before → after |
|---|---|---|
| **ko** | 0.083 → **0.041** (−50%) | 0.886 → **0.950** |
| **ja** | 0.091 → **0.063** (−31%) | 0.871 → **0.905** |
| en | 0.036 → 0.049 (+36%) | 0.935 → 0.929 |
| macro | 0.067 → **0.050** (−25%) | — |

**CER and exact match move together**, which matters: had only CER improved, the
gain would have been the model learning the output *format* (the unadapted model
occasionally answers with grounding coordinates instead of text) rather than
reading better.

**The gain is CJK-specific and English regresses slightly.** That shape is the
result, not a defect — English was already near ceiling (CER 0.036) and the
adaptation trades a little of it for a large CJK gain. A paper claiming a
multilingual medical-document contribution should report the regression, not
average it away.

**Caveats:** single seed, synthetic rendered text rather than real scans, and
Qwen2-VL-2B rather than PaddleOCR-VL (whose remote modeling code does not load
for generation under transformers 5.x). Validating the CJK gain on real scans
needs XFUND (ja/zh), which FUNSD cannot provide.

---

## 5. Real scanned pages (FUNSD)

Everything above is synthetic, because no public scanned corpus carries PII
annotations — a gap all four literature reviews confirmed independently. FUNSD is
real scanned paper with real scanner noise and handwriting. Its annotation scheme
carries the distinction the protocol needs, so `answer` regions (filled-in values:
names, phone numbers, dates) become the sensitive class and `question`/`header`
(the printed form structure) the utility class. PaddleOCR-VL, 3 seeds, 606
sensitive regions, majority baseline 0.652.

| Mechanism | utility F1 | ctx linear | ctx nonlinear | inversion EM |
|---|---:|---:|---:|---:|
| none | 0.680 | 0.885 | 0.896 | 0.109 |
| **hard_mask** | 0.674 | 0.873 | 0.866 | **0.008** |
| gate | 0.696 | 0.892 | 0.894 | 0.011 |
| leace | 0.643 | 0.846 | 0.869 | 0.101 |

The prior floor is 0.000 for both decoders here — real form values are diverse
enough that nothing is guessable from the text distribution alone — so every
recovery number is fully attributable to the representation.

**5.1 On real scans, LEACE does not protect the value at all.** Inversion
exact-match is 0.101 against an unprotected 0.109; structural masking cuts it to
0.008 and the gate to 0.011. An attacker reads back roughly the same fraction of
real names and phone numbers from the LEACE-erased representation as from the
unprotected one. This is the sharpest version of the §4 result and it is on real
data: **erasing the linear concept direction does not remove the content.**

**5.2 Nothing protects the role information on real scans.** Every mechanism sits
at 0.866–0.894 nonlinear-probe accuracy against an unprotected 0.896 and a
majority of 0.652. On synthetic pages contextual mixing pushed hard masking to
chance; on real scans it does not, and the difference is worth explaining rather
than hiding: FUNSD regions are far more heterogeneous, so role is predictable
from geometry and neighbourhood alone.

**5.3 What this arm does and does not support.** It supports the mechanism
ranking on the reconstruction axis, on real scans, with real noise. It cannot
speak to the erasure-transfer result: FUNSD is one document population with no
disjoint value families, so §4's transfer failure remains a synthetic-data
finding. The two must not be blurred in the paper.
