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

**3-seed replication (2026-08-24, reviewer response; regenerated corpora
`data/meddoc_default` / `data/meddoc_cf`, 400 train docs, 80 epochs;
`configs/cascade_*_ep80.yaml`):** default 0.948/0.949/0.948 → 0.942/0.944/0.944
(drop ≤0.007, flat in 3/3); counterfactual 0.639/0.645/0.673 →
0.548/0.570/0.579 (drop 0.075–0.094, declining in 3/3). Level and shape of
the original single run reproduce. A 40-epoch / 120-doc variant underfits the
counterfactual task (F1 0.14–0.21) but still shows the flat-vs-declining
contrast. The paper's Fig. 3 now draws the per-seed curves.

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
| "Adaptation makes the model read better, because CER and exact match move together" — then its retraction, "adaptation is format control, not reading" | Both were single-run reads of a heavy-tailed distribution. Over seeds: synthetic improves both metrics in every language; real-scan ja improves EM massively in 6 of 7 runs (the two runs behind the format-control claim were tail draws); real-scan es genuinely does trade EM for CER. Neither slogan was right. §4b.1–4b.2. |
| "XFUND adaptation cuts CER by 0.98" | The split was language-grouped, so it trained on ja and evaluated on es. A cross-lingual transfer result mislabelled as domain adaptation, with the ja row missing entirely. §4b.4. |
| "Synthetic English regresses on both metrics under adaptation" | One seed. The 3-seed mean improves English on both metrics (ΔCER −0.029±0.013, ΔEM +0.065±0.009). §4b.1. |
| "ja's outcome depends on its co-training partner (−0.569 with es vs −0.437 with zh)" | Stated from one run per partner, with the *direction* backwards: at 7 draws per side, es co-training beats zh co-training on the identical ja eval half (EM p = 0.026, CER p = 0.053; §4b.2). The single-run numbers that prompted the claim were both tail draws. |

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

## 4b. Domain adaptation helps — but individual runs are not reproducible

The one experiment here that *improves* OCR rather than measuring it. LoRA
fine-tune Qwen2-VL-2B (29M LoRA params, 2 epochs) on medical-form region crops,
evaluate the same crops before and after in one paired run. Everything below is
3+ seeds per config; the single-seed §4b as originally written made two claims
that did not survive seeding (see the retraction table, §2.3).

All numbers are measured with `MAX_CROP_SIDE = 512`. That cap is load bearing
and was added late — see §4b.4 — so the synthetic run was repeated under it. The
pre-cap synthetic numbers are kept in `results/ocr_adapt_uncapped_crops.json`
and are **not** comparable to anything else here.

### 4b.1 Synthetic rendered forms (train family A → eval family C, 3 seeds)

Disjoint name, address, organisation, phone-prefix and ID pools, so a gain
cannot be memorisation of the values. 1,440 train / 960 eval regions.

| Language | CER before | after (per seed) | ΔCER mean±std | ΔEM mean±std |
|---|---|---|---|---|
| ko | 0.121 | 0.016 / 0.017 / 0.071 | **−0.086**±0.026 | **+0.084**±0.009 |
| ja | 0.104 | 0.015 / 0.042 / 0.073 | **−0.061**±0.024 | **+0.090**±0.028 |
| en | 0.047 | 0.006 / 0.013 / 0.036 | **−0.029**±0.013 | **+0.065**±0.009 |
| macro | 0.088 | — | −0.057±0.020 | — |

**All three languages improve on both CER and exact match, every seed.** The
single-seed version of this table said English regresses on both metrics; that
was one draw. The CJK gains are still the largest, consistent with the
adaptation mattering most where the baseline is weakest.

### 4b.2 Real scanned forms (XFUND, language-stratified halves of one split)

Weaker held-out than §4b.1: XFUND ships a single split with no generated PII, so
train and eval are halves of one document population. This answers "does
adaptation help on real scans", not "does it generalise to unseen values".

Pooling every training run of the ja+es config under the current code (two
invocations, 7 trained adapters — including one repeated seed, which is the
point of §4b.3), against a baseline of ja CER 0.846 / EM 0.400 and es CER 0.924
/ EM 0.520:

| Language | after CER, per trained adapter | after EM, per trained adapter |
|---|---|---|
| ja | 0.099 0.099 0.104 0.114 0.139 0.147 **0.630** | 0.824 0.798 0.789 0.759 0.751 0.729 **0.100** |
| es | 0.132 0.170 0.227 0.299 0.329 0.375 0.399 | 0.538 0.358 0.293 0.260 0.237 0.233 0.190 |

And the ja+zh config (7 trained adapters across two invocations), baseline zh
CER 0.481 / EM 0.660:

| Language | after CER, per trained adapter | after EM, per trained adapter |
|---|---|---|
| ja | 0.129 0.155 0.171 0.198 0.200 0.243 0.332 | 0.666 0.575 0.666 0.580 0.664 0.612 0.452 |
| zh | 0.080 0.109 0.113 0.125 0.185 0.215 0.254 | 0.857 0.835 0.807 0.790 0.790 0.803 0.643 |

What survives across every run:

- **CER improves in all 28 trained-adapter × language outcomes**, usually by a
  lot (ja median 0.846 → 0.114). Even the worst outlier (0.630) beats baseline.
- **ja exact match usually improves massively** (0.400 → 0.73–0.82 in 6 of 7
  runs) — the earlier claim that real-scan adaptation is "format control, not
  reading" was based on two single runs that are now recognisable as tail draws.
  But the seventh run collapsed EM to 0.100, *below* baseline.
- **es exact match usually degrades** (0.520 → 0.19–0.36 in 6 of 7 runs). For
  Spanish the format-control story still fits: CER down, exact transcription
  worse. The per-language direction of the EM effect is real and opposite.

**What does not survive: any per-run number.** See §4b.3.

**Partner-language effect: real on exact match, borderline on CER (7 vs 7).**
The ja eval half is identical in both configs, so the co-training language is
the only difference. Four more ja+zh seeds brought both sides to 7 draws. Exact
two-sided Mann-Whitney: EM p = 0.026 (median 0.759 beside es vs 0.612 beside
zh), CER p = 0.053 (median 0.114 vs 0.198) — the CER test just misses because
the es side contains the 0.630 diverged run, which cannot be excluded post hoc.
Direction is the counterintuitive one: Japanese is helped *more* by Spanish
co-training than by Chinese. No mechanism is claimed; one mundane candidate is
that same-script zh crops compete for the shared CJK reading capacity while es
does not.

### 4b.3 Identical seed, identical code, different result

The reproducibility control: re-run the ja+es config with seeds [0, 3, 4, 5],
where **seed 0 exactly repeats the previous invocation** — same code, same data,
same machine, same seed.

Seed 0 produced ja after-CER **0.099** in the first invocation and **0.630** in
the repeat. Its three sibling seeds in the same repeat run landed at 0.114–0.147,
so the repeat is an outlier *within its own run* — while sharing every
controllable input with a run that was not. Train loss does not flag it: the
divergent run's loss history (0.218 → 0.114) is inside the range of the normal
runs (0.208–0.218 → 0.099–0.118). The failure is invisible until evaluation.

The mechanism consistent with everything observed: bf16 GPU non-determinism
compounds over ~2,400 training steps, and the outcome distribution is
heavy-tailed — most runs land in a good basin, an occasional run diverges
badly. The same non-determinism showed up in the LoRA privacy ablation (§2.2)
as sign flips on identical configs; here it produces a 6× CER spread on an
identical seed.

Consequences:

- A seed is not a reproducibility mechanism for this training setup. Nominal
  seeding does not pin the outcome, so "seed N gave X" is not a reportable fact.
- The reporting unit must be the across-run distribution (median + range +
  outlier rate), not a mean±std over seeds that pretends draws are labelled.
- Any deployment of this adaptation needs a post-training eval gate, because
  train loss cannot tell a diverged adapter from a good one.

### 4b.3a The determinism control: cause confirmed, failure not removed

Reviewer-requested control: `deterministic: true` (CUBLAS workspace pinned,
`torch.use_deterministic_algorithms`, memory-efficient/flash SDPA disabled so
attention uses the math kernel — smoke-tested to zero non-deterministic-op
warnings). ja+es config, seeds [0, 0, 3, 4].

| seed | ja CER / EM | es CER / EM | loss history |
|---|---|---|---|
| 0 | 0.0799 / 0.7876 | 0.1271 / 0.6267 | 0.2269 → 0.0966 |
| 0 (repeat) | **0.0799 / 0.7876** | **0.1271 / 0.6267** | **identical** |
| 3 | 0.1809 / **0.2742** | 0.1603 / 0.5167 | 0.2412 → 0.0894 |
| 4 | 0.0922 / 0.7642 | 0.1205 / 0.5950 | 0.2334 → 0.0798 |

Two conclusions, one expected and one not.

**Cause confirmed.** The seed-0 pair is bit-identical: 1198/1198 predictions
and the full loss trajectory. Under deterministic kernels an identical seed
gives an identical adapter, so the 0.099-vs-0.630 divergence of §4b.3 was
kernel non-determinism and nothing else — not data ordering, not uncontrolled
RNG state. The hypothesis is promoted to a finding.

**Failure not removed.** Seed 3 lands in a bad basin *reproducibly*: ja EM
collapses to 0.274 (siblings 0.76–0.79) with a loss history inside the normal
range. Determinism makes a bad draw repeatable; it does not make it rarer.
The divergence is a property of the training dynamics (seed-sensitive
basins); kernel non-determinism only decouples it from the nominal seed. For
deployment this strengthens the gate recommendation: a deterministic stack
does not substitute for post-training evaluation.

**Side finding on baselines.** Forcing math attention moved the *base model's*
zero-shot scores (ja 0.846 → 0.875, es 0.924 → 0.906): junk-prone greedy
generation is sensitive to which attention kernel runs. Across invocations
of the same kernel the drift is ~±0.005; across kernels it reaches ~0.03.
Baselines are therefore comparable only within one kernel configuration —
all other tables in this file use the default (memory-efficient) kernel.

### 4b.4 Two measurement faults found and fixed here

**Language-grouped split.** `test.jsonl` is written language-grouped, so the
XFUND fallback's flat `docs[:half], docs[half:]` trained on all-ja and evaluated
on all-es: a cross-lingual transfer experiment that reported no ja row at all,
and looked like a clean −0.98 CER win. Fixed by `halve_by_language()`
(alternates within each language); regression test in `tests/test_ocr_adapt.py`.
The same failure family hit a second code path: the cross-corpus route (§4b.7)
head-slices `docs[:max_docs]`, so its first syn→real run evaluated on 50 ja
documents and zero es. Fixed by `interleave_by_language()` on both train and
eval in every branch.

**Unbounded crops.** Real scans are far larger than rendered pages and `_crop`
upscales 3× on top, so Qwen2-VL's dynamic resolution produced thousands of
visual tokens per crop — an XFUND run sat at 100% GPU for 18 hours on a step
count the synthetic run finished quickly. `MAX_CROP_SIDE = 512` fixed the
throughput (0.70 s/region). But **all 960 synthetic crops exceeded 512** (max
2,544 px), so the cap changed the synthetic measurement too: baseline CER rose
(ko 0.083 → 0.121, ja 0.091 → 0.104, en 0.036 → 0.047) because downsampling
costs real detail. The adaptation *effect* survived (macro ΔCER −0.017 both
ways), the absolute levels did not.

**Caveats:** Qwen2-VL-2B rather than PaddleOCR-VL (whose remote modeling code
does not load for generation under transformers 5.x) for the main runs — but
see §4b.6 for a second backbone; no Korean real scans exist publicly, so ko is
synthetic-only; XFUND train/eval halves are one document population, so the
real-scan numbers carry no unseen-value guarantee.

### 4b.5 Less is more: epochs and rank (XFUND ja+es, 3 seeds each)

Motivated by two loose ends: train loss *rose* in epoch 2 of every synthetic
seed, and r=16 was never justified. Defaults for comparison: 2 epochs, r=16
(after-CER ja 0.099–0.630, es 0.132–0.399; es EM 0.19–0.36 in 6/7 runs).

| Variant | ja after CER | ja after EM | es after CER | es after EM |
|---|---|---|---|---|
| **1 epoch** (r16) | 0.067 0.083 0.172 | 0.537–0.804 | 0.136 0.154 0.161 | **0.407–0.638** |
| **r=8** (2 ep) | 0.061 0.080 0.139 | 0.651–0.806 | **0.077–0.154** | **0.468–0.667** |
| r=16, 2 ep (default) | 0.099–0.147, one 0.630 | 0.729–0.824, one 0.100 | 0.132–0.399 | 0.190–0.538 |
| **r=32** (2 ep) | 0.086 0.203 **0.335** | 0.271–0.784 | 0.165 0.378 **0.558** | 0.098–0.577 |

α/r held at 2 throughout, so rank varies capacity only. The pattern is
monotone: **shrinking the training budget (1 epoch) or capacity (r=8) keeps the
CER gain, softens the es exact-match degradation, and tightens the spread;
growing capacity (r=32) makes everything worse and wilder** — r=32's worst es
run lands at EM 0.098, far below the 0.520 baseline. The paper should default
to r=8 and 1–2 epochs and present r=16/2ep as the ablation, not the other way
around. (3 seeds per variant: the spread ordering is consistent across four
variants, but tail *rates* per variant are not estimable at n=3.)

**DoRA (co-author suggestion, 3 seeds each).** Weight-decomposed LoRA at the
same ranks, same everything else (peft `use_dora=True`; its float32 magnitude
vectors must be cast to the base dtype or a bf16 forward crashes):

| Variant | ja after CER | ja after EM | es after CER | es after EM |
|---|---|---|---|---|
| DoRA r=16 | 0.110 0.146 0.151 | 0.635–0.764 | 0.128–0.184 | **0.550–0.663** |
| DoRA r=8 | 0.098 0.177 **0.448** | 0.266–0.749 | 0.148–0.202 | 0.323–0.568 |

Two findings. **DoRA at r=16 genuinely fixes the es exact-match collapse**
(baseline 0.520 → 0.55–0.66, above baseline in 3/3 seeds, where LoRA r=16
degraded it in 6/7 runs; exact 3-vs-7 Mann–Whitney p = 0.017) with a tighter
spread and no diverged draw. So the co-author's intuition was right at the
default rank — magnitude/direction decoupling helps. **But it does not stack
with the rank fix**: DoRA r=8 is worse than LoRA r=8 everywhere and contains a
bad tail draw (ja 0.448 / EM 0.266). The two remedies are alternative routes to
the same place, and the simpler one (halve the rank) reaches it with the best
ja CER of any cell and half the adapter parameters.

**The wider PEFT-variant sweep (rsLoRA, PiSSA, VeRA; 3 seeds each, same
protocol).** Each variant tested a specific hypothesis about the failure modes
above:

| Variant | trainable | ja after CER | ja after EM | es after CER | es after EM |
|---|---:|---|---|---|---|
| **rsLoRA r=8** (3 seeds) | 14.5M | **0.056 0.069 0.086** | **0.786–0.821** | 0.071 0.083 0.179 | **0.545–0.705** |
| rsLoRA r=8 (7 seeds, pooled) | 14.5M | 0.056–0.214, med 0.093 | 0.570–0.821 | 0.071–0.275, med 0.153 | 0.487–0.705 |
| rsLoRA r=16 | 29.0M | 0.161 0.345 **1.018** | **0.000**–0.704 | 0.286 0.299 **1.697** | **0.000**–0.522 |
| rsLoRA r=16, lr 3e-5 | 29.0M | 0.057 0.069 0.078 | 0.784–0.819 | 0.106 0.107 0.132 | 0.642–0.692 |
| PiSSA r=8 | 13.8M | 0.134 0.215 0.299 | 0.644–0.774 | 0.306 0.356 0.588 | 0.160–0.367 |
| PiSSA r=8, lr 3e-5 | 13.8M | 0.069 0.081 0.086 | **0.818–0.834** | 0.322 0.322 0.493 | 0.637–0.692 |
| VeRA (r=256 scal.) | **1.0M** | 0.532 0.533 0.548 | 0.480–0.490 | 0.433–0.471 | 0.580–0.590 |

- **rsLoRA r=8 is the best cell measured at n=3** — ja CER median 0.069 (vs
  LoRA r=8's 0.080), es EM above baseline in 3/3 seeds, no diverged draw.
  **Top-up to 7 seeds (seeds 3–6, reviewer response, 2026-08-24):** ja
  0.136 / 0.175 / 0.214 / 0.093, es 0.090 / 0.225 / 0.275 / 0.153. No
  model-destroying draw in 7, but the tail reappears (seed 5: ja 0.214 / EM
  0.570, es EM 0.487 below baseline) and the pooled median moves to ja 0.093 /
  es 0.153 — still ahead of the LoRA r16 default pool (ja 0.114 over 14 runs)
  and still the tightest *real-scan* cell with no divergence, but the 3-seed
  picture was optimistic, exactly as §4b.3 predicts for any 3-seed read. New
  recommended default, with the gate.
- **rsLoRA r=16 is the worst cell measured**: one seed destroys the model
  outright (both languages EM 0.000, es CER *worse than baseline*). rsLoRA
  raises the effective adapter scale (α/√r > α/r); a moderate boost at r=8
  helps, the same boost on double the capacity is catastrophic. Scale is a tail
  *amplifier*, not a stabiliser — and the r8-vs-r16 gap widens under it.
  **At lr 3e-5 (reviewer response, 2026-08-24) rsLoRA r16 fully stabilises**:
  ja 0.057/0.069/0.078, es 0.106/0.107/0.132, EM ja 0.78–0.82 / es 0.64–0.69 —
  the best-behaved 3-seed real-scan cell measured (with the usual caveat that
  3-seed reads are optimistic, per the r8 top-up). So the catastrophe is an
  *effective-scale × learning-rate product* problem, not rank per se: what the
  rs-scaling multiplies, the lr can divide back. Also notable: the lr-1e-4
  seed-0 catastrophe was the only divergence in the project *visible in
  training loss* (3.67 vs siblings' 0.33) — genuine training collapse, unlike
  the silent basin divergences of §4b.3.
- **PiSSA r=8 refutes the initialisation hypothesis.** If divergence were a bad
  init-basin lottery, principled SVD init should shrink the spread. Instead
  everything is worse than LoRA r=8 (ja median 0.215 vs 0.080), the es EM
  collapse returns (3/3 below baseline), and the spread stays wide. Consistent
  with §4b.3's attribution to non-determinism *during* training.
  **At its authors' learning rate (3e-5, reviewer response, 2026-08-24)
  PiSSA recovers on Japanese**: ja CER 0.069/0.081/0.086 (median 0.081,
  level with rsLoRA r8 / LoRA r8) and the *highest ja EM of any cell*
  (0.818–0.834); es EM goes above baseline in 3/3 (0.637–0.692) but es CER
  stays poor (0.322–0.493, vs LoRA r8's 0.077–0.154). So the lr-1e-4 PiSSA
  row above is a learning-rate artefact, not an init result — the
  "one untuned lr" caveat is real and the sweep is a screening study. The
  init-basin hypothesis is still not supported: the spread at 3e-5 is as wide
  on es (0.32–0.49) as any LoRA cell.
- **VeRA locates the capacity floor.** Gains are the smallest (ja 0.846 →
  ~0.54) but the spread is the tightest of any cell ever measured (ja
  0.532/0.533/0.548) and EM improves over baseline on both languages with 1/14
  the parameters of LoRA r=8. The capacity–stability trade in its purest form:
  the "less is more" trend continues below r=8 for stability, but the CER gain
  falls off a cliff.

Ranking by ja-CER median (n=3 for every cell, for comparability; rsLoRA r8
is 0.093 at n=7): rsLoRA r8 (0.069) > LoRA r8 (0.080) > DoRA r16
(0.146) > PiSSA r8 (0.215) > VeRA (0.533) ≫ rsLoRA r16 (0.345, with a
model-destroying draw).

**Winner validation on synthetic ko/ja/en (family A→C, 3 seeds).** rsLoRA r8
improves both metrics in **8 of 9** language×seed cells — the exception is en
seed 2, which regresses slightly on both (CER 0.060 vs 0.047, EM 0.893 vs
0.902). An earlier version of this paragraph claimed 9/9; the adversarial
review caught the miscount (the paper text has been corrected too):

| Language | baseline CER/EM | after CER | after EM |
|---|---|---|---|
| ko | 0.124 / 0.837 | **0.000** / 0.002 / 0.046 | 0.907–0.997 |
| ja | 0.099 / 0.830 | 0.003 / 0.017 / 0.079 | 0.843–0.974 |
| en | 0.047 / 0.902 | 0.003 / 0.008 / 0.060 | 0.893–0.994 |

Korean reaches CER 0.000 on one seed *on held-out PII families*, and every
cell beats the LoRA-r16 synthetic result. With the XFUND ja/es results this
makes five language–corpus combinations, all improved on both metrics:
**rsLoRA r8, 1–2 epochs is the recommended default**, displacing LoRA r8.

### 4b.6 Second backbone: Qwen2.5-VL-3B (XFUND ja+es, 3 seeds)

The "single model" objection, and a harder test: Qwen2.5-VL-3B's baseline is
~7× stronger (ja CER 0.118 / EM 0.590 vs 2B's 0.846 / 0.400), so there is no
junk-output cliff for adaptation to fix.

| Language | baseline | after CER | after EM |
|---|---|---|---|
| ja | 0.118 / 0.590 | 0.075 / **0.226** / 0.047 | 0.826 / 0.597 / 0.831 |
| es | 0.171 / 0.615 | 0.107 / 0.153 / 0.063 | 0.700 / 0.602 / 0.740 |

Both phenomena replicate. The gain: 2 of 3 seeds roughly halve CER and raise
EM on both languages, so adaptation is not just weak-model repair. The tail:
seed 1's ja lands at 0.226 — **worse than its own baseline**, the first
observed case of adaptation hurting CER — while its siblings sit at
0.047/0.075. The heavy tail is not a Qwen2-VL-2B quirk.

### 4b.7 Cross-corpus transfer: synthetic ↔ real

The question that decides whether synthetic-only Korean adaptation means
anything: does adaptation learned on rendered forms transfer to real scans?
Train and eval corpora share no documents, rendering/scanning process, or value
distribution — the strongest held-out in the project.

**Synthetic → real (3 seeds):** on the interleaved ja+es eval half — baselines
ja 1.073 / EM 0.385, es 1.218 / EM 0.493:

| Language | after CER | after EM |
|---|---|---|
| ja | 0.236 / 0.310 / 0.295 | 0.522 / 0.540 / 0.573 |
| es | 0.171 / 0.254 / 0.167 | 0.497 / 0.433 / 0.533 |

Training on *rendered* text recovers most of the CER gain that in-domain
real-scan training achieves (ja ~0.28 vs ~0.12 against a ~1.0 baseline), and
part of the EM gain (ja to ~0.55 vs ~0.78). A first ja-only pass (before the
interleave fix, 50 ja docs) agrees: 0.937 → 0.202–0.266. And the es row is the
surprise: **Spanish does not exist in the synthetic training corpus at all**,
yet es CER drops to in-domain levels (0.17–0.25 vs in-domain 0.13–0.40) with EM
roughly flat — the adaptation transfers across corpus *and* language, which
fits it being mostly a transferable reading/format skill rather than
language-specific memorisation.

**Real → synthetic (3 seeds):** baseline ko 0.121 / ja 0.104 / en 0.047 →
seeds 0 and 1 *degrade* every language (ko 0.140/0.141, ja 0.133/0.135, en
0.082/0.250); seed 2 improves everything dramatically (ko 0.015, ja 0.021, en
0.016, EM 0.955–0.982). The transfer is **asymmetric**: syn→real helps in 3/3
runs, real→syn helps in 1/3 and hurts in 2/3 — with the heavy tail appearing on
the *good* side for once. Clean synthetic text is apparently in-distribution
enough for scan-trained adapters only when the draw is lucky.

For the paper: the syn→real direction is the useful one, and it is the
direction that works. Synthetic-only Korean adaptation is defensible.

**Matched-evaluation re-run (reviewer response, 2026-08-24).** The table
above evaluates on a different XFUND half than the in-domain runs (so the
baselines differ: 1.073 vs 0.846). Re-running syn→real with
`eval_matched_half: true` — the *same* 1,198-region half the in-domain runs
score on — gives baselines bit-identical to the in-domain ones (ja 0.8457 /
EM 0.400, es 0.9238 / EM 0.520), so the two settings are now directly
comparable:

| Language | in-domain after CER (14-run median) | syn→real after CER (3 seeds) | syn→real after EM |
|---|---|---|---|
| ja | 0.114 | 0.155 / 0.155 / 0.138 | 0.475 / 0.465 / 0.517 |
| es | ~0.14–0.40 | **0.567** / 0.148 / 0.137 | 0.623 / 0.608 / 0.633 |

Transfer recovers ~85% of the in-domain ja CER gain (0.846→0.14–0.16 vs
→0.114) and about a third of the EM gain (0.40→0.47–0.52 vs →~0.78). Spanish,
still absent from training, lands at in-domain level in 2/3 seeds — and seed 0
is a **language-selective divergence**: es 0.567 with ja 0.155 and training
loss 2e-5, indistinguishable from its siblings. That is the 15th trained
adapter in the pool and the second divergence (after qwen25 seed 1), and it
repeats the lesson of §4b.3: a per-language gate, not a macro gate, since the
macro CER 0.452 would still "beat baseline" 0.902.

### 4b.8 Sizing the eval gate

§4b.3 concluded a post-training eval gate is mandatory. The saved per-region
(pred, gold) pairs put a number on it: bootstrap gate sets of n regions (paired
sampling — one gate set, both adapters run on it), and measure how often the
gate ranks a diverged adapter worse.

| Detection task | n=10 | n=25 | n=50 | n=100 | n=200 |
|---|---|---|---|---|---|
| Subtle: qwen25 ja diverged seed vs baseline (ΔCER 0.11) | 0.851 | 0.919 | 0.969 | **0.997** | 1.000 |
| Gross: r32 es diverged seed vs sibling (ΔCER 0.39) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**A 100-region gate catches the subtlest observed divergence 99.7% of the
time; gross divergence is caught with 10.** At 0.7 s/region the gate costs
~70 seconds. Script: `scripts/eval_gate_analysis.py`.

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

## 4c. Second adversarial review round (7-agent, 2026-08-24): what changed

The user-run 7-agent review (`paper/Review/MACULAR_adversarial_review.md`)
returned REJECT with 12 resubmission conditions. This section records what was
verified, what was fixed in the text, what was answered with new measurement,
and what remains open.

### 4c.1 Numerical audit (all four reviewer recalculations checked against data)

| Claim in paper | Reviewer | Recomputed | Verdict |
|---|---|---|---|
| Mann–Whitney es-vs-zh co-training, EM p=0.023 / CER p=0.051 | 0.026 / 0.053 | **0.026 / 0.053** (exact, raw values) | reviewer right; fixed |
| Table 1 "macro" CER before 0.088 | 0.0907 | 0.088 is the length-weighted all-region CER; 0.0907 is the unweighted 3-language mean | label ambiguity; row renamed "all regions (length-weighted)" |
| Transfer recovers "~85%" of in-domain gain | 94–99% | (0.846−0.155)/(0.846−0.114) = **0.944** | reviewer right; fixed to 94% with the formula printed |
| Abstract "ja median 0.846→0.114" in the 28-cell (14-adapter) context | pooled median 0.151 | 14-run pooled median = **0.151**; 0.114 is ja+es only | reviewer right; abstract/§4.3 now give both |
| "gross ΔCER 0.39" provenance | untraceable | r32 es seed 1 (0.558) vs seed 0 (0.165) | now stated in text |
| Donut cited for SynthDoG | citation error | SynthDoG *was* released in the Donut paper (Kim et al., ECCV 2022) | citation correct; wording now says so |

### 4c.2 Gate: document-level bootstrap, the missing sibling case, prospective rule

`scripts/gate_reviewer_response.py` (doc membership reconstructed by replaying
the eval-item construction and checking golds against the saved pairs).

Detection rate, region-level / document-level bootstrap:

| case | ΔCER | n=10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| qwen25 s1 vs baseline (subtle) | 0.11 | .851/.837 | .919/.945 | .969/.977 | .997/.995 | 1/1 |
| r32 es s1 vs s0 (gross) | 0.39 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| qwen25 s1 vs sibling s0 | 0.15 | .914/1 | .981/1 | .998/1 | 1/1 | 1/1 |
| rsLoRA r8 s5 vs s6 | 0.12 | .880/1 | .936/1 | .980/1 | 1/1 | 1/1 |
| det s3 (bad basin) vs s0 | 0.10 | .954/1 | .990/1 | .999/1 | 1/1 | 1/1 |
| rsLoRA r8 s3 vs s6 (marginal) | 0.04 | .487/.526 | .540/.561 | .516/.604 | .542/.661 | .617/.728 |
| syn→real es s0 vs s1 (language-selective) | 0.42 | .430/.430 | .495/.527 | .601/.585 | .755/.703 | .911/.831 |

- Document-level resampling does **not** weaken the gate — sibling cases go
  to 1.000 at n=10 because a diverged adapter fails whole pages. The
  i.i.d. objection was worth testing and turned out to point the other way.
- Beats-baseline-but-loses-to-sibling at Δ 0.10–0.15: separated ≥98% at n=50,
  100% at n=100. Δ 0.04 is not separable at any n tested: the gate's
  resolution is ~0.1 CER.
- **New failure mode**: the language-selective transfer divergence (es 0.567)
  is caught only 75% at n=100 despite Δ 0.42, because **79% of its excess CER
  sits in 10 regions** with runaway generation (per-region CER > 1; sibling:
  19% in top-10). Length-weighted CER is fragile to runaways → recommend
  clipping per-region CER at 1 *and* gating on EM.
- **Prospective rule** (fixed before looking: per-language 100-region gate,
  reject if gate CER > 1.5× the 14-run pool median → ja 0.171 / es 0.449),
  applied to the 17 adapters trained after §4b.8 was written (34 language
  cells): 3 rejects (rsLoRA r8 s4 ja 0.175 p=.55, s5 ja 0.214 p=.82, det s3
  ja 0.181 p=.64), 31 accepts, **32/34 agree with the full-set verdict**. Both
  misses are es runs over the ceiling on the strength of a few runaway regions
  (PiSSA-3e-5 s0 0.493, syn→real s0 0.567) — same fragility.

### 4c.3 Adaptation vs model swap vs classical engine (same 1,198 crops)

`scripts/easyocr_xfund_baseline.py` → `results/easyocr_xfund_eval_half.json`.

| system | ja CER | ja EM | es CER | es EM |
|---|---|---|---|---|
| EasyOCR (no training) | 0.213 | 0.441 | 0.208 | 0.548 |
| Qwen2-VL-2B zero-shot | 0.846 | 0.400 | 0.924 | 0.520 |
| + LoRA r16 (7-run median) | 0.114 | 0.759 | 0.299 | 0.260 |
| + rsLoRA r8 (7-run median) | 0.093 | 0.779 | 0.153 | 0.658 |
| Qwen2.5-VL-3B zero-shot | 0.118 | 0.590 | 0.171 | 0.615 |
| + LoRA r16 (3 seeds) | 0.047–0.226 | 0.597–0.831 | 0.063–0.153 | 0.602–0.740 |

The reviewer's "most destructive" point is conceded in print: the headline
0.846→0.114 is mostly junk-baseline recovery, and a model swap reaches it
zero-shot. Adaptation still helps on top of the 3B (2/3 seeds, best cells in
the table), and EasyOCR's es EM 0.548 beats every adapted-2B es EM — the
reference the es-EM-degradation finding lacked. New paper §4.8.

### 4c.4 Text-level fixes applied

- Redaction: every universal ("no mechanism defeats a nonlinear attacker")
  scoped to the attackers run; LEACE value-shift = *partial* protection (removes
  27% / 67% / 19% of the linear excess on Paddle/Qwen/Ministral), stated with
  the same-scale-as-hard-mask-residual double standard acknowledged; hard mask
  explicitly oracle-conditioned and described as an upper bound on a deployed
  masking pipeline; attacker-strength caveat in the protocol paragraph.
- Determinism: "repeatable, not rarer" → "repeatable rather than removed"
  (the rate claim was unsupported: 1/3 distinct seeds deterministic vs 1/14).
- Medical framing and the non-joint nature of §4/§6 stated in the intro's
  second paragraph; generator spec (2 doc types, Noto CJK, rotation+noise) and
  the no-realism-check caveat in §3.
- Related work: Dodge 2020, Mosbach 2021, He 2025 (nondeterminism), LayoutLMv3
  / UDOP (task non-comparability stated), DP-SGD (different quantity, not
  compared). 34 references.
- Implementation details: LoRA targets, dropout, schedule, step size, region
  counts; relation graph = 2-layer Transformer encoder (d=128, 4 heads); repo URL.
- Discussion: PEFT recommendation qualified by the 7-seed tail and the two lr
  controls; full-FT control absent stated.

### 4c.5 Cascade re-run with seeds

See §1.2: 3/3 seeds flat on default, 3/3 declining on counterfactual, absolute
levels reproduce the original single run at 80 epochs. Condition 7 closed.

### 4c.6 Still open (need new experiments or are out of reach)

| condition | status |
|---|---|
| Vec2Text-style iterative inverter | not run (est. 1–2 GPU-days incl. training an inverter per backbone) |
| PII detector composed with hard mask (end-to-end leakage) | not run (detector exists; ~half a day) |
| Per-method lr sweep for all PEFT variants | two controls run (PiSSA, rsLoRA r16); full sweep ~2 GPU-days |
| PaddleOCR-VL as a *recognizer* on the same crops | not run (~hours; env conflict with torch noted in baselines/ocr.py) |
| Full fine-tuning control for the 2B | not run |
| XFUND template near-duplicate audit | not run (cheap; image-hash pass) |
| Probe selectivity / control task | not run |
| Family-shift artefact control | not run |
| Real Korean scanned medical corpus | does not exist; stated as field gap |

### 4c.7 Round-2 experiments (2026-08-24, running list)

- **XFUND template near-duplicate audit** (`scripts/xfund_template_audit.py`,
  16×16 dHash, <24/256 bits = same template): ja+es **1 cross-half pair**
  (es_val_16 / es_val_5, 17 bits) out of 1,250 same-language cross pairs,
  ja+zh **0**; median cross-half distance 112 bits; 0 within-half pairs.
  Template leakage across the self-split is negligible. Condition 9 closed.
- **Hard mask is NOT an oracle** — correction of our own paper text. `core.py`
  `redaction="hard"` thresholds the model's *own* PII head at 0.5 (detached),
  so every hard_mask row already composes a learned detector (val AP:
  Paddle 0.982, Qwen 0.923, Ministral 0.989, FUNSD 0.888). The paper's threat
  model, intro, abstract, conclusion and limitations now say so, and Table 6
  gained a det.-AP column. Reviewer condition 5 ("oracle synthesis") is
  answered by the existing measurement; what remains unmeasured is an
  *external* tagger composed with the mask.
- **PaddleOCR-VL as recognizer**: blocked — its remote generation code targets
  transformers 4.x and fails on 5.14 (ROPE_INIT_FUNCTIONS); documented in
  `ocr_adapt.py`. Not run.
- **Stronger attacker + probe selectivity** (`scripts/strong_attack.py`,
  PaddleOCR-VL cached features, CPU): GRU inverter with h=1024 / 1200 epochs /
  beam-5, and a Hewitt–Liang control task (random label per unique text) for
  the linear and MLP probes. Running.
- **LoRA r16 / r8 / DoRA r16 at lr 3e-5** (3 seeds each): running on GPU, so
  the rsLoRA-r8 recommendation is compared against every other method at the
  lower rate as well.

**Stronger attacker + selectivity — result** (`results/strong_attack_paddleocr_vl.json`,
3-seed means; paper's original inverter in parentheses):

| mechanism | inv-XL EM | inv-XL EM ctx | MLP selectivity (task / control) | ctx sel. |
|---|---|---|---|---|
| none | 0.163 (0.157) | 0.076 (0.076) | 0.209 (0.982 / 0.773) | 0.290 |
| hard_mask | 0.029 (0.013) | 0.007 (0.006) | 0.215 (0.967 / 0.753) | 0.163 |
| gate | 0.033 (0.022) | 0.014 (0.012) | 0.220 (0.977 / 0.757) | 0.234 |
| leace | 0.137 (0.144) | 0.101 (0.086) | 0.170 (0.942 / 0.772) | 0.204 |

Beam ≈ greedy (±0.005); prior floor 0.000. 2× hidden / 3× epochs / beam-5
recovers nothing extra from raw features, doubles the small masking residue,
leaves LEACE unchanged: **ordering unchanged**. Probes have selectivity
0.17–0.22 everywhere → they read the representation. What remains untested is
an attacker that re-embeds hypotheses (needs the backbone in the loop). New
paper §6.3 + Table 7. Condition 1 partially closed (decoder strength), not
the Vec2Text loop itself.

**LoRA r16 at lr 3e-5 (3 seeds)** — ja 0.087 / 0.096 / 0.102 (EM 0.751–0.759),
es 0.112 / 0.135 / 0.120 (EM 0.633–0.662, *above* baseline 0.520 in 3/3).
Against the 1e-4 default (ja 0.099–0.147 + 0.630; es 0.132–0.399 with EM
below baseline in 6/7): the lower rate alone tightens ja, fixes the es
exact-match collapse and produced no divergence in three draws. The "less
is more" ablation (fewer epochs, lower rank) and the rs-scaling result were
all moving the same knob — effective update size — and the default lr was
simply too high for this task. r8 / DoRA at 3e-5 pending before the paper
paragraph is rewritten.

**LoRA r8 at lr 3e-5 (3 seeds)** — ja 0.102 / 0.113 / 0.116 (EM 0.753–0.761),
es 0.127 / 0.136 / 0.129 (EM 0.642–0.652). Tight and undiverged like r16 at
3e-5, but slightly *behind* r16 on ja: once the rate is right, the extra rank
helps a little instead of hurting — the r8-over-r16 advantage at 1e-4 was
the rank acting as an update-size brake, not capacity being harmful.

**DoRA r16 at lr 3e-5 (3 seeds)** — ja 0.084 / 0.099 / 0.101 (EM 0.774–0.801),
es 0.104 / 0.125 / **0.328** (EM 0.662–0.708). ja matches LoRA r16 at 3e-5;
es has one moderate tail draw (0.328, EM still above baseline). The lr-matched
sweep at 3e-5 now covers LoRA r16/r8, DoRA r16, PiSSA r8, rsLoRA r16 —
**ja medians: rsLoRA r16 0.069 < PiSSA 0.081 < LoRA r16 0.096 ≈ DoRA 0.099 <
LoRA r8 0.113** — every method's es EM is above baseline, and no
model-destroying divergence appeared in any of the fifteen 3e-5 runs. At the
matched lower rate the differences between PEFT methods shrink to ~0.04 CER,
versus the 10× spread the shared 1e-4 produced. The 1e-4 variant ranking was
mostly a learning-rate artefact; the update-size story (§4b.5) survives, the
method ranking does not.

**Full fine-tuning control (language tower 1.54B params, vision frozen,
lr 1e-5, 3 "seeds")** — ja 0.127 / EM 0.656, es 0.387 / EM 0.362, identical
across all three seeds: full FT has no seed-dependent randomness in this
pipeline (no adapter init, no dropout in the base model), so within one
process the three runs are bit-identical — effectively a single run.
Worse than every 3e-5 LoRA cell on both languages (ja 0.127 vs 0.084–0.116;
es 0.387 vs 0.104–0.136) and es EM is *below* baseline. One untuned lr
(1e-5), so this is a control, not a tuned comparison — but PEFT beating
full FT at 26–58× fewer trainable parameters closes reviewer condition
"full-FT control absent". GPU queue complete.

## 4d. Round-2 revision review (BORDERLINE): mandatory fixes + small experiments

The 7-agent re-review returned BORDERLINE (Reject 0; was 5) with 5 mandatory
text corrections (N1–N5) and small-experiment recommendations. Actions:

- **N1** Abstract gate claim rewritten: no more bare "≥99.7%" — now states
  ≥99.5% region- and document-level for every ΔCER ≥ 0.10 case, names the
  0.04-margin and runaway-es failures, and cites the 32/34 prospective result.
- **N2** §7 "no protection under value shift" → "partial protection
  (19–67% of linear excess removed)" — now consistent with §6.4.
- **N3** Gate rule wording fixed: 0.171/0.449 are *ceilings* = 1.5× the
  per-language medians (0.114/0.299) of the **7-run ja+es** pool (not
  "14-run pool medians"). Script comment fixed too.
- **N4** §4.8 EasyOCR scope fixed: es EM 0.548 beats every
  *default-configuration* 2B run (≤0.538); rsLoRA r8 (0.658) and the 3e-5
  cells surpass it.
- **N5** Table 6 inv. CER column regenerated from the Zenodo archive
  (3-seed means of ctx_inversion_cer): paddle 0.433/1.163/0.755/0.544,
  qwen 0.556/1.039/0.814/0.913, ministral 0.476/1.024/0.597/0.630. The old
  column carried values from the pre-rerun run1 file; probes/AP/F1 columns
  were already archive-consistent, and claim (4)'s above-prior EMs verify
  exactly. Aggregation ("mean over 3 seeds") now stated in the caption; the
  Qwen det.AP = clinical-F1 = 0.899 equality is a verified coincidence
  (AP per seed 0.909/0.869/0.919).
- **N6** hard mask description corrected: not a "learned constant" — a
  detached probability-weighted *type embedding* (type retained by design,
  value dropped). Claim (3) "share one constant vector" reworded.
- **N7** Abstract floor claim scoped: floor reached on 1 backbone, within
  0.012–0.032 on the other two.
- **N8** `requirements-lock.txt` added (transformers 5.14.1, peft 0.20.0,
  torch 2.11.0+cu128, easyocr 1.7.2 …); paper env sentence now points at it.

**Recommendation 6+7 executed** (`scripts/detector_probe_metrics.py`,
paddle cache, 3 seeds, held-out family B):

| mechanism | det P@0.5 | det R@0.5 | ctx MLP balanced acc | (raw acc) |
|---|---|---|---|---|
| none | 0.74–0.76 | 0.980–0.983 | 0.861–0.865 | 0.973–0.975 |
| hard_mask | 0.815–0.823 | 0.977–0.978 | **0.164–0.196** | 0.837–0.849 |
| gate | 0.784–0.789 | 0.977–0.983 | 0.593–0.743 | 0.924–0.946 |
| leace | 0.738–0.751 | 0.980–0.981 | 0.532–0.585 | 0.912–0.924 |

Two upgrades: (a) the detector runs **recall-biased** at the deployed
threshold (<3% of sensitive regions escape the mask, on the held-out
family — the paper's own value-shift condition); (b) balanced accuracy
shows the hard mask collapses per-type discrimination to near chance
(0.125), which raw accuracy against the 0.847 majority had compressed.
Both now in §6.1/§6.2.

**Not done, flagged to the user**: PaddleOCR-VL as recognizer (blocked on
transformers 5.14, documented), §4+§6 coupling experiment (adapted-backbone
features through the attack protocol — feasible, ~half a day GPU), real
medical arm / title decision (CHIP2022 MedOCR), Vec2Text-loop attacker.
