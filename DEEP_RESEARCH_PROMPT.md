# MACULAR — Deep Research Prompt

다른 AI(Deep Research / o3 / Gemini DR 등)에 그대로 붙여 넣는 용도입니다.
본문을 **영어로** 쓴 이유: 문헌 검색·인용 정확도가 한국어 질의보다 확실히 높습니다.
필요하면 맨 아래 `## 사용법`만 읽고 복사하세요.

---

## THE PROMPT (copy from here)

You are a research analyst helping a team turn a **negative experimental result** into a
publishable contribution at a strong venue. Do a deep literature and tooling review.
Prioritize **peer-reviewed work with released code**. Cite everything (venue + year +
link). Where you are inferring rather than citing, say so explicitly.

### 1. What the system is

We are building **MACULAR**: privacy-preserving information extraction from **scanned
medical documents** (images, not clean text), multilingual with a focus on **Korean /
Japanese / Chinese / English**.

Architecture as currently implemented:

1. A vision-language-model (VLM) vision tower encodes the page in **one forward pass**;
   region features come from **ROI pooling over the patch grid** (no per-crop re-encode).
2. A **region proposer** does detection only (it never reads text), so layout position
   cannot leak label information through the proposer.
3. A **PII classifier head** predicts, per region, whether the region is identifying and
   of what type.
4. A **differentiable redaction gate** replaces the representation of PII regions with a
   learned per-type embedding:
   `z_safe = (1 − m) ⊙ z + m ⊙ e_type`, where `m = 1 − P(non-PII)` is a **soft, gradient-carrying**
   mask. The claimed novelty is that redaction is *learned end-to-end in representation
   space* rather than applied as a post-hoc hard mask over text.
5. A **relation graph** (transformer encoder over regions) contextualizes regions.
6. An **EMA teacher on the raw (unredacted) context** and a **student on the safe view**,
   trained with a consistency loss, so clinical utility survives redaction.
7. A **privacy adversary** attached through a **gradient reversal layer (GRL)** that tries
   to reconstruct region content from the safe representation.
8. Outputs compile to **FHIR** resources (Observation with LOINC, MedicationRequest with
   RxNorm).

### 2. What we have already measured (do NOT re-derive this; target the gaps)

Setup: PaddleOCR-VL-0.9B and four other VLM towers, single RTX 5090 (32 GB), synthetic
multilingual scanned medical forms plus FUNSD as real scanned data. Privacy metric:
`identity_leakage = (held-out attacker's cosine similarity to true region content)
− (PII-type-mean baseline cosine)`. The attacker is a **different architecture, trained
post-hoc, on unseen regions**.

Positive results:
- **Backbone features matter.** Replacing hand-crafted features with real VLM ROI features
  improves PII average precision by **+0.152 to +0.377 AP** across five backbones
  (Qwen3-VL, Qwen2-VL-2B, Ministral-3, Llama-3.2-Vision, PaddleOCR-VL).
- **OCR error cascade is real.** Downstream PII F1 falls 0.648 → 0.530 as OCR CER rises,
  but only on shortcut-controlled data.
- **Shortcut control works.** A geometry-only (coordinate-only) audit classifier reaches
  F1 0.432 ≈ the base rate on our counterfactual-layout data, i.e. position alone no
  longer predicts PII.

Negative results — **this is the problem we need help with**:
- With a **frozen** vision tower, the gate + consistency + adversary **cost a lot of
  clinical utility** (clinical macro-F1: full 0.779 vs. no-consistency 0.953 vs. no-gate
  0.933) with **no measurable privacy gain** (identity leakage: safe +0.0103 vs.
  unprotected raw +0.0091).
- With **LoRA on the vision tower**, leakage *does* drop, but **every variant drops it by
  the same amount**: full (soft gate + adversary) −0.0224, no-gate (hard mask) −0.0236,
  no-adversary (gate only) −0.0238, within-row against each variant's own raw baseline
  (n = 863 PII regions).
- **The leakage measurement does not reproduce, and a 3-seed sweep confirmed it is
  unusable.** Re-running the identical config first gave reductions that disagreed in
  *sign*. A proper 3-seed × 3-variant sweep then gave: `full` −0.0020 ± 0.0066, `no_gate`
  −0.0458 ± 0.0457, `no_adversary` −0.0261 ± 0.0617 — **std larger than the mean in every
  row**, and **2 of 9 runs collapsed outright** (PII average precision 0.15, i.e. below the
  base rate, clinical macro-F1 exactly 0.0, while other seeds of the same config reached
  0.99). So we can say nothing about the gate or the adversary from this experiment: the
  training is unstable at this budget (24 docs, 12 epochs) *and* the metric is too noisy.
  Both are being fixed; assume neither is the interesting question.
- GRL sign, warm-up, and gradient arrival are unit-tested, so the null effect is not an
  implementation bug.

Caveats we already know: tiny training budget (24 documents, 12 epochs), one backbone.
Do not just tell us to add seeds; assume we will.

### 3. Your questions to answer

**Q1 — Is our negative result already known, and what replaced the failed method?**
Adversarial / GRL-based removal of an attribute from a representation. We suspect this is
a known failure mode. Find and summarize:
- the papers showing adversarial attribute removal does not actually remove the
  information (and what their diagnostic was);
- the **provable / closed-form concept-erasure** line of work that succeeded it
  (e.g. nullspace-projection and least-squares concept-erasure families), including what
  guarantee each one gives, under which threat model, and its cost;
- whether any of it has been applied to **vision or multimodal region features**, as
  opposed to text embeddings. If nobody has done the vision case, say so plainly — that is
  our opening.
For each candidate method report: guarantee type (empirical / linear-guarded / DP),
compute cost, code + license, and whether it is differentiable end-to-end.

**Q2 — Our privacy metric is too noisy AND too weak. What should replace it?**
Cosine similarity to content features is a weak adversary; a reviewer will not accept it,
and per §2 it does not even reproduce across runs — its variance swamps the effect. Two
sub-questions, both important:
(a) *Sensitivity.* What makes a representation-leakage measurement stable enough to detect
small effects? Look for reported variance/error bars in privacy-attack papers, standard
seed counts, whether people bootstrap over the attack set, and whether any paper documents
that weak attacks are too noisy to rank defenses. If there is a recommended protocol
(fixed attacker, multiple attacker restarts, attack-set bootstrap CI), give it to us.
(b) *Strength.* Find the strongest published attacks that would apply to per-region VLM
representations:
- **embedding inversion / text reconstruction from embeddings** (methods that recover the
  literal string from a vector);
- **attribute inference** and **membership inference** protocols with their standard
  reporting conventions;
- **reconstruction from intermediate vision features** in document or medical imaging.
Tell us which attack is the current standard to report against, what metric it uses
(exact-match, BLEU/ROUGE, ε, AUC), whether an implementation exists, and — critically —
whether that metric is **discrete enough to be stable** (e.g. exact-match recovery rate)
rather than a continuous similarity that drifts with training noise.

**Q3 — What privacy guarantee would a top venue actually require?**
Compare, for our setting: (a) empirical attack results only, (b) linear-guardedness
certificates, (c) **DP-SGD** with a stated ε, (d) k-anonymity-style structural claims.
For each: what would we have to change, what utility loss is reported in the literature at
useful ε, and are there papers that combine DP with document/multimodal extraction?
Include what ε values reviewers in health informatics currently treat as meaningful.

**Q4 — Baselines we will be required to beat.**
List the de-identification and document-extraction baselines a reviewer in this area will
demand, with code links and licenses. We specifically need:
- rule-based / hybrid clinical de-id systems;
- transformer clinical de-id models;
- **layout-aware document models** — note that we have deliberately excluded
  CC-BY-NC-SA-licensed models from our core, so flag license for each;
- **prompted large VLMs** as a strong zero/few-shot baseline;
- anything specifically evaluated on **CJK** documents or **scanned** (not text) clinical
  forms.
For each: what metric it reports, on what dataset, and its published number, so we can
build a comparison table.

**Q5 — Datasets. What is publicly usable, and what is the CJK situation?**
We cannot touch real PHI. We need:
- public **scanned document** corpora with layout annotation, including multilingual/CJK
  ones;
- public clinical de-identification corpora and their **access procedure** (DUA, training,
  cost, how long approval takes);
- any public **medical form / prescription / lab-report image** corpus in Korean,
  Japanese, or Chinese — including hospital or government open-data releases, shared tasks,
  and non-English-language publications. This is the single hardest gap for us, so search
  in Korean, Japanese, and Chinese too.
For each dataset: size, languages, annotation type, license, and whether images are real
scans or synthetic.

**Q6 — Is the synthetic-data-only route publishable, and how do we defend it?**
Find papers that publish clinical NLP/vision results on **synthetic or surrogate** data
and survived review. What did they do to establish that findings transfer — a real-data
subset, an expert audit, a distribution-shift analysis? Give us the concrete defense
pattern that worked.

**Q7 — Venues and the framing decision.**
Suggest target journals/conferences for a **privacy-preserving multilingual scanned
medical document extraction** paper. For each: scope fit, whether they publish negative or
replication results, typical evaluation bar, and review timeline. Then advise on framing:
- **Option A** — reframe as a rigorous negative/replication result ("differentiable
  redaction in representation space provides no privacy benefit over hard masking, and
  here is the attack methodology that shows it"). Is that publishable, and where?
- **Option B** — replace the mechanism with something that has a real guarantee (from Q1)
  and publish the positive result.
- **Option C** — pivot the contribution to what already works for us: multilingual scanned
  document extraction with quantified OCR-cascade effects and shortcut controls, with
  privacy as a secondary axis.
Recommend one, with reasoning tied to the numbers in §2.

**Q8 — Highest-expected-gain changes, ranked.**
Given a **single RTX 5090 (32 GB)**, no PHI access, and Apache-2.0/MIT-only model
licensing for anything in the core: rank concrete changes by expected effect size per
GPU-hour. Include the ones we may be blind to (e.g. whether our gate is applied at the
wrong layer, whether region-level redaction is the wrong granularity, whether the teacher
should see the raw view at all). Say explicitly which of our current design choices the
literature suggests is simply wrong.

### 4. Hard constraints (respect these in every recommendation)

- No real patient PHI, ever. Public or synthetic data only.
- Core models must be **Apache-2.0 / MIT**. Flag any CC-BY-NC-SA or research-only license
  as excluded-from-core (usable only as a cited comparison).
- Single RTX 5090, 32 GB VRAM. No multi-node.
- Must handle **scanned images**, not clean text, and must work for **ko/ja/zh**, where
  whitespace-based WER is meaningless.
- Solo author + one collaborator. Prefer methods with released code.

### 5. Output format

1. **Bottom line** — 10 lines max: is the gate salvageable, what replaces it, which venue.
2. One section per question Q1–Q8.
3. **Method shortlist table**: method | guarantee | cost | code | license | fits our stack?
4. **Dataset table**: name | languages | real scans? | annotation | license | access path.
5. **Baseline table**: system | dataset | metric | published number | license.
6. **Reading list**: 15–25 items, ranked, one line each on why it matters to us.
7. **What you could not find** — explicit gaps. Do not fill them with plausible guesses;
   an honest "no public CJK medical form corpus exists" is more useful to us than a
   fabricated citation.

Do not invent citations. If a number is not in the source, do not report it.

## END OF PROMPT

---

## 사용법

- 위 `## THE PROMPT` ~ `## END OF PROMPT` 사이만 복사해서 붙이면 됩니다.
- 여러 AI에 돌릴 때 **§2의 수치를 절대 지우지 마세요.** 이미 측정한 걸 다시 하라고
  답하는 걸 막아 주는 유일한 장치입니다.
- 답변 받으면 Q1(concept erasure)과 Q2(공격 강도·안정성) 두 개를 먼저 봐 주세요.
  나머지는 논문 포장 문제고, 이 둘은 **구현이 바뀌는** 문제입니다.
- Q2가 특히 급해진 이유: 동일 설정 재실행에서 누출 수치의 **부호가 뒤집혔습니다.**
  측정기가 신호보다 잡음이 크면 seed를 늘려도 결론이 안 나옵니다. 그래서 Q2를
  "더 센 공격"만이 아니라 "**안정적인** 공격"까지 묻도록 (a)/(b)로 쪼갰습니다.
