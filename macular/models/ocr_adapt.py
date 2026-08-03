"""Domain-adapt a document VLM to medical forms, and measure whether CER drops.

Everything else in this project MEASURES OCR (engine benchmarks, error-cascade
curves, redaction mechanisms). Nothing improves it. This does the missing piece:
LoRA fine-tune PaddleOCR-VL on medical-form region crops and report per-language
CER on a held-out split against the same model before adaptation.

Design notes:
  - Paired comparison in ONE run. The baseline is evaluated first, then LoRA is
    attached and trained, then the same regions are evaluated again. Same model,
    same crops, same decoding — so the delta is the adaptation and nothing else.
  - Held out by GENERATOR FAMILY. Train uses family A, evaluation uses family C,
    so no name, address, phone prefix or ID pattern is shared. Adaptation cannot
    be memorisation of the values.
  - Per-language CER, and WER is not reported: whitespace word error is
    meaningless for ko/ja/zh.
"""

from __future__ import annotations

import os

import torch

from ..baselines.ocr import _crop
from ..evaluation.metrics import cer as _cer

# Qwen2-VL-2B, not PaddleOCR-VL. PaddleOCR-VL is the stronger document model and
# it is what the feature-extraction experiments use, but its remote modeling code
# is written against transformers 4.x: loading it for GENERATION on 5.14 fails
# first on ROPE_INIT_FUNCTIONS['default'] and then on a missing
# compute_default_rope_parameters attribute. Two monkeypatches deep into a
# third-party file is the wrong trade for a fine-tuning target, and Qwen2-VL is
# natively supported, Apache-2.0, OCR-capable, and fits the 32 GB budget.
MODEL = "Qwen/Qwen2-VL-2B-Instruct"
PROMPT = "Transcribe the text in this image exactly. Output only the text."


def _load(device="cuda", dtype="bfloat16", model_id=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    src = model_id or os.environ.get("MACULAR_OCR_MODEL", MODEL)
    proc = AutoProcessor.from_pretrained(src)
    model = AutoModelForImageTextToText.from_pretrained(
        src, dtype=getattr(torch, dtype), device_map=device)
    return proc, model


def _prompt_text(proc):
    """Chat-formatted prompt ending where the model should start transcribing."""
    msg = [{"role": "user", "content": [{"type": "image"},
                                        {"type": "text", "text": PROMPT}]}]
    return proc.apply_chat_template(msg, add_generation_prompt=True,
                                    tokenize=False)


# Longest crop side handed to the model. Real scans are far larger than our
# rendered pages, and _crop upscales 3x on top of that, so an unbounded crop
# turns into thousands of visual tokens under Qwen2-VL's dynamic resolution: an
# XFUND run sat at 100% GPU for 18 hours on the same step count the synthetic
# run finished quickly. These are single-line text crops; 512px reads them.
MAX_CROP_SIDE = 512


def interleave_by_language(docs):
    """Round-robin the languages so a head-slice keeps all of them.

    Everything downstream slices ``docs[:max_docs]``, and corpus files can be
    language-grouped (XFUND's is). A grouped file plus a head-slice is silently
    monolingual: the syn2real run evaluated on 50 ja documents and reported no
    es at all — the same failure family as the flat halving in
    halve_by_language, one code path over.
    """
    by_lang: dict[str, list] = {}
    for d in docs:
        by_lang.setdefault(d.language, []).append(d)
    pools = [by_lang[k] for k in sorted(by_lang)]
    out = []
    for i in range(max(len(p) for p in pools)):
        out.extend(p[i] for p in pools if i < len(p))
    return out


def halve_by_language(docs):
    """Split one document population into (train, eval), alternating per language.

    XFUND ships a single split, so the fallback is a halving of it. It has to
    alternate WITHIN each language: the jsonl is written language-grouped, so a
    flat ``docs[:half], docs[half:]`` trained on all-ja and evaluated on all-es
    — a cross-lingual transfer test that reported no ja row at all.
    """
    ordered = sorted(docs, key=lambda d: d.language)
    return ordered[0::2], ordered[1::2]


def _regions(docs, data_dir, max_docs, max_regions):
    """Yield (crop, gold_text, language) for non-empty regions."""
    from PIL import Image
    for doc in docs[:max_docs]:
        img = Image.open(os.path.join(data_dir, doc.image_path)).convert("RGB")
        for c in doc.candidates[:max_regions]:
            text = (c.text or "").strip()
            if not text:
                continue
            crop = _crop(img, c.bbox, doc.width, doc.height)
            if crop is None:
                continue
            crop = crop.convert("RGB")
            if max(crop.size) > MAX_CROP_SIDE:
                crop.thumbnail((MAX_CROP_SIDE, MAX_CROP_SIDE), Image.LANCZOS)
            yield crop, text, doc.language


@torch.no_grad()
def evaluate_cer(proc, model, items, max_new_tokens=48, pairs_out=None):
    """Per-language CER over (crop, gold, language) triples.

    ``pairs_out``: optional list; receives (language, pred, gold) per region.
    Saved into results so the eval-gate question — how few regions does it take
    to detect a diverged adapter (§4b.3)? — is answerable offline instead of
    needing reruns.
    """
    model.eval()
    per_lang: dict[str, list] = {}
    for i, (crop, gold, lang) in enumerate(items):
        if i % 200 == 0:
            print(f"  eval {i}/{len(items)}", flush=True)
        inputs = proc(images=crop, text=_prompt_text(proc),
                      return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
        # Decode only the generated continuation, not the echoed prompt.
        pred = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()
        per_lang.setdefault(lang, []).append((pred, gold))
        if pairs_out is not None:
            pairs_out.append((lang, pred, gold))

    report = {}
    num = den = 0
    for lang, pairs in per_lang.items():
        n = sum(_cer(p, g) * max(1, len(g)) for p, g in pairs)
        d = sum(max(1, len(g)) for p, g in pairs)
        # Exact match alongside CER, because part of any gain here is the model
        # learning the output FORMAT rather than reading better: the unadapted
        # model sometimes answers with grounding coordinates instead of text.
        report[lang] = {"cer": n / d, "n_regions": len(pairs), "wer": None,
                        "exact_match": sum(p == g for p, g in pairs) / len(pairs)}
        num, den = num + n, den + d
    report["macro"] = {"cer": num / max(1, den),
                       "n_regions": sum(len(p) for p in per_lang.values())}
    return report


def _batch(proc, model, crop, gold):
    """One training example: loss on the target text only, prompt masked out."""
    prompt = _prompt_text(proc)
    full = proc(images=crop, text=prompt + gold, return_tensors="pt")
    prompt_len = proc(images=crop, text=prompt,
                      return_tensors="pt")["input_ids"].shape[1]
    full = {k: v.to(model.device) for k, v in full.items()}
    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100          # do not train on the prompt
    full["labels"] = labels
    return full


# How the train/eval split was obtained. This has to travel WITH the numbers:
# the results file carried the family-held-out sentence into the XFUND runs,
# where it is simply false -- XFUND has no generated PII at all -- and a reader
# would have credited those CER drops with a generalisation guarantee they do
# not have.
FAMILY_SPLIT_NOTE = (
    "Train and eval use disjoint PII generator families, so a CER drop cannot "
    "be memorisation of the values.")
HALVED_SPLIT_NOTE = (
    "Train and eval are language-stratified halves of ONE document population "
    "(this corpus ships a single split), so a CER drop shows adaptation helps "
    "on these scans, NOT that it generalises to unseen values.")
CROSS_CORPUS_NOTE = (
    "Train and eval come from DIFFERENT corpora ({train} -> {eval}): different "
    "documents, rendering/scanning processes, and value distributions. The "
    "strongest held-out in this project.")
_WER_NOTE = " WER is null for ko/ja/zh: whitespace word error is meaningless."


def _train_one_seed(proc, model, train_items, eval_items, seed, epochs, lr,
                    lora_r, lora_alpha, pairs_out=None, use_dora=False,
                    use_rslora=False, init_lora_weights=True, use_vera=False):
    """Attach a fresh adapter, train it, and evaluate. Mutates ``model``."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    base_dtype = next(model.parameters()).dtype
    # "proj" also names the vision patch-embed Conv3d. Plain LoRA/DoRA wrap
    # conv layers fine; PiSSA init and VeRA support only nn.Linear, so they
    # get the linear-only target list.
    linear_only = use_vera or (isinstance(init_lora_weights, str)
                               and init_lora_weights.startswith("pissa"))
    targets = _linear_targets(model) if linear_only else _targets(model)
    if use_vera:
        # VeRA shares one frozen random (A, B) pair across layers and trains
        # only per-layer scaling vectors — the extreme of the "less capacity
        # is better" trend from the rank ablation.
        from peft import VeraConfig
        peft_cfg = VeraConfig(r=lora_r, vera_dropout=0.05,
                              target_modules=targets)
    else:
        peft_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
            use_dora=use_dora, use_rslora=use_rslora,
            init_lora_weights=init_lora_weights,
            target_modules=targets)
    model = get_peft_model(model, peft_cfg)
    # peft initialises DoRA magnitudes (and some other variants' extras) in
    # float32, which crashes a bf16 forward ("Input type FloatTensor and
    # weight type BFloat16Type"). Casting to the base dtype is harmless for
    # plain LoRA, so do it unconditionally.
    model = model.to(base_dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    history = []
    for ep in range(epochs):
        total = 0.0
        for i, (crop, gold, _lang) in enumerate(train_items):
            if i % 200 == 0:
                print(f"  seed {seed} epoch {ep} step {i}/{len(train_items)}",
                      flush=True)
            opt.zero_grad()
            loss = model(**_batch(proc, model, crop, gold)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            total += float(loss)
        history.append(total / len(train_items))
    return (evaluate_cer(proc, model, eval_items, pairs_out=pairs_out),
            history, trainable)


def _mean_std(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {"mean": mean, "std": var ** 0.5, "n": n}


def finetune_and_measure(train_docs, eval_docs, data_dir, epochs=2,
                         max_docs=60, eval_max_docs=40, max_regions=24,
                         lr=1e-4, lora_r=16, lora_alpha=32, device="cuda",
                         dtype="bfloat16", seed=0, seeds=None, model_id=None,
                         eval_data_dir=None, use_dora=False, use_rslora=False,
                         init_lora_weights=True, use_vera=False,
                         split_note=FAMILY_SPLIT_NOTE):
    """Baseline CER -> LoRA adaptation -> adapted CER, paired, over N seeds.

    Seeds are not optional decoration here. On XFUND, ja appears in two runs on
    an IDENTICAL eval half and its CER delta was -0.569 next to es but -0.437
    next to zh, with the exact-match delta changing sign — so a single-seed,
    single-partner per-language number is not evidence of anything.

    The baseline is evaluated ONCE, outside the seed loop: no LoRA is attached
    yet and decoding is greedy, so it is seed-independent by construction and
    re-running it would cost ~15 min per seed for an identical answer.
    """
    seeds = list(seeds) if seeds else [seed]
    proc, model = _load(device, dtype, model_id)

    eval_items = list(_regions(eval_docs, eval_data_dir or data_dir,
                               eval_max_docs, max_regions))
    train_items = list(_regions(train_docs, data_dir, max_docs, max_regions))
    if not eval_items or not train_items:
        raise ValueError("no non-empty regions found — check data_dir")

    print(f"regions: train {len(train_items)}, eval {len(eval_items)}", flush=True)
    before_pairs = []
    before = evaluate_cer(proc, model, eval_items, pairs_out=before_pairs)
    langs = sorted(set(before) - {"macro"})

    per_seed = []
    for i, s in enumerate(seeds):
        if i:
            # get_peft_model injects adapter layers into the base model in
            # place, so seed i+1 would otherwise train on top of seed i's
            # weights. Reloading costs ~30s against a ~50 min run.
            del model
            torch.cuda.empty_cache()
            proc, model = _load(device, dtype, model_id)
        after_pairs = []
        after, history, trainable = _train_one_seed(
            proc, model, train_items, eval_items, s, epochs, lr, lora_r,
            lora_alpha, pairs_out=after_pairs, use_dora=use_dora,
            use_rslora=use_rslora, init_lora_weights=init_lora_weights,
            use_vera=use_vera)
        per_seed.append({
            "seed": s, "loss_history": history, "cer_after": after,
            "eval_pairs": after_pairs,
            # Negative delta = adaptation helped.
            "cer_delta": {l: after[l]["cer"] - before[l]["cer"]
                          for l in langs if l in after},
            "cer_delta_macro": after["macro"]["cer"] - before["macro"]["cer"],
            "exact_match_delta": {l: after[l]["exact_match"]
                                  - before[l]["exact_match"]
                                  for l in langs if l in after},
        })

    return {
        "seeds": seeds,
        "n_train_regions": len(train_items),
        "n_eval_regions": len(eval_items),
        "lora_trainable_params": trainable,
        "cer_before": before,
        "eval_pairs_before": before_pairs,
        "per_seed": per_seed,
        "cer_delta_agg": {
            l: _mean_std([r["cer_delta"][l] for r in per_seed if l in r["cer_delta"]])
            for l in langs},
        "cer_delta_macro_agg": _mean_std([r["cer_delta_macro"] for r in per_seed]),
        "exact_match_delta_agg": {
            l: _mean_std([r["exact_match_delta"][l] for r in per_seed
                          if l in r["exact_match_delta"]])
            for l in langs},
        "note": split_note + _WER_NOTE,
    }


def _targets(model):
    """Leaf Linear names to wrap. Discovered, because they differ per family."""
    import torch.nn as nn
    names = {n.split(".")[-1] for n, m in model.named_modules()
             if isinstance(m, nn.Linear)}
    preferred = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj",
                 "fc1", "fc2", "gate_proj", "up_proj", "down_proj"}
    hit = sorted(names & preferred)
    if not hit:
        raise RuntimeError(f"no LoRA targets found; linear leaves were {names}")
    return hit


def _linear_targets(model):
    """_targets, minus any name that ALSO matches a non-Linear module.

    peft matches target names by suffix, so a name shared between a Linear
    and a conv (Qwen2-VL's "proj" is both the attention out-proj and the
    patch-embed Conv3d) wraps both — fatal for adapters that support only
    nn.Linear (PiSSA init, VeRA).
    """
    import torch.nn as nn
    hit = set(_targets(model))
    for n, m in model.named_modules():
        if n.split(".")[-1] in hit and not isinstance(m, nn.Linear):
            hit.discard(n.split(".")[-1])
    return sorted(hit)
