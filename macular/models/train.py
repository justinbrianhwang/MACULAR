"""Training utilities for the MACULAR core (CPU-runnable smoke training).

``train_step`` runs one optimizer step; ``fit_synthetic`` fits the model on the
learnable synthetic task and returns the loss history — used both as a smoke
test and as the template for the real training loop (proposal 11.13 staged
training) once a real BackboneAdapter is plugged in.
"""

from __future__ import annotations

import json

import torch

from .core import MacularModel, MacularConfig, macular_loss
from .backbone import make_synthetic_batch


def train_step(model, optimizer, batch, weights=None, recon_target=None,
               adv_lambda=1.0):
    weights = weights or {}
    feats, boxes, pii_labels, clin_labels, mask = batch
    optimizer.zero_grad()
    out = model(feats, boxes, key_padding_mask=~mask, adv_lambda=adv_lambda)
    loss, parts = macular_loss(out, pii_labels, clin_labels, valid_mask=mask,
                               recon_target=recon_target, **weights)
    loss.backward()
    optimizer.step()
    model.update_teacher()
    return parts


def warmup(step: int, total: int, target: float, frac: float = 0.3) -> float:
    """Linear warm-up (proposal 11.12).

    The consistency and adversarial terms must not be applied at full strength
    from step 0: early on the EMA teacher is still poor, so a strong KL drags
    the student toward wrong targets and costs clinical utility. Measured: a
    constant w_cons=1.0 left clinical F1 at 0.798 vs 0.956 without consistency.
    """
    if total <= 0:
        return target
    n = max(1, int(total * frac))
    return target * min(1.0, step / n)


def fit_synthetic(steps=60, cfg=None, lr=1e-3, seed=0):
    cfg = cfg or MacularConfig()
    torch.manual_seed(seed)
    model = MacularModel(cfg)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    batch = make_synthetic_batch(cfg, seed=seed)
    history = []
    for _ in range(steps):
        parts = train_step(model, opt, batch)
        history.append(parts["total"])
    return model, history


def _pii_macro_f1(model, batch):
    """PII-presence metrics (class>0 vs 0) over valid regions.

    Reports the thresholded P/R/F1 *and* threshold-free AP/AUC. The latter two
    matter because runs can land on very different operating points (e.g. VLM
    features tend toward high recall / low precision); comparing F1 across
    different operating points is not a fair comparison, while AP/AUC ranks the
    underlying scores independently of where the threshold sits (proposal 18.8).
    """
    feats, boxes, pii, _clin, mask = batch
    model.eval()
    with torch.no_grad():
        out = model(feats, boxes, key_padding_mask=~mask)
    model.train()
    return _pii_metrics_from(out["pii_logits"][mask], pii[mask])


def _pii_metrics_from(logits, gold):
    """Same metrics, from already-collected logits/labels (LoRA path reuses this
    so there is only one implementation of the PII scoring rule)."""
    import torch.nn.functional as F

    pred = logits.argmax(-1)
    y = (gold > 0).int()
    # P(any PII) = 1 - P(NON_PII)
    score = 1.0 - F.softmax(logits, dim=-1)[:, 0]

    tp = int(((pred > 0) & (gold > 0)).sum())
    fp = int(((pred > 0) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold > 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    ap = auc = None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        yn, sn = y.cpu().numpy(), score.detach().cpu().numpy()
        if yn.min() != yn.max():          # both classes present
            ap = float(average_precision_score(yn, sn))
            auc = float(roc_auc_score(yn, sn))
    except Exception:
        pass
    return {"precision": prec, "recall": rec, "f1": f1,
            "average_precision": ap, "roc_auc": auc,
            "positive_rate": float(y.float().mean())}


def fit_on_documents(train_docs, val_docs=None, epochs=15, lr=2e-3, seed=0,
                     source="gt", engine=None, data_dir="", max_docs=200,
                     max_regions=48, cer=0.0, cache_engine=""):
    """Train the model core on REAL Documents via OCR-derived features."""
    from .features import documents_to_batch, config_for_features

    torch.manual_seed(seed)
    cfg = config_for_features()
    model = MacularModel(cfg)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    train_batch = documents_to_batch(
        train_docs[:max_docs], max_regions=max_regions,
        source=source, engine=engine, data_dir=data_dir, cer=cer, seed=seed,
        cache_engine=cache_engine)

    # Cost-sensitive PII loss (proposal 11.4). SQRT of inverse frequency, not raw
    # inverse frequency: raw inverse weights make NON_PII so cheap that the model
    # collapses to predicting PII everywhere (recall 1.0, precision = base rate)
    # and stops using the text at all. Sqrt keeps the recall bias the proposal
    # wants while still forcing the model to learn real features.
    _, _, pii_labels, _, mask = train_batch
    counts = torch.bincount(pii_labels[mask], minlength=cfg.n_pii_classes).float()
    pii_weight = (counts.sum() / (cfg.n_pii_classes * counts.clamp(min=1)))
    pii_weight = pii_weight.sqrt().clamp(max=5.0)
    weights = {"pii_weight": pii_weight}

    history = []
    for _ in range(epochs):
        parts = train_step(model, opt, train_batch, weights=weights)
        history.append(parts["total"])

    result = {"loss_history": history,
              "train_pii_f1": _pii_macro_f1(model, train_batch)}
    if val_docs:
        val_batch = documents_to_batch(
            val_docs[:max_docs], max_regions=max_regions,
            source=source, engine=engine, data_dir=data_dir,
            cer=cer, seed=seed + 1, cache_engine=cache_engine)
        result["val_pii_f1"] = _pii_macro_f1(model, val_batch)
    return model, result


def ocr_error_propagation(train_docs, val_docs, cer_levels=(0.0, 0.1, 0.2, 0.3, 0.5),
                          epochs=40, lr=3e-3, seed=0, max_docs=120):
    """Measure how OCR quality propagates into downstream PII performance.

    Trains/evaluates the core at several simulated character error rates. This
    quantifies the cascade the proposal's problem statement is built on
    (section 2: OCR errors propagate into PII detection and clinical fields).

    IMPORTANT — run this on COUNTERFACTUAL-layout documents. On the default
    layout, PII sits in a fixed column, so the model solves the task from
    geometry alone and the curve is flat no matter how badly the text is
    corrupted (measured: flat 0.941 across CER 0.0-0.5). Only after the
    positional shortcut is removed does text quality actually matter.
    """
    rows = []
    for cer in cer_levels:
        _m, res = fit_on_documents(train_docs, val_docs, epochs=epochs, lr=lr,
                                   seed=seed, source="noisy", cer=cer,
                                   max_docs=max_docs)
        rows.append({
            "cer": cer,
            "loss_end": res["loss_history"][-1],
            "val_pii_f1": res["val_pii_f1"]["f1"],
            "val_pii_recall": res["val_pii_f1"]["recall"],
            "val_pii_precision": res["val_pii_f1"]["precision"],
        })
    return rows


def engine_downstream_comparison(train_docs, val_docs, engines, data_dir,
                                 epochs=80, lr=3e-3, seed=0, max_docs=120):
    """Do STRONGER OCR ENGINES yield better downstream PII detection?

    Runs each real engine over the region crops, trains the core on the text it
    produced, and reports downstream PII metrics — plus a "gt" upper bound
    (perfect text). This turns "PP-OCR beats Tesseract on CER" into the question
    that actually matters for the proposal: does it make de-identification safer?

    Must be run on COUNTERFACTUAL-layout documents; otherwise the model answers
    from geometry and every engine ties (see ocr_error_propagation).
    """
    from ..baselines.ocr import ENGINES

    rows = []
    for name in engines:
        if name == "gt":
            _m, res = fit_on_documents(train_docs, val_docs, epochs=epochs,
                                       lr=lr, seed=seed, source="gt",
                                       data_dir=data_dir, max_docs=max_docs)
            rows.append({"engine": "gt (perfect text)", "available": True,
                         **_row(res)})
            continue
        # Prefer a prebuilt cache (required for engines that cannot share a
        # process with torch, e.g. PaddleOCR); fall back to running live.
        from ..baselines.ocr_cache import cache_path
        import os as _os
        if _os.path.exists(cache_path(data_dir, name)):
            _m, res = fit_on_documents(train_docs, val_docs, epochs=epochs,
                                       lr=lr, seed=seed, source="cache",
                                       cache_engine=name, data_dir=data_dir,
                                       max_docs=max_docs)
            rows.append({"engine": name, "available": True, "from_cache": True,
                         **_row(res)})
            continue
        cls = ENGINES.get(name)
        eng = cls() if cls else None
        if eng is None or not eng.available():
            rows.append({"engine": name, "available": False,
                         "reason": "engine not installed and no OCR cache found"})
            continue
        _m, res = fit_on_documents(train_docs, val_docs, epochs=epochs, lr=lr,
                                   seed=seed, source="ocr", engine=eng,
                                   data_dir=data_dir, max_docs=max_docs)
        rows.append({"engine": name, "available": True, "from_cache": False,
                     **_row(res)})
    return rows


def _clinical_metrics(model, batch):
    """Utility side of the trade-off: how well the SAFE view still recovers
    clinical fields (proposal 18.6). Reported as macro-F1 over the non-'NONE'
    clinical classes, so it is not dominated by the empty class."""
    feats, boxes, _pii, clin, mask = batch
    model.eval()
    with torch.no_grad():
        out = model(feats, boxes, key_padding_mask=~mask)
    model.train()
    return _clinical_metrics_from(out["clinical_student"].argmax(-1)[mask],
                                  clin[mask])


def _clinical_metrics_from(pred, gold):
    """Same metric, from already-collected predictions (LoRA path reuses this)."""
    f1s = []
    for c in sorted(set(gold.tolist())):
        if c == 0:                       # skip the "NONE" class
            continue
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    acc = float((pred == gold).float().mean())
    return {"macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
            "accuracy": acc, "n_classes": len(f1s)}


def fit_with_vlm(train_docs, val_docs, backbone, data_dir, epochs=80, lr=3e-3,
                 seed=0, max_docs=120, max_regions=48, model_cfg=None,
                 loss_weights=None, batches=None, recon=None, eraser=None):
    """Train the core on REAL VLM vision features (A2 track, proposal 11.2).

    ``eraser``: a fitted LEACE eraser (macular.privacy.fit_leace) to use INSTEAD
    of the redaction gate. Must have been fit on train features only.
    """
    from .features import documents_to_vlm_batch, config_for_features

    torch.manual_seed(seed)
    if batches is None:
        cache: dict = {}
        stats: dict = {}      # normalization fitted on TRAIN, reused for val
        train_batch = documents_to_vlm_batch(train_docs[:max_docs], backbone,
                                             data_dir, max_regions, cache,
                                             stats=stats)
        val_batch = documents_to_vlm_batch(val_docs[:max_docs], backbone,
                                           data_dir, max_regions, cache,
                                           stats=stats)
    else:
        train_batch, val_batch = batches
    recon_train, recon_val = (recon if recon else (None, None))

    d_in = train_batch[0].shape[-1]
    cfg = config_for_features(d_in=d_in)
    for k, v in (model_cfg or {}).items():    # ablation switches
        setattr(cfg, k, v)
    model = MacularModel(cfg)
    if eraser is not None:
        model.set_eraser(eraser)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    _, _, pii_labels, _, mask = train_batch
    counts = torch.bincount(pii_labels[mask], minlength=cfg.n_pii_classes).float()
    w = (counts.sum() / (cfg.n_pii_classes * counts.clamp(min=1))).sqrt().clamp(max=5.0)
    base = {"pii_weight": w, **(loss_weights or {})}
    w_cons_target = base.pop("w_cons", 1.0)
    w_adv_target = base.pop("w_adv", 1.0)

    history = []
    for step in range(epochs):
        weights = {**base,
                   "w_cons": warmup(step, epochs, w_cons_target),
                   "w_adv": warmup(step, epochs, w_adv_target)}
        history.append(train_step(model, opt, train_batch, weights=weights,
                                  recon_target=recon_train)["total"])

    res = {"loss_history": history, "d_in": int(d_in),
           "train_pii_f1": _pii_macro_f1(model, train_batch),
           "val_pii_f1": _pii_macro_f1(model, val_batch),
           "val_clinical": _clinical_metrics(model, val_batch)}
    if recon_val is not None:
        res["leakage"] = evaluate_leakage(model, val_batch, recon_val, seed=seed)
    return model, res


def backbone_contribution(train_docs, val_docs, backbone, data_dir, epochs=80,
                          lr=3e-3, seed=0, max_docs=120, seeds=(0, 1, 2)):
    """RQ6 / gate #3: does the VLM backbone actually contribute?

    Compares A1 (text-only features: what the shared parser already gives) with
    A2 (the backbone's own visual features) under identical training. The
    proposal requires Delta > 0 BEFORE spending GPU on full training; if it is
    ~0, the backbone is dead weight and that must be reported.

    Delta is measured on AVERAGE PRECISION, not F1: the two feature sets settle
    on different operating points (VLM features skew to high recall), and
    comparing F1 across different operating points is not a fair test. AP ranks
    the underlying scores threshold-free. Repeated over seeds so a difference
    can be told apart from run-to-run noise.
    """
    a1_ap, a2_ap, a1_f1, a2_f1 = [], [], [], []
    a1_last = a2_last = None
    for s in seeds:
        _m1, a1 = fit_on_documents(train_docs, val_docs, epochs=epochs, lr=lr,
                                   seed=s, source="gt", data_dir=data_dir,
                                   max_docs=max_docs)
        _m2, a2 = fit_with_vlm(train_docs, val_docs, backbone, data_dir,
                               epochs=epochs, lr=lr, seed=s, max_docs=max_docs)
        a1_last, a2_last = a1, a2
        a1_ap.append(a1["val_pii_f1"]["average_precision"])
        a2_ap.append(a2["val_pii_f1"]["average_precision"])
        a1_f1.append(a1["val_pii_f1"]["f1"])
        a2_f1.append(a2["val_pii_f1"]["f1"])

    def _stat(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None, None
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return m, sd

    a1m, a1s = _stat(a1_ap)
    a2m, a2s = _stat(a2_ap)
    d_ap = (a2m - a1m) if (a1m is not None and a2m is not None) else None
    f1m_1, _ = _stat(a1_f1)
    f1m_2, _ = _stat(a2_f1)
    # only call it a contribution if the mean gain clears the seed noise
    noise = max(a1s or 0.0, a2s or 0.0)
    contributes = bool(d_ap is not None and d_ap > noise)
    return {
        "seeds": list(seeds),
        "A1_text_only": {**_row(a1_last), "ap_mean": a1m, "ap_std": a1s,
                         "f1_mean": f1m_1},
        "A2_vlm_features": {**_row(a2_last), "d_in": a2_last["d_in"],
                            "ap_mean": a2m, "ap_std": a2s, "f1_mean": f1m_2},
        "delta_ap": d_ap,
        "delta_f1": (f1m_2 - f1m_1) if (f1m_1 is not None) else None,
        "seed_noise": noise,
        "backbone_contributes": contributes,
        "criterion": "delta_ap > max(seed std) on average precision",
    }


def fit_with_lora(train_docs, val_docs, backbone, data_dir, epochs=30, lr=3e-3,
                  lora_lr=1e-4, seed=0, max_docs=40, max_regions=48,
                  model_cfg=None, loss_weights=None, recon=None,
                  eval_max_docs=None):
    """Train MACULAR **with the vision tower trainable via LoRA**.

    The frozen-backbone ablation found the gate cost utility with no measurable
    privacy gain — but a frozen backbone can only be reweighted, never reshaped,
    so the gate/adversary had no way to remove content. Here the backbone runs
    inside the training loop (proposal 11.13 Stage 1), so gradients from the
    gate and the adversary reach the representation itself.

    Far more expensive than the cached path: one vision forward per document per
    step. Keep max_docs/epochs small.
    """
    import os
    from PIL import Image
    from .features import (config_for_features, PII_TO_IDX, CLIN_TO_IDX,
                           NON_PII, char_features, CHAR_DIM)

    torch.manual_seed(seed)
    docs = train_docs[:max_docs]
    # Leakage evaluation needs many more PII regions than training needs docs:
    # the held-out attacker has to fit a regression, and with only ~180 PII
    # regions it underperforms the type-mean baseline and measures nothing.
    # Evaluation is forward-only, so it can cover more documents cheaply.
    vdocs = val_docs[:(eval_max_docs or max_docs)]

    def prep(dlist):
        out = []
        for doc in dlist:
            cands = doc.candidates[:max_regions]
            img = Image.open(os.path.join(data_dir, doc.image_path)).convert("RGB")
            bx = torch.tensor([c.bbox.as_list() for c in cands], dtype=torch.float32)
            pii = torch.tensor([[PII_TO_IDX.get(c.pii_type or NON_PII, 0)
                                 for c in cands]])
            clin = torch.tensor([[CLIN_TO_IDX.get(c.clinical_type or "NONE", 0)
                                  for c in cands]])
            rec = torch.stack([char_features(c.text) for c in cands]).unsqueeze(0)
            out.append((img, bx, pii, clin, rec))
        return out

    train_items, val_items = prep(docs), prep(vdocs)

    cfg = config_for_features(d_in=backbone.d_in)
    for k, v in (model_cfg or {}).items():
        setattr(cfg, k, v)
    model = MacularModel(cfg)
    dev = next(backbone._model.parameters()).device
    model.to(dev)

    base = {**(loss_weights or {})}
    w_cons_t = base.pop("w_cons", 1.0)
    w_adv_t = base.pop("w_adv", 1.0)

    opt = torch.optim.Adam([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr},
        {"params": backbone.trainable_parameters(), "lr": lora_lr},
    ])

    history = []
    for step in range(epochs):
        tot = 0.0
        for img, bx, pii, clin, rec in train_items:
            feats = backbone.encode_page(img, bx).unsqueeze(0)     # (1,N,d_in)
            mask = torch.ones(1, feats.shape[1], dtype=torch.bool, device=dev)
            weights = {**base,
                       "w_cons": warmup(step, epochs, w_cons_t),
                       "w_adv": warmup(step, epochs, w_adv_t)}
            opt.zero_grad()
            out = model(feats, bx.unsqueeze(0).to(dev), key_padding_mask=~mask)
            loss, parts = macular_loss(out, pii.to(dev), clin.to(dev),
                                       valid_mask=mask,
                                       recon_target=rec.to(dev), **weights)
            loss.backward()
            # Gradient clipping is not optional here. Without it, 2 of 9 runs in
            # a 3-seed x 3-variant sweep collapsed outright (PII average
            # precision 0.15, i.e. below the base rate, and clinical macro-F1
            # exactly 0.0) while other seeds of the SAME config reached 0.99.
            # Single-batch steps on 24 documents produce occasional huge
            # gradients that destroy the LoRA adapter in one update.
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] +
                list(backbone.trainable_parameters()), max_norm=1.0)
            opt.step()
            model.update_teacher()
            tot += parts["total"]
        history.append(tot / max(1, len(train_items)))

    # Evaluate on val: collect representations with the trained LoRA backbone.
    # The privacy side (leakage) and the utility side (clinical macro-F1 read off
    # the SAFE view) must come from the SAME forward pass, otherwise a variant
    # can look private simply because it destroyed the representation. Gate #4 of
    # the proposal is a trade-off question, so both axes are collected here.
    model.eval()
    zs, zr, ys, pl = [], [], [], []
    plog, cpred, cgold = [], [], []
    with torch.no_grad():
        for img, bx, pii, clin, rec in val_items:
            feats = backbone.encode_page(img, bx).unsqueeze(0)
            mask = torch.ones(1, feats.shape[1], dtype=torch.bool, device=dev)
            out = model(feats, bx.unsqueeze(0).to(dev), key_padding_mask=~mask)
            zs.append(out["z_safe"][0].cpu())
            zr.append(model.projector(feats)[0].cpu())
            ys.append(rec[0])
            pl.append(pii[0])
            plog.append(out["pii_logits"][0].float().cpu())
            cpred.append(out["clinical_student"].argmax(-1)[0].cpu())
            cgold.append(clin[0])
    model.train()
    pii_all = torch.cat(pl)
    vp = _pii_metrics_from(torch.cat(plog), pii_all)
    vc = _clinical_metrics_from(torch.cat(cpred), torch.cat(cgold))
    # A collapsed run must be visible, not silently averaged in. "Collapsed"
    # means the model did worse than a random ranker at PII, or never predicted
    # a single clinical class right. Averaging such runs together with healthy
    # ones is what produced a std larger than the mean in the first sweep.
    ap = vp.get("average_precision")
    collapsed = bool(vc["macro_f1"] == 0.0 or
                     (ap is not None and ap <= vp["positive_rate"]))
    return model, {
        "loss_history": history,
        "eval": (torch.cat(zs), torch.cat(zr), torch.cat(ys), pii_all),
        "val_pii": vp, "val_clinical": vc, "collapsed": collapsed,
    }


def leakage_from_reps(z_safe, z_raw, recon, pii, epochs=150, lr=1e-2, seed=0,
                      hidden=256):
    """Held-out attacker on already-collected representations (LoRA path)."""
    import torch.nn as nn

    torch.manual_seed(seed)
    is_pii = pii > 0
    if int(is_pii.sum()) < 20:
        return None
    zs, zr, yy, tt = z_safe[is_pii], z_raw[is_pii], recon[is_pii], pii[is_pii]
    n = zs.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = int(n * 0.7)
    tr, te = perm[:cut], perm[cut:]

    type_mean = torch.zeros_like(yy[te])
    for t in tt.unique():
        sel = tt[tr] == t
        type_mean[tt[te] == t] = yy[tr][sel].mean(0) if sel.any() else yy[tr].mean(0)
    base = float(torch.nn.functional.cosine_similarity(type_mean, yy[te], dim=-1).mean())

    def attack(z):
        net = nn.Sequential(nn.Linear(z.shape[-1], hidden), nn.GELU(),
                            nn.Linear(hidden, hidden), nn.GELU(),
                            nn.Linear(hidden, yy.shape[-1]))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(z[tr]), yy[tr]).backward()
            opt.step()
        with torch.no_grad():
            pred = net(z[te])
        cos = float(torch.nn.functional.cosine_similarity(pred, yy[te], dim=-1).mean())
        return {"cosine": cos, "identity_leakage": cos - base}

    return {"safe": attack(zs.detach()), "raw_unprotected": attack(zr.detach()),
            "type_baseline_cosine": base, "n_pii_regions": int(n)}


def evaluate_leakage(model, batch, recon_target, epochs=150, lr=1e-2, seed=0,
                     hidden=256):
    """HELD-OUT attacker: how much original PII content survives in z_safe?

    The proposal (11.9) is explicit that the in-training adversary proves
    nothing — it shares weights and objective with the model. Privacy must be
    judged by a separate attacker with a DIFFERENT architecture, trained
    post-hoc on FROZEN representations, and evaluated on regions it never saw.

    Returns the attacker's reconstruction quality on held-out PII regions:
      cosine  -- cosine similarity to the true text signature (1.0 = full leak)
      r2      -- coefficient of determination
    Lower is better for privacy. Also reports the same attack against the RAW
    (ungated) representation as the no-protection reference point.
    """
    import torch.nn as nn

    torch.manual_seed(seed)
    feats, boxes, pii, _clin, mask = batch
    model.eval()
    with torch.no_grad():
        out = model(feats, boxes, key_padding_mask=~mask)
        z_safe = out["z_safe"][mask]
        z_raw = model.projector(feats)[mask]
    model.train()

    y = recon_target.reshape(-1, recon_target.shape[-1])[mask.reshape(-1)]
    is_pii = pii[mask] > 0
    if int(is_pii.sum()) < 20:
        return None

    # Attack only PII regions; split them so the attacker is scored on regions
    # it never trained on.
    zs, zr, yy = z_safe[is_pii], z_raw[is_pii], y[is_pii]
    n = zs.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    cut = int(n * 0.7)
    tr, te = perm[:cut], perm[cut:]

    def attack(z):
        # deliberately a different architecture from PrivacyAdversary (deeper,
        # wider, GELU) so success is not an artifact of a shared inductive bias
        net = nn.Sequential(nn.Linear(z.shape[-1], hidden), nn.GELU(),
                            nn.Linear(hidden, hidden), nn.GELU(),
                            nn.Linear(hidden, yy.shape[-1]))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        zt, yt = z[tr].detach(), yy[tr].detach()
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(zt), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = net(z[te].detach())
        gold = yy[te]
        cos = float(torch.nn.functional.cosine_similarity(pred, gold, dim=-1).mean())
        ss_res = float(((pred - gold) ** 2).sum())
        ss_tot = float(((gold - gold.mean(0)) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"cosine": cos, "r2": r2}

    # Type-only reference. The gate KEEPS the PII type by design, and a text
    # signature is largely determined by its type (all phone numbers look alike
    # to a char histogram). Measured: predicting the per-type mean already
    # scores 0.737 cosine, ABOVE what the attacker gets from either
    # representation — so raw cosine measures type predictability, not identity
    # leakage. What matters is how far the attacker beats the type baseline.
    type_ids = pii[mask][is_pii]
    yy_te = yy[te]
    type_mean = torch.zeros_like(yy_te)
    for t in type_ids.unique():
        sel_tr = (type_ids[tr] == t)
        if sel_tr.any():
            mu = yy[tr][sel_tr].mean(0)
        else:
            mu = yy[tr].mean(0)
        type_mean[type_ids[te] == t] = mu
    base_cos = float(torch.nn.functional.cosine_similarity(
        type_mean, yy_te, dim=-1).mean())

    safe, raw = attack(zs), attack(zr)
    for name, res in (("safe", safe), ("raw_unprotected", raw)):
        # >0 means the representation reveals WHICH value it was, beyond type
        res["identity_leakage"] = res["cosine"] - base_cos
    return {"safe": safe, "raw_unprotected": raw,
            "type_baseline_cosine": base_cos,
            "n_pii_regions": int(n), "n_attack_test": int(len(te))}


# MACULAR ablations on real VLM features (proposal 17.1). Each entry is
# (label, model-config overrides, loss-weight overrides).
MACULAR_ABLATIONS = [
    ("full",            {},                            {}),
    ("hard_mask",       {"hard_mask": True},           {}),
    ("no_consistency",  {},                            {"w_cons": 0.0}),
    ("no_graph",        {"use_graph": False},          {}),
    ("no_adversary",    {"use_adversary": False},      {"w_adv": 0.0}),
    ("no_gate",         {"hard_mask": True},           {"w_cons": 0.0}),
]


def build_recon_targets(docs, max_regions=48, max_docs=None):
    """Per-region signature of the ORIGINAL text — what a leakage attacker
    tries to recover from the safe representation (proposal 11.9)."""
    from .features import char_features, CHAR_DIM

    subset = docs[:max_docs] if max_docs else docs
    out = torch.zeros(len(subset), max_regions, CHAR_DIM)
    for b, doc in enumerate(subset):
        for i, c in enumerate(doc.candidates[:max_regions]):
            out[b, i] = char_features(c.text)
    return out


def macular_ablation(train_docs, val_docs, backbone, data_dir, epochs=80,
                     lr=3e-3, max_docs=120, seeds=(0, 1, 2), variants=None):
    """Do MACULAR's own components help, on REAL VLM features?

    The backbone gate (RQ6) only showed the features are informative. This asks
    the research question: does the differentiable redaction gate + dual-view
    consistency + relation graph actually improve the privacy/utility trade-off
    over the sequential alternatives (proposal 17.1)?

    Privacy is scored with average precision (threshold-free; see
    backbone_contribution for why F1 misleads here) and utility with clinical
    macro-F1 on the SAFE view.
    """
    from .features import documents_to_vlm_batch

    variants = variants or MACULAR_ABLATIONS
    # Encode once with the frozen backbone; every variant reuses the tensors so
    # the comparison isolates the method, not the feature extraction.
    cache: dict = {}
    stats: dict = {}
    train_batch = documents_to_vlm_batch(train_docs[:max_docs], backbone,
                                         data_dir, 48, cache, stats=stats)
    val_batch = documents_to_vlm_batch(val_docs[:max_docs], backbone,
                                       data_dir, 48, cache, stats=stats)
    recon = (build_recon_targets(train_docs, max_docs=max_docs),
             build_recon_targets(val_docs, max_docs=max_docs))

    rows = []
    for label, mcfg, lw in variants:
        aps, clins, leaks, raws = [], [], [], []
        last = None
        for s in seeds:
            _m, res = fit_with_vlm(None, None, backbone, data_dir, epochs=epochs,
                                   lr=lr, seed=s, model_cfg=mcfg,
                                   loss_weights=lw,
                                   batches=(train_batch, val_batch),
                                   recon=recon)
            last = res
            ap = res["val_pii_f1"]["average_precision"]
            if ap is not None:
                aps.append(ap)
            clins.append(res["val_clinical"]["macro_f1"])
            lk = res.get("leakage")
            if lk:
                leaks.append(lk["safe"]["identity_leakage"])
                raws.append(lk["raw_unprotected"]["identity_leakage"])

        def _ms(xs):
            if not xs:
                return None, None
            m = sum(xs) / len(xs)
            return m, (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

        ap_m, ap_s = _ms(aps)
        cl_m, cl_s = _ms(clins)
        lk_m, lk_s = _ms(leaks)
        raw_m, _ = _ms(raws)
        rows.append({"variant": label,
                     "pii_detection_ap_mean": ap_m, "pii_detection_ap_std": ap_s,
                     "clinical_f1_mean": cl_m, "clinical_f1_std": cl_s,
                     # THE privacy number: how far a held-out attacker beats the
                     # PII-type baseline, i.e. identity leakage (>0 = the safe
                     # representation still reveals WHICH value it was).
                     "identity_leakage_mean": lk_m, "identity_leakage_std": lk_s,
                     "identity_leakage_raw_reference": raw_m,
                     "val_pii_recall": last["val_pii_f1"]["recall"],
                     "loss_end": last["loss_history"][-1]})

    full = next((r for r in rows if r["variant"] == "full"), None)
    if full:
        for r in rows:
            if r["variant"] == "full":
                continue
            r["delta_clinical_vs_full"] = full["clinical_f1_mean"] - r["clinical_f1_mean"]
            if (full["identity_leakage_mean"] is not None
                    and r["identity_leakage_mean"] is not None):
                # positive => full MACULAR leaks LESS than this variant
                r["leakage_reduction_by_full"] = (
                    r["identity_leakage_mean"] - full["identity_leakage_mean"])
    return rows


ERASURE_MECHANISMS = [
    # (label, redaction mode, needs a fitted LEACE eraser)
    ("none",       "none",  False),   # unprotected upper bound on leakage
    ("hard_mask",  "hard",  False),   # the privacy FLOOR every method must beat
    ("gate",       "gate",  False),   # MACULAR's differentiable gate
    ("leace",      "leace", True),    # closed-form linear concept erasure
]


def _val_region_texts(docs, max_docs, max_regions):
    """Region texts aligned with documents_to_vlm_batch's flattened valid mask."""
    out = []
    for doc in docs[:max_docs]:
        cands = doc.candidates[:max_regions]
        out.extend(c.text or "" for c in cands)
        out.extend([""] * (max_regions - len(cands)))     # padding slots
    return out


def erasure_comparison(train_docs, val_docs, backbone, data_dir, epochs=80,
                       lr=3e-3, max_docs=120, seeds=(0, 1, 2), max_regions=48,
                       mechanisms=None):
    """Does closed-form erasure beat the gate AND the hard-mask floor?

    This is the experiment the gate ablation could not answer, run with the two
    fixes its failure demanded:

    * **Mechanism**: LEACE (closed form, provable against all linear readouts)
      replaces a gate that our own ablation showed adds nothing over a hard mask.
    * **Measurement**: discrete attacks — a fresh linear probe, a fresh NONLINEAR
      probe, and an inversion attack scored by exact-match/CER — instead of a
      cosine similarity whose run-to-run drift exceeded the effect it measured.

    The nonlinear probe is not optional. Linear erasure guards linear readouts by
    construction; reporting only the linear probe would restate the guarantee as
    if it were evidence.
    """
    from .features import documents_to_vlm_batch
    from ..privacy import fit_leace, probe_leakage, inversion_attack, prior_floor

    mechanisms = mechanisms or ERASURE_MECHANISMS
    cache: dict = {}
    stats: dict = {}
    train_batch = documents_to_vlm_batch(train_docs[:max_docs], backbone,
                                         data_dir, max_regions, cache,
                                         stats=stats)
    val_batch = documents_to_vlm_batch(val_docs[:max_docs], backbone, data_dir,
                                       max_regions, cache, stats=stats)
    recon = (build_recon_targets(train_docs, max_regions, max_docs),
             build_recon_targets(val_docs, max_regions, max_docs))

    tf, _, tp, _, tm = train_batch
    _vf, _, vp, _, vm = val_batch
    # LEACE is fit on TRAIN pooled region features only.
    eraser = fit_leace(tf[tm], tp[tm])
    fit_residual = eraser.residual_covariance(tf[tm], tp[tm])
    # The guarantee is proven on the fitting distribution. Whether it TRANSFERS
    # is an empirical question, and it is the first thing a reviewer will ask:
    # a covariance that is ~0 on train but not on val means the linear-guardedness
    # claim does not hold where it matters.
    val_residual = eraser.residual_covariance(_vf[vm], vp[vm])
    # Diagnostic that separates two very different failures, because the first
    # measured run showed LEACE leaving a linear probe at 0.916 (majority 0.848)
    # when the theorem says it should be at chance:
    #   (a) the erasure did not TRANSFER — train and val use DISJOINT PII
    #       generator families by design, so the concept subspace fitted on
    #       family A need not be the one carrying family B; or
    #   (b) the erasure transferred, but downstream training put the concept back.
    # Probing the erased features directly, before the projector and before any
    # training, tells them apart: high here => (a), low here => (b).
    leace_direct = probe_leakage(eraser(_vf[vm]), vp[vm], seed=0)
    raw_direct = probe_leakage(_vf[vm], vp[vm], seed=0)

    texts = _val_region_texts(val_docs, max_docs, max_regions)
    flat_valid = vm.reshape(-1)
    val_texts = [t for t, keep in zip(texts, flat_valid.tolist()) if keep]
    val_pii = vp[vm]
    pii_rows = val_pii > 0
    # Refuse to produce a table when there is nothing sensitive to protect. With
    # zero positive labels every probe scores 1.000 against a majority baseline
    # of 1.000 and the run looks successful — that exact failure was produced by
    # a label-name typo, so make it an error rather than a plausible result.
    if int(pii_rows.sum()) < 20:
        raise ValueError(
            f"only {int(pii_rows.sum())} sensitive regions in the evaluation "
            f"split — check the label mapping for this dataset. The erasure "
            f"comparison is meaningless without a positive class.")

    # How much the inverter recovers with NO information from the representation
    # (features shuffled against texts). Structured strings let a decoder guess
    # some exactly right from the prior alone, so this floor must be subtracted
    # before any recovery rate is called leakage. It depends only on the text
    # distribution and the attack, so it is computed once rather than per
    # mechanism.
    _pii_texts_all = [t for t, k in zip(val_texts, pii_rows.tolist()) if k]
    floors = {
        d: prior_floor(_vf[vm][pii_rows], _pii_texts_all, seed=0, decoder=d)
        for d in ("autoregressive", "linear")
    }

    rows = []
    for label, mode, needs_eraser in mechanisms:
        per_seed = []
        for s in seeds:
            _m, res = fit_with_vlm(None, None, backbone, data_dir, epochs=epochs,
                                   lr=lr, seed=s,
                                   model_cfg={"redaction": mode},
                                   batches=(train_batch, val_batch), recon=recon,
                                   eraser=eraser if needs_eraser else None)
            with torch.no_grad():
                out = _m(val_batch[0], val_batch[1], key_padding_mask=~vm)
                z = out["z_safe"][vm]
                z_ctx = out["z_ctx_safe"][vm]
            pii_texts = [t for t, k in zip(val_texts, pii_rows.tolist()) if k]
            # Attribute inference: can an attacker still read the PII TYPE?
            probes = probe_leakage(z, val_pii, seed=s)
            # Same attack after the relation graph — the representation that is
            # actually handed downstream. If this is higher than the pre-graph
            # number, redacting per-region is too late: context mixing puts the
            # content back.
            probes_ctx = probe_leakage(z_ctx, val_pii, seed=s)
            # Inversion: can an attacker read the literal VALUE back? Scored on
            # PII regions only — non-PII text is not a leak.
            inv = inversion_attack(z[pii_rows], pii_texts, seed=s)
            inv_ctx = inversion_attack(z_ctx[pii_rows], pii_texts, seed=s)
            # The same attack with a weaker decoder. Reporting the pair tells a
            # reviewer whether a defense only holds against weak attackers —
            # measured on raw features the stronger decoder lifts exact-match
            # recovery from 0.141 to 0.181, so the gap is not negligible.
            inv_weak = inversion_attack(z[pii_rows], pii_texts, seed=s,
                                        decoder="linear")
            per_seed.append({
                "seed": s,
                "clinical_macro_f1": res["val_clinical"]["macro_f1"],
                "pii_average_precision": res["val_pii_f1"]["average_precision"],
                "probe_linear_acc": probes["linear"]["accuracy"],
                "probe_mlp_acc": probes["mlp"]["accuracy"],
                "probe_majority_baseline": probes["majority_baseline"],
                "nonlinear_leakage": probes["nonlinear_leakage"],
                "inversion_exact_match": inv.get("exact_match"),
                "inversion_cer": inv.get("cer"),
                "inversion_weak_exact_match": inv_weak.get("exact_match"),
                "inversion_weak_cer": inv_weak.get("cer"),
                # Neither decoder is uniformly stronger, so the attacker gets to
                # pick; and recovery only counts above the prior floor.
                "inversion_best_exact_match": max(
                    inv.get("exact_match") or 0.0,
                    inv_weak.get("exact_match") or 0.0),
                "inversion_leakage_above_prior": max(
                    (inv.get("exact_match") or 0.0)
                    - floors["autoregressive"]["exact_match"],
                    (inv_weak.get("exact_match") or 0.0)
                    - floors["linear"]["exact_match"]),
                # post-relation-graph (deployed) surface
                "ctx_probe_linear_acc": probes_ctx["linear"]["accuracy"],
                "ctx_probe_mlp_acc": probes_ctx["mlp"]["accuracy"],
                "ctx_inversion_exact_match": inv_ctx.get("exact_match"),
                "ctx_inversion_cer": inv_ctx.get("cer"),
            })
            # Progress is not cosmetic here: the attack battery runs for tens of
            # minutes on CPU with no GPU activity to watch, so a silent process
            # is indistinguishable from a hung one.
            print("ROW " + json.dumps(per_seed[-1]), flush=True)
        rows.append({"mechanism": label, "per_seed": per_seed,
                     **_mean_std(per_seed)})

    return {"results": rows,
            "leace_fit_residual_covariance": fit_residual,
            "leace_val_residual_covariance": val_residual,
            "inversion_prior_floor": {
                d: {"exact_match": f["exact_match"], "cer": f["cer"]}
                for d, f in floors.items()},
            "leace_transfer_diagnostic": {
                "erased_val_features_linear_probe": leace_direct["linear"]["accuracy"],
                "erased_val_features_mlp_probe": leace_direct["mlp"]["accuracy"],
                "raw_val_features_linear_probe": raw_direct["linear"]["accuracy"],
                "raw_val_features_mlp_probe": raw_direct["mlp"]["accuracy"],
                "majority_baseline": raw_direct["majority_baseline"],
                "note": ("train and val use DISJOINT PII generator families, so "
                         "this measures whether the erasure transfers to unseen "
                         "PII — the deployment condition, and the reason a "
                         "guarantee proven on the fitting split is not enough."),
            },
            "n_pii_regions_attacked": int(pii_rows.sum()),
            "note": ("Report probe_mlp_acc alongside probe_linear_acc: LEACE "
                     "guards LINEAR readouts by construction, so a linear-only "
                     "drop is the guarantee restated, not evidence. hard_mask is "
                     "the floor any mechanism must beat to be worth its cost. "
                     "ctx_* are the same attacks AFTER the relation graph, i.e. "
                     "on the representation actually handed downstream; LEACE's "
                     "linear guarantee survives the linear projector but NOT the "
                     "graph's nonlinearity, so ctx_* is the deployed risk.")}


def _mean_std(per_seed):
    keys = [k for k in per_seed[0] if k != "seed"]
    out = {}
    for k in keys:
        vals = [r[k] for r in per_seed if r.get(k) is not None]
        if not vals:
            out[k] = {"mean": None, "std": None}
            continue
        m = sum(vals) / len(vals)
        sd = ((sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
              if len(vals) > 1 else None)
        out[k] = {"mean": m, "std": sd}
    return {"summary": out}


def _row(res):
    f = res["val_pii_f1"]
    return {"loss_end": res["loss_history"][-1],
            "val_pii_f1": f["f1"], "val_pii_precision": f["precision"],
            "val_pii_recall": f["recall"]}
