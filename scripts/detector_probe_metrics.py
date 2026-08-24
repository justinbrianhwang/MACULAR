"""Reviewer round-2 recommendations 6-7 (CPU, PaddleOCR-VL cached features):

  6. the PII detector's operating point: is-PII precision/recall/F1 at the
     deployed threshold 0.5, on the held-out value family (val = family B) --
     the AP in Table 6 is threshold-free and hides the recall the mask
     actually runs at;
  7. balanced accuracy and per-class recall for the MLP probe, since raw
     accuracy against a 0.847 majority baseline compresses the picture.

Same protocol as erasure_comparison (80 epochs, lr 3e-3, 120 docs, 3 seeds).

Usage: python scripts/detector_probe_metrics.py
    -> results/detector_probe_metrics.json
"""
import json
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macular.models.train import ERASURE_MECHANISMS, build_recon_targets, fit_with_vlm  # noqa: E402
from macular.privacy import fit_leace  # noqa: E402
from macular.schema import read_jsonl  # noqa: E402
from scripts.strong_attack import labels_boxes, DATA, CACHE, MAX_DOCS, MAX_REGIONS, SEEDS  # noqa: E402


def probe_with_preds(x, y, seed, hidden=256, epochs=300, lr=1e-2):
    """mlp_probe from macular.privacy.probes, but returning predictions."""
    x = x.detach().float()
    y = y.detach().long()
    n_classes = int(y.max().item()) + 1
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    cut = int(x.shape[0] * 0.5)
    tr, te = perm[:cut], perm[cut:]
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, n_classes))
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    xs = (x - mu) / sd
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(net(xs[tr]), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        pred = net(xs[te]).argmax(-1)
    gold = y[te]
    recalls = {}
    for c in range(n_classes):
        n_c = int((gold == c).sum())
        if n_c:
            recalls[c] = float(((pred == c) & (gold == c)).sum()) / n_c
    return {"accuracy": float((pred == gold).float().mean()),
            "balanced_accuracy": sum(recalls.values()) / len(recalls),
            "per_class_recall": recalls,
            "majority_baseline": float(torch.bincount(gold).max()) / len(gold)}


def detector_operating_point(pii_logits, gold_pii):
    """is-PII precision/recall/F1 at the deployed mask threshold m>0.5,
    i.e. P(NON_PII) < 0.5."""
    p = torch.softmax(pii_logits, dim=-1)
    flag = (1.0 - p[..., 0]) > 0.5
    is_pii = gold_pii > 0
    tp = int((flag & is_pii).sum())
    fp = int((flag & ~is_pii).sum())
    fn = int((~flag & is_pii).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision_at_0.5": prec, "recall_at_0.5": rec,
            "f1_at_0.5": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "n_pii": tp + fn}


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 4))
    c = torch.load(CACHE)
    train_docs = read_jsonl(f"{DATA}/train.jsonl")[:MAX_DOCS]
    val_docs = read_jsonl(f"{DATA}/val.jsonl")[:MAX_DOCS]
    tb, tp, tc, tm = labels_boxes(train_docs)
    vb, vp, vc, vm = labels_boxes(val_docs)
    train_batch = (c["train_feats"], tb, tp, tc, tm)
    val_batch = (c["val_feats"], vb, vp, vc, vm)
    recon = (build_recon_targets(train_docs, MAX_REGIONS, MAX_DOCS),
             build_recon_targets(val_docs, MAX_REGIONS, MAX_DOCS))
    eraser = fit_leace(c["train_feats"][tm], tp[tm])
    val_pii = vp[vm]

    out = {"backbone": "paddleocr_vl", "note": "held-out value family B; "
           "3 seeds; detector = the model's own PII head at mask threshold 0.5",
           "mechanisms": []}
    for label, mode, needs_eraser in ERASURE_MECHANISMS:
        per_seed = []
        for s in SEEDS:
            m, res = fit_with_vlm(None, None, None, DATA, epochs=80, lr=3e-3,
                                  seed=s, model_cfg={"redaction": mode},
                                  batches=(train_batch, val_batch), recon=recon,
                                  eraser=eraser if needs_eraser else None)
            with torch.no_grad():
                o = m(val_batch[0], val_batch[1], key_padding_mask=~vm)
            row = {"seed": s,
                   "detector": detector_operating_point(o["pii_logits"][vm], val_pii),
                   "mlp_ctx": probe_with_preds(o["z_ctx_safe"][vm], val_pii, s),
                   "mlp_out": probe_with_preds(o["z_safe"][vm], val_pii, s)}
            per_seed.append(row)
            print(label, s, "det P/R@0.5",
                  round(row["detector"]["precision_at_0.5"], 3),
                  round(row["detector"]["recall_at_0.5"], 3),
                  "ctx balacc", round(row["mlp_ctx"]["balanced_accuracy"], 3),
                  "acc", round(row["mlp_ctx"]["accuracy"], 3), flush=True)
        out["mechanisms"].append({"mechanism": label, "per_seed": per_seed})
        json.dump(out, open("results/detector_probe_metrics.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
