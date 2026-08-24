"""Stronger attackers + probe selectivity on the redaction mechanisms (CPU).

Reviewer conditions 1 and C2: the paper's attackers were a linear probe, a
2-layer MLP and a teacher-forced GRU inverter. This re-trains every mechanism
from the cached PaddleOCR-VL region features (identical protocol to
erasure_comparison: 80 epochs, lr 3e-3, 120 docs, 3 seeds) and attacks
z_safe / z_ctx with:

  * inverter-XL : GRU with 2x hidden (1024), 3x epochs (1200), beam search k=5
                  instead of greedy — the strongest decoder we can train
                  without re-embedding hypotheses (Vec2Text needs the
                  embedder in the loop; the mechanism output has no embedder).
  * probe selectivity (Hewitt & Liang 2019): the same linear/MLP probes on a
    control task whose label is a random function of the region's text,
    reported as (task acc - control acc). Low selectivity = the probe is
    fitting the probe, not reading the representation.

Usage: python scripts/strong_attack.py  -> results/strong_attack_paddleocr_vl.json
"""
import json
import os
import random
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macular.models.train import (ERASURE_MECHANISMS, _val_region_texts,  # noqa: E402
                                  build_recon_targets, fit_with_vlm)
from macular.models.features import PII_TO_IDX, CLIN_TO_IDX, NON_PII  # noqa: E402
from macular.privacy import fit_leace, probe_leakage  # noqa: E402
from macular.privacy.inversion import (_ARDecoder, _build_vocab, _decode,  # noqa: E402
                                       _encode, BOS, levenshtein)
from macular.schema import read_jsonl  # noqa: E402

DATA = "data/meddoc_cf_engines"
CACHE = "results/_features_paddleocr_vl.pt"
MAX_DOCS, MAX_REGIONS, SEEDS = 120, 48, (0, 1, 2)


def labels_boxes(docs):
    B = len(docs)
    boxes = torch.zeros(B, MAX_REGIONS, 4)
    pii = torch.zeros(B, MAX_REGIONS, dtype=torch.long)
    clin = torch.zeros(B, MAX_REGIONS, dtype=torch.long)
    mask = torch.zeros(B, MAX_REGIONS, dtype=torch.bool)
    for b, doc in enumerate(docs):
        cands = doc.candidates[:MAX_REGIONS]
        for i, c in enumerate(cands):
            boxes[b, i] = torch.tensor(c.bbox.as_list())
            pii[b, i] = PII_TO_IDX.get(c.pii_type or NON_PII, 0)
            clin[b, i] = CLIN_TO_IDX.get(c.clinical_type or "NONE", 0)
            mask[b, i] = True
    return boxes, pii, clin, mask


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def greedy_dev(net, z, max_len, bos_idx):
    """net.greedy with the start token on the same device as z."""
    cur = torch.full((z.shape[0], 1), bos_idx, dtype=torch.long, device=z.device)
    h = torch.tanh(net.init(z)).unsqueeze(0)
    outs = []
    for _ in range(max_len):
        e = net.emb(cur[:, -1:])
        step, h = net.rnn(torch.cat([e, z.unsqueeze(1)], dim=-1), h)
        nxt = net.out(step[:, -1]).argmax(-1, keepdim=True)
        outs.append(nxt)
        cur = torch.cat([cur, nxt], dim=1)
    return torch.cat(outs, dim=1)


def beam_decode(net, z, max_len, bos, k=5):
    """Beam search over the GRU decoder, batch of 1 per region."""
    outs = []
    for i in range(z.shape[0]):
        zi = z[i:i + 1]
        h0 = torch.tanh(net.init(zi)).unsqueeze(0)
        beams = [(0.0, [bos], h0)]
        for _ in range(max_len):
            cand = []
            for score, seq, h in beams:
                e = net.emb(torch.tensor([[seq[-1]]], device=z.device))
                step, h2 = net.rnn(torch.cat([e, zi.unsqueeze(1)], dim=-1), h)
                logp = torch.log_softmax(net.out(step[:, -1]), dim=-1)[0]
                top = torch.topk(logp, k)
                for lp, idx in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((score + lp, seq + [idx], h2))
            beams = sorted(cand, key=lambda t: -t[0])[:k]
        outs.append(torch.tensor(beams[0][1][1:]))
    return torch.stack(outs)


def invert_xl(x, texts, seed, epochs=1200, hidden=1024, max_len=24, k=5):
    texts = [t or "" for t in texts]
    n = len(texts)
    stoi, itos = _build_vocab(texts)
    y = _encode(texts, stoi, max_len)
    x = x.detach().float()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * 0.5)
    tr, te = perm[:cut], perm[cut:]
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    xs = ((x - mu) / sd).to(DEV)
    y = y.to(DEV)
    torch.manual_seed(seed)
    v = len(itos)
    net = _ARDecoder(x.shape[1], v, hidden=hidden).to(DEV)
    bos = stoi.get(BOS, 1)
    inp = torch.cat([torch.full((len(tr), 1), bos, dtype=torch.long, device=DEV), y[tr][:, :-1]], dim=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(xs[tr], inp).reshape(-1, v), y[tr].reshape(-1))
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred_g = greedy_dev(net, xs[te], max_len, bos).cpu()
        pred_b = beam_decode(net, xs[te], max_len, bos, k=k)
    res = {}
    for name, pred in (("greedy", pred_g), ("beam5", pred_b)):
        exact, num, den = 0, 0, 0
        for row, i in zip(pred, te.tolist()):
            gold = texts[i][:max_len]
            got = _decode(row, itos)
            exact += got == gold
            num += levenshtein(got, gold)
            den += max(1, len(gold))
        res[name] = {"exact_match": exact / len(te), "cer": num / den}
    return res


def selectivity(z, y, texts, seed):
    """Control task: label = hash of the region text -> random class id with
    the same number of classes; same probes, same split."""
    rng = random.Random(seed)
    uniq = sorted(set(texts))
    n_cls = int(y.max().item()) + 1
    ctrl_map = {t: rng.randrange(n_cls) for t in uniq}
    y_ctrl = torch.tensor([ctrl_map[t] for t in texts])
    task = probe_leakage(z, y, seed=seed)
    ctrl = probe_leakage(z, y_ctrl, seed=seed)
    return {
        "linear_task": task["linear"]["accuracy"], "linear_control": ctrl["linear"]["accuracy"],
        "linear_selectivity": task["linear"]["accuracy"] - ctrl["linear"]["accuracy"],
        "mlp_task": task["mlp"]["accuracy"], "mlp_control": ctrl["mlp"]["accuracy"],
        "mlp_selectivity": task["mlp"]["accuracy"] - ctrl["mlp"]["accuracy"],
        "majority_baseline": task["majority_baseline"],
    }


def main():
    torch.set_num_threads(max(1, os.cpu_count() - 4))
    c = torch.load(CACHE)
    train_docs = read_jsonl(f"{DATA}/train.jsonl")[:MAX_DOCS]
    val_docs = read_jsonl(f"{DATA}/val.jsonl")[:MAX_DOCS]
    tb, tp, tc, tm = labels_boxes(train_docs)
    vb, vp, vc, vm = labels_boxes(val_docs)
    assert torch.equal(tp, c["train_pii"]) and torch.equal(vm, c["val_mask"]), "cache misaligned"
    train_batch = (c["train_feats"], tb, tp, tc, tm)
    val_batch = (c["val_feats"], vb, vp, vc, vm)
    recon = (build_recon_targets(train_docs, MAX_REGIONS, MAX_DOCS),
             build_recon_targets(val_docs, MAX_REGIONS, MAX_DOCS))
    eraser = fit_leace(c["train_feats"][tm], tp[tm])
    texts = _val_region_texts(val_docs, MAX_DOCS, MAX_REGIONS)
    val_texts = [t for t, keep in zip(texts, vm.reshape(-1).tolist()) if keep]
    val_pii = vp[vm]
    pii_rows = val_pii > 0
    pii_texts = [t for t, k in zip(val_texts, pii_rows.tolist()) if k]

    out = {"backbone": "paddleocr_vl", "protocol": "80 ep, lr 3e-3, 120 docs, seeds 0-2; "
           "inverter-XL: GRU h=1024, 1200 ep, beam k=5", "mechanisms": []}
    for label, mode, needs_eraser in ERASURE_MECHANISMS:
        per_seed = []
        for s in SEEDS:
            m, res = fit_with_vlm(None, None, None, DATA, epochs=80, lr=3e-3, seed=s,
                                  model_cfg={"redaction": mode},
                                  batches=(train_batch, val_batch), recon=recon,
                                  eraser=eraser if needs_eraser else None)
            with torch.no_grad():
                o = m(val_batch[0], val_batch[1], key_padding_mask=~vm)
                z, z_ctx = o["z_safe"][vm], o["z_ctx_safe"][vm]
            row = {"seed": s,
                   "clinical_macro_f1": res["val_clinical"]["macro_f1"],
                   "pii_average_precision": res["val_pii_f1"]["average_precision"],
                   "inv_xl": invert_xl(z[pii_rows], pii_texts, s),
                   "inv_xl_ctx": invert_xl(z_ctx[pii_rows], pii_texts, s),
                   "selectivity": selectivity(z, val_pii, val_texts, s),
                   "selectivity_ctx": selectivity(z_ctx, val_pii, val_texts, s)}
            per_seed.append(row)
            print(label, s, {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in row.items() if k in ("clinical_macro_f1",)},
                  "invXL beam EM", round(row["inv_xl"]["beam5"]["exact_match"], 3),
                  "ctx", round(row["inv_xl_ctx"]["beam5"]["exact_match"], 3),
                  "sel mlp", round(row["selectivity"]["mlp_selectivity"], 3), flush=True)
        out["mechanisms"].append({"mechanism": label, "per_seed": per_seed})
        json.dump(out, open("results/strong_attack_paddleocr_vl.json", "w"), indent=1)
    # prior floor for the XL inverter (features shuffled against texts)
    g = torch.Generator().manual_seed(0)
    xz = c["val_feats"][vm][pii_rows]
    out["prior_floor_inv_xl"] = invert_xl(xz[torch.randperm(xz.shape[0], generator=g)], pii_texts, 0)
    json.dump(out, open("results/strong_attack_paddleocr_vl.json", "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
