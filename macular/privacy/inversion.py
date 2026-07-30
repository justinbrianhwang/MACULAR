"""Embedding-inversion attack: recover the literal region text from its feature.

This replaces the cosine-similarity leakage metric. The reason is not only that
cosine is a weak attack — it is that cosine is CONTINUOUS, so it drifts with
training noise. Our measured failure: identical configs produced leakage numbers
that disagreed in sign, and a 3-seed sweep gave a std larger than the mean. An
exact-match rate cannot do that; it either recovers the string or it does not.

The threat model matches the deployment story: an attacker holds the pooled
region representations (the "safe" view that MACULAR would hand downstream) and
tries to read the patient's name out of them. It is the modality-adapted form of
the embedding-inversion attacks that are the accepted standard for representation
privacy (Vec2Text-class for text embeddings; CapRecover-class for vision
features).

Scope, stated plainly: this is a *fixed-length character decoder*, not a
Vec2Text-strength iterative inverter. It is a lower bound on what an attacker
gets. Reporting it as "leakage is at least X" is honest; reporting it as
"leakage is X" is not.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..evaluation.metrics import levenshtein

PAD = "\x00"


def _build_vocab(texts):
    chars = sorted({c for t in texts for c in t})
    itos = [PAD] + chars
    return {c: i for i, c in enumerate(itos)}, itos


def _encode(texts, stoi, max_len):
    out = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        for j, c in enumerate(t[:max_len]):
            out[i, j] = stoi.get(c, 0)
    return out


def _decode(idx_row, itos):
    return "".join(itos[i] for i in idx_row.tolist()).replace(PAD, "")


def inversion_attack(x: torch.Tensor, texts, seed: int = 0, epochs: int = 400,
                     lr: float = 1e-2, hidden: int = 512, max_len: int = 24,
                     train_frac: float = 0.5) -> dict:
    """Train an inverter on half the regions, report recovery on the other half.

    Returns exact-match rate and character error rate. Both are discrete, so they
    are stable across runs in a way the cosine metric was not.
    """
    texts = [t or "" for t in texts]
    n = len(texts)
    if n < 20:
        return {"n": n, "skipped": "need at least 20 regions"}

    stoi, itos = _build_vocab(texts)
    y = _encode(texts, stoi, max_len)
    x = x.detach().float()

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * train_frac)
    tr, te = perm[:cut], perm[cut:]

    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    xs = (x - mu) / sd

    torch.manual_seed(seed)
    v = len(itos)
    net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, max_len * v))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = net(xs[tr]).view(len(tr), max_len, v)
        loss = lossf(logits.reshape(-1, v), y[tr].reshape(-1))
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = net(xs[te]).view(len(te), max_len, v).argmax(-1)

    exact, cer_num, cer_den = 0, 0, 0
    for row, i in zip(pred, te.tolist()):
        gold = texts[i][:max_len]
        got = _decode(row, itos)
        if got == gold:
            exact += 1
        cer_num += levenshtein(got, gold)
        cer_den += max(1, len(gold))

    return {
        "n_eval": len(te),
        "exact_match": exact / max(1, len(te)),
        "cer": cer_num / max(1, cer_den),
        # 1.0 means the attacker reconstructed nothing; 0.0 means perfect theft.
        "protection": min(1.0, cer_num / max(1, cer_den)),
    }
