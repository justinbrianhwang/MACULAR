"""Post-hoc probes: the accepted evidence class for representation leakage.

The methodology that killed adversarial attribute removal (Elazar & Goldberg,
EMNLP 2018) is simple: FREEZE the representation, then train a FRESH classifier
on it. If the attribute comes back, it was hidden, not removed. That is exactly
what we run here.

Two probes, always reported together:

* ``linear_probe``    — the readout LEACE provably defeats. Alone it is a
                        rubber stamp for any linear erasure method.
* ``mlp_probe``       — a small nonlinear probe. Published stress tests recover
                        much of an "erased" concept this way, so a privacy claim
                        without it is not credible.

Both report ACCURACY-class metrics against the majority-class baseline, not a
similarity score. That is deliberate: our earlier cosine-similarity leakage
metric flipped sign between identical runs because a continuous similarity
drifts with training noise. Discrete metrics do not.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _split(n: int, seed: int, frac: float = 0.5):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * frac)
    return perm[:cut], perm[cut:]


def _metrics(pred: torch.Tensor, gold: torch.Tensor, n_classes: int) -> dict:
    acc = float((pred == gold).float().mean())
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        if tp + fn == 0:               # class absent from the eval half
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    # The floor a probe must beat to have learned anything at all.
    counts = torch.bincount(gold, minlength=n_classes)
    majority = float(counts.max()) / max(1, int(counts.sum()))
    return {"accuracy": acc, "macro_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
            "majority_baseline": majority,
            "accuracy_above_majority": acc - majority}


def _train_probe(net, xtr, ytr, xte, yte, n_classes, epochs, lr):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(xte).argmax(-1)
    return _metrics(pred, yte, n_classes)


def linear_probe(x, y, seed=0, epochs=300, lr=1e-2):
    """Logistic regression on frozen features. LEACE provably defeats this."""
    return _run(x, y, seed, epochs, lr, hidden=None)


def mlp_probe(x, y, seed=0, epochs=300, lr=1e-2, hidden=256):
    """Small nonlinear probe. LEACE does NOT guard against this by construction."""
    return _run(x, y, seed, epochs, lr, hidden=hidden)


def _run(x, y, seed, epochs, lr, hidden):
    x = x.detach().float()
    y = y.detach().long()
    n_classes = int(y.max().item()) + 1
    tr, te = _split(x.shape[0], seed)
    torch.manual_seed(seed)
    d = x.shape[1]
    net = (nn.Linear(d, n_classes) if hidden is None else
           nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_classes)))
    # Standardize with TRAIN statistics only.
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    xs = (x - mu) / sd
    return _train_probe(net, xs[tr], y[tr], xs[te], y[te], n_classes, epochs, lr)


def probe_leakage(x: torch.Tensor, y: torch.Tensor, seed: int = 0) -> dict:
    """Run both probes and return them together.

    Reporting rule: a defense is only credible if BOTH probes drop toward the
    majority baseline. A linear-only drop means the concept moved into nonlinear
    structure, which is what "hiding, not removing" looks like.
    """
    lin = linear_probe(x, y, seed=seed)
    mlp = mlp_probe(x, y, seed=seed)
    return {
        "linear": lin,
        "mlp": mlp,
        "majority_baseline": lin["majority_baseline"],
        # Headline: how much a NONLINEAR attacker still gets above chance.
        "nonlinear_leakage": mlp["accuracy_above_majority"],
    }
