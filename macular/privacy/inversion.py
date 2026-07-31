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

Two decoders are provided, and both should be reported:

``linear``          predicts every character position independently from the
                    representation. Cheap, and a clear lower bound.
``autoregressive``  a GRU conditioned on the representation that decodes
                    characters left to right, so it can exploit the structure of
                    the string ("010-" prefixes, name morphology, date formats)
                    the way a real inverter does. Strictly stronger.

Neither decoder is uniformly stronger — the autoregressive one wins on real
features (exact match 0.141 -> 0.181) and loses on some small sets where it is
undertrained. A real attacker would simply use whichever works, so report both
and take the better one.

**The prior floor.** A structured-string decoder recovers some strings from the
text distribution alone, with no information from the representation at all: told
that inputs look like ``patient-###``, it will get a few exactly right by
guessing. Measured on pure noise features, the autoregressive decoder still
reached 0.017 exact match. So a raw recovery rate OVERSTATES leakage.
``prior_floor`` runs the identical attack with the representations shuffled
against the texts, and ``leakage_above_prior`` is the difference. Quote that.

Scope, stated plainly: even the autoregressive decoder is not a Vec2Text-strength
iterative inverter with an encoder in the loop. What we measure is a lower bound
on what an attacker gets. Reporting it as "leakage is at least X" is honest;
reporting it as "leakage is X" is not.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..evaluation.metrics import levenshtein

PAD = "\x00"
BOS = "\x02"


def _build_vocab(texts):
    chars = sorted({c for t in texts for c in t})
    itos = [PAD, BOS] + chars
    return {c: i for i, c in enumerate(itos)}, itos


def _encode(texts, stoi, max_len):
    out = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        for j, c in enumerate(t[:max_len]):
            out[i, j] = stoi.get(c, 0)
    return out


def _decode(idx_row, itos):
    return ("".join(itos[i] for i in idx_row.tolist())
            .replace(PAD, "").replace(BOS, ""))


class _ARDecoder(nn.Module):
    """GRU conditioned on the representation, decoding characters left to right.

    Stronger than independent per-position prediction because PII strings are
    highly structured: once the decoder has emitted "010-" the rest of a phone
    number is far more predictable, and the same holds for date formats and name
    morphology. An attacker would exploit that, so the evaluation must too.
    """

    def __init__(self, d_in, vocab, hidden=512, emb=64):
        super().__init__()
        self.init = nn.Linear(d_in, hidden)
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb + d_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, z, inputs):
        # z conditions BOTH the initial state and every step, so the signal
        # cannot fade out along the sequence.
        h0 = torch.tanh(self.init(z)).unsqueeze(0)
        e = self.emb(inputs)
        zz = z.unsqueeze(1).expand(-1, e.shape[1], -1)
        h, _ = self.rnn(torch.cat([e, zz], dim=-1), h0)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, z, max_len, bos_idx):
        cur = torch.full((z.shape[0], 1), bos_idx, dtype=torch.long)
        h = torch.tanh(self.init(z)).unsqueeze(0)
        outs = []
        for _ in range(max_len):
            e = self.emb(cur[:, -1:])
            step, h = self.rnn(torch.cat([e, z.unsqueeze(1)], dim=-1), h)
            nxt = self.out(step[:, -1]).argmax(-1, keepdim=True)
            outs.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
        return torch.cat(outs, dim=1)


def inversion_attack(x: torch.Tensor, texts, seed: int = 0, epochs: int = 400,
                     lr: float = 1e-2, hidden: int = 512, max_len: int = 24,
                     train_frac: float = 0.5,
                     decoder: str = "autoregressive") -> dict:
    """Train an inverter on half the regions, report recovery on the other half.

    Returns exact-match rate and character error rate. Both are discrete, so they
    are stable across runs in a way the cosine metric was not.

    ``decoder``: "autoregressive" (default, stronger) or "linear" (the weaker
    per-position baseline). Report both when making a privacy claim — the gap
    between them tells a reviewer how much attack strength matters here.
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
    if decoder == "autoregressive":
        net = _ARDecoder(x.shape[1], v, hidden=hidden)
        bos = torch.full((len(tr), 1), stoi.get(BOS, 1), dtype=torch.long)
        inp = torch.cat([bos, y[tr][:, :-1]], dim=1)     # teacher forcing
        opt = torch.optim.Adam(net.parameters(), lr=lr * 0.1)
        lossf = nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad()
            logits = net(xs[tr], inp)
            loss = lossf(logits.reshape(-1, v), y[tr].reshape(-1))
            loss.backward()
            opt.step()
        pred = net.greedy(xs[te], max_len, stoi.get(BOS, 1))
    else:
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
        "decoder": decoder,
        "exact_match": exact / max(1, len(te)),
        "cer": cer_num / max(1, cer_den),
        # 1.0 means the attacker reconstructed nothing; 0.0 means perfect theft.
        "protection": min(1.0, cer_num / max(1, cer_den)),
    }


def prior_floor(x: torch.Tensor, texts, seed: int = 0, **kw) -> dict:
    """The same attack with the representation shuffled against the texts.

    Whatever it recovers comes from the text distribution alone. Subtract it
    before calling anything leakage: on structured strings a decoder guesses a
    few exactly right with no information at all.
    """
    g = torch.Generator().manual_seed(seed + 9973)
    perm = torch.randperm(len(texts), generator=g)
    out = inversion_attack(x[perm], texts, seed=seed, **kw)
    out["shuffled_control"] = True
    return out


def attack_with_floor(x: torch.Tensor, texts, seed: int = 0, **kw) -> dict:
    """Run the attack and its prior floor, and report the difference."""
    real = inversion_attack(x, texts, seed=seed, **kw)
    floor = prior_floor(x, texts, seed=seed, **kw)
    return {
        **real,
        "prior_floor_exact_match": floor["exact_match"],
        "prior_floor_cer": floor["cer"],
        # THE number: recovery attributable to the representation.
        "leakage_above_prior_exact_match": real["exact_match"] - floor["exact_match"],
        "leakage_above_prior_cer": floor["cer"] - real["cer"],
    }
