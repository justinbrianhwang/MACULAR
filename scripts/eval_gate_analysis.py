"""How many eval regions does the post-training gate need?

FINDINGS 4b.3: train loss cannot tell a diverged adapter from a good one, so
deployment needs an eval gate. This puts a number on the gate's size using the
per-region (pred, gold) pairs the runs now save, via bootstrap over gate sets
of size n.

Two verdicts a gate must get right:
  worse-than-baseline  (qwen25 seed 1: ja CER 0.226 vs baseline 0.118)
  worse-than-sibling   (r32 seed 1 vs seed 0: es CER 0.558 vs 0.165)

Usage: python scripts/eval_gate_analysis.py
"""

import json
import random

from macular.evaluation.metrics import cer as _cer


def _per_region(pairs):
    """(cer*weight, weight) per region, computed once so bootstrap is sums."""
    return [(_cer(p, g) * max(1, len(g)), max(1, len(g))) for _, p, g in pairs]


def detection_rate(pairs_bad, pairs_ref, n, boots=2000, rng=None):
    """P(gate ranks bad worse than ref) with n regions sampled per side.

    Paired sampling: the same region indices are used for both adapters, which
    is the deployable setup (one gate set, both models run on it).
    """
    rng = rng or random.Random(0)
    bad, ref = _per_region(pairs_bad), _per_region(pairs_ref)
    m = min(len(bad), len(ref))
    hits = 0
    for _ in range(boots):
        idx = rng.sample(range(m), n)
        cb = sum(bad[i][0] for i in idx) / sum(bad[i][1] for i in idx)
        cr = sum(ref[i][0] for i in idx) / sum(ref[i][1] for i in idx)
        hits += cb > cr
    return hits / boots


def _lang_pairs(pairs, lang):
    return [p for p in pairs if p[0] == lang]


def main():
    cases = []
    q = json.load(open("results/ocr_adapt_xfund_qwen25.json", encoding="utf-8"))
    cases.append(("qwen25 ja: diverged seed 1 vs baseline",
                  _lang_pairs(q["per_seed"][1]["eval_pairs"], "ja"),
                  _lang_pairs(q["eval_pairs_before"], "ja")))
    r = json.load(open("results/ocr_adapt_xfund_r32.json", encoding="utf-8"))
    cases.append(("r32 es: diverged seed 1 vs sibling seed 0",
                  _lang_pairs(r["per_seed"][1]["eval_pairs"], "es"),
                  _lang_pairs(r["per_seed"][0]["eval_pairs"], "es")))

    print(f"{'case':45s} " + " ".join(f"n={n:<4d}" for n in (10, 25, 50, 100, 200)))
    for name, bad, ref in cases:
        rates = [detection_rate(bad, ref, n) for n in (10, 25, 50, 100, 200)]
        print(f"{name:45s} " + " ".join(f"{r:.3f} " for r in rates))


if __name__ == "__main__":
    main()
