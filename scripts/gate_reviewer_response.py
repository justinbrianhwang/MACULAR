"""Gate sizing, reviewer-response edition.

Three things the retrospective analysis in eval_gate_analysis.py did not do:

  1. document-level bootstrap (regions in a page share layout/degradation, so
     region-i.i.d. resampling overstates the effective n);
  2. the missing scenario — a run that BEATS the baseline but LOSES to a
     healthy sibling by a small margin (reference-based subtle detection);
  3. a fixed decision rule applied to every adapter trained AFTER the gate
     section was written (prospective application, no re-tuning).

Doc membership is reconstructed by replaying the exact eval-item construction
(halve -> interleave -> docs[:50] -> regions[:24]) and checking the gold
sequence against the saved eval pairs.

Usage: python scripts/gate_reviewer_response.py
"""
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macular.evaluation.metrics import cer as _cer
from macular.models.ocr_adapt import (_regions, halve_by_language,
                                      interleave_by_language)
from macular.schema import read_jsonl

R = "results/ocr_adapt_{}.json"


def load(name):
    return json.load(open(R.format(name), encoding="utf-8"))


def doc_index(data_dir="data/xfund", max_docs=50, max_regions=24):
    """doc id per eval item, in eval order."""
    docs = read_jsonl(os.path.join(data_dir, "test.jsonl"))
    evald = interleave_by_language(halve_by_language(docs)[1])
    ids, golds = [], []
    for d in evald[:max_docs]:
        for _crop, gold, lang in _regions([d], data_dir, 1, max_regions):
            ids.append(d.doc_id)
            golds.append((lang, gold))
    return ids, golds


def per_region(pairs):
    return [(_cer(p, g) * max(1, len(g)), max(1, len(g))) for _, p, g in pairs]


def detect(bad, ref, n, docs=None, boots=2000, seed=0):
    """P(gate says bad is worse than ref). docs: doc id per index -> sample
    whole documents (with replacement) until >= n regions."""
    rng = random.Random(seed)
    b, r = per_region(bad), per_region(ref)
    m = len(b)
    by_doc = {}
    if docs is not None:
        for i, d in enumerate(docs):
            by_doc.setdefault(d, []).append(i)
        keys = list(by_doc)
    hits = 0
    for _ in range(boots):
        if docs is None:
            idx = rng.sample(range(m), n)
        else:
            idx = []
            while len(idx) < n:
                idx.extend(by_doc[rng.choice(keys)])
        cb = sum(b[i][0] for i in idx) / sum(b[i][1] for i in idx)
        cr = sum(r[i][0] for i in idx) / sum(r[i][1] for i in idx)
        hits += cb > cr
    return hits / boots


def gate_cer_dist(pairs, n, boots=2000, seed=0):
    rng = random.Random(seed)
    pr = per_region(pairs)
    out = []
    for _ in range(boots):
        idx = rng.sample(range(len(pr)), n)
        out.append(sum(pr[i][0] for i in idx) / sum(pr[i][1] for i in idx))
    return out


def lang(pairs, l):
    return [p for p in pairs if p[0] == l]


def main():
    ids, golds = doc_index()
    NS = (10, 25, 50, 100, 200)
    # --- sanity: saved pairs line up with the replayed item order ------------
    q = load("xfund_qwen25")
    assert [(p[0], p[2]) for p in q["eval_pairs_before"]] == golds, "order mismatch"
    lang_ids = {l: [d for d, (ll, _) in zip(ids, golds) if ll == l] for l in ("ja", "es")}

    r32 = load("xfund_r32")
    rs = load("xfund_rslora_r8_more")     # seeds 3,4,5,6
    s2r = load("syn2real_matched")
    det = load("xfund_det")               # seeds 0,0,3,4

    def full(pairs):
        pr = per_region(pairs)
        return sum(a for a, _ in pr) / sum(w for _, w in pr)

    cases = [
        ("qwen25 ja: seed1 vs baseline (subtle, vs baseline)", "ja",
         lang(q["per_seed"][1]["eval_pairs"], "ja"), lang(q["eval_pairs_before"], "ja")),
        ("r32 es: seed1 vs seed0 (gross, vs sibling)", "es",
         lang(r32["per_seed"][1]["eval_pairs"], "es"), lang(r32["per_seed"][0]["eval_pairs"], "es")),
        ("qwen25 ja: seed1 vs sibling seed0 (subtle, vs sibling)", "ja",
         lang(q["per_seed"][1]["eval_pairs"], "ja"), lang(q["per_seed"][0]["eval_pairs"], "ja")),
        ("rsLoRA r8 ja: seed5 vs seed6 (subtle, vs sibling)", "ja",
         lang(rs["per_seed"][2]["eval_pairs"], "ja"), lang(rs["per_seed"][3]["eval_pairs"], "ja")),
        ("rsLoRA r8 ja: seed3 vs seed6 (marginal, vs sibling)", "ja",
         lang(rs["per_seed"][0]["eval_pairs"], "ja"), lang(rs["per_seed"][3]["eval_pairs"], "ja")),
        ("det ja: seed3 bad basin vs seed0 (vs sibling)", "ja",
         lang(det["per_seed"][2]["eval_pairs"], "ja"), lang(det["per_seed"][0]["eval_pairs"], "ja")),
        ("syn2real es: seed0 vs seed1 (language-selective, vs sibling)", "es",
         lang(s2r["per_seed"][0]["eval_pairs"], "es"), lang(s2r["per_seed"][1]["eval_pairs"], "es")),
    ]
    print("\n== detection rate: region-level / document-level bootstrap ==")
    print(f"{'case':62s} dCER   " + "  ".join(f"n={n:<11d}" for n in NS))
    table = []
    for name, l, bad, ref in cases:
        d = full(bad) - full(ref)
        row = []
        for n in NS:
            rr = detect(bad, ref, n)
            dd = detect(bad, ref, n, docs=lang_ids[l])
            row.append((rr, dd))
        table.append({"case": name, "delta_cer": d, "rates": dict(zip(NS, row))})
        print(f"{name:62s} {d:5.3f}  " + "  ".join(f"{a:.3f}/{b:.3f}" for a, b in row))

    # --- prospective application of a fixed rule -----------------------------
    # Rule fixed in advance: per-language 100-region gate; reject if gate CER
    # exceeds a ceiling of 1.5x the per-language median of the 7-run ja+es
    # default pool (medians ja 0.114 / es 0.299 -> ceilings 0.171 / 0.449).
    ceil = {"ja": 1.5 * 0.114, "es": 1.5 * 0.299}
    later = [("rsLoRA r8 more", rs), ("PiSSA lr3e-5", load("xfund_pissa_r8_lr3e5")),
             ("rsLoRA r16 lr3e-5", load("xfund_rslora_r16_lr3e5")),
             ("syn->real matched", s2r), ("deterministic", det)]
    print(f"\n== prospective rule: 100-region per-language gate, ceiling ja {ceil['ja']:.3f} / es {ceil['es']:.3f} ==")
    print(f"{'run':28s} lang  fullCER  P(gate rejects)  verdict")
    pros = []
    for name, res in later:
        for s in res["per_seed"]:
            for l in ("ja", "es"):
                pairs = lang(s["eval_pairs"], l)
                f = full(pairs)
                dist = gate_cer_dist(pairs, 100)
                p = sum(x > ceil[l] for x in dist) / len(dist)
                truth = f > ceil[l]
                pros.append({"run": f"{name} seed{s['seed']}", "lang": l, "full_cer": f,
                             "p_reject": p, "truth_over_ceiling": truth})
                flag = "REJECT" if p > 0.5 else "accept"
                mark = "" if (p > 0.5) == truth else "  <-- gate/truth disagree"
                print(f"{name+' s'+str(s['seed']):28s} {l:4s}  {f:.3f}    {p:.3f}            {flag}{mark}")
    json.dump({"detection": table, "prospective": pros, "ceiling": ceil},
              open("results/gate_reviewer_response.json", "w"), indent=1)


if __name__ == "__main__":
    main()
