"""Does fitting concept erasure across SEVERAL PII value families restore transfer?

Established so far:
  - LEACE works exactly as advertised in-distribution: fitted on validation and
    probed on validation, the linear probe sits at 0.847 against a majority
    baseline of 0.847.
  - Fitted on train (family A) and probed on validation (family B), it leaves the
    linear probe at 0.932 and the residual cross-covariance at 0.771. The
    guarantee does not transfer to unseen PII values.
  - Cheap remedies fail: a coarser binary concept (0.956), heavier shrinkage
    (0.928), and rank truncation (0.952-0.956) are all no better, several worse.

The remaining hypothesis is that one value family pins down a subspace that is
specific to it. This script fits the eraser on the union of several TRAIN-SIDE
families (A, D, E — all disjoint from the val family B and the test family C, so
nothing leaks) and measures transfer to B.

Outcome either way is informative:
  - if transfer improves with family count, family-diverse fitting is the fix and
    the number of families needed is a reportable quantity;
  - if it does not, the failure is intrinsic to erasing a lexical attribute from
    visual features, and the negative result covers closed-form erasure properly.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macular.models import VLMBackbone, VLMBackboneConfig             # noqa: E402
from macular.models.features import documents_to_vlm_batch            # noqa: E402
from macular.privacy import fit_leace, probe_leakage                  # noqa: E402
from macular.schema import read_jsonl                                 # noqa: E402

BASE = "data/meddoc_cf_engines"
EXTRA = {"D": "data/meddoc_cf_famD", "E": "data/meddoc_cf_famE"}
CACHE = "results/_features_multifamily.pt"
MAX_DOCS, MAX_REGIONS = 120, 48


def encode_all():
    if os.path.exists(CACHE):
        print(f"using cached features: {CACHE}")
        return torch.load(CACHE)

    bk = VLMBackbone(VLMBackboneConfig(family="paddleocr_vl", device="cuda",
                                       dtype="bfloat16"))
    bk.load()
    cache, stats = {}, {}
    out = {}
    # Normalization statistics are fitted on family A and reused everywhere, so
    # every family lands in the same feature space; refitting per family would
    # make the comparison meaningless.
    tr = documents_to_vlm_batch(read_jsonl(f"{BASE}/train.jsonl")[:MAX_DOCS], bk,
                                BASE, MAX_REGIONS, cache, stats=stats)
    out["A"] = (tr[0][tr[4]], tr[2][tr[4]])
    va = documents_to_vlm_batch(read_jsonl(f"{BASE}/val.jsonl")[:MAX_DOCS], bk,
                                BASE, MAX_REGIONS, cache, stats=stats)
    out["B_val"] = (va[0][va[4]], va[2][va[4]])
    for fam, path in EXTRA.items():
        b = documents_to_vlm_batch(read_jsonl(f"{path}/train.jsonl")[:MAX_DOCS],
                                   bk, path, MAX_REGIONS, cache, stats=stats)
        out[fam] = (b[0][b[4]], b[2][b[4]])
    os.makedirs("results", exist_ok=True)
    torch.save(out, CACHE)
    return out


def main():
    d = encode_all()
    xva, zva = d["B_val"]
    ref = probe_leakage(xva, zva, seed=0)
    maj = ref["majority_baseline"]
    print(f"raw val                 lin {ref['linear']['accuracy']:.3f}  "
          f"mlp {ref['mlp']['accuracy']:.3f}   (majority {maj:.3f})")

    combos = [("A",), ("A", "D"), ("A", "D", "E")]
    rows = {"majority_baseline": maj,
            "raw_val_linear": ref["linear"]["accuracy"],
            "raw_val_mlp": ref["mlp"]["accuracy"],
            "fits": {}}
    for combo in combos:
        x = torch.cat([d[f][0] for f in combo])
        z = torch.cat([d[f][1] for f in combo])
        er = fit_leace(x, z)
        p = probe_leakage(er(xva), zva, seed=0)
        name = "+".join(combo)
        rows["fits"][name] = {
            "n_fit_regions": int(x.shape[0]),
            "linear": p["linear"]["accuracy"],
            "mlp": p["mlp"]["accuracy"],
            "linear_above_majority": p["linear"]["accuracy_above_majority"],
            "val_residual_covariance": er.residual_covariance(xva, zva),
            "fit_residual_covariance": er.residual_covariance(x, z),
        }
        r = rows["fits"][name]
        print(f"fit on {name:<8} (n={r['n_fit_regions']:>5})  "
              f"lin {r['linear']:.3f}  mlp {r['mlp']:.3f}  "
              f"resid_val {r['val_residual_covariance']:.3e}")

    # In-distribution control, the floor this could reach.
    within = fit_leace(xva, zva)
    p = probe_leakage(within(xva), zva, seed=0)
    rows["within_family_control"] = {
        "linear": p["linear"]["accuracy"], "mlp": p["mlp"]["accuracy"]}
    print(f"control fit on val      lin {p['linear']['accuracy']:.3f}  "
          f"mlp {p['mlp']['accuracy']:.3f}")

    with open("results/leace_multifamily.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
