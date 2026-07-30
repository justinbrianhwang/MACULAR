"""Control experiment: is LEACE's failure a transfer failure or an implementation bug?

The main run found the guarantee holds perfectly where it is fit (residual
cross-covariance 1.5e-07) and not at all on validation (0.771). Two readings:

  (a) TRANSFER FAILURE. Our splits use DISJOINT PII generator families by design
      (train = family A, val = family B), so the concept subspace carrying
      family A's PII is not the one carrying family B's. This is the deployment
      condition — new patients, unseen name distributions.
  (b) A BUG in our LEACE implementation, which would make the whole comparison
      meaningless.

This script separates them by fitting the eraser ON VALIDATION and probing
validation — the in-distribution upper bound. If that collapses to the majority
baseline, the implementation is correct and (a) is the answer.

Encoded features are cached to disk so this can be re-run without paying for the
vision forward pass again.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macular.models import VLMBackbone, VLMBackboneConfig          # noqa: E402
from macular.models.features import documents_to_vlm_batch          # noqa: E402
from macular.privacy import fit_leace, probe_leakage                # noqa: E402
from macular.schema import read_jsonl                               # noqa: E402

DATA = "data/meddoc_cf_engines"
CACHE = "results/_features_paddleocr_vl.pt"
MAX_DOCS, MAX_REGIONS = 120, 48


def load_features():
    if os.path.exists(CACHE):
        print(f"using cached features: {CACHE}")
        return torch.load(CACHE)
    bk = VLMBackbone(VLMBackboneConfig(family="paddleocr_vl", device="cuda",
                                       dtype="bfloat16"))
    bk.load()
    cache, stats = {}, {}
    tr = documents_to_vlm_batch(read_jsonl(f"{DATA}/train.jsonl")[:MAX_DOCS], bk,
                                DATA, MAX_REGIONS, cache, stats=stats)
    va = documents_to_vlm_batch(read_jsonl(f"{DATA}/val.jsonl")[:MAX_DOCS], bk,
                                DATA, MAX_REGIONS, cache, stats=stats)
    out = {"train_feats": tr[0], "train_pii": tr[2], "train_mask": tr[4],
           "val_feats": va[0], "val_pii": va[2], "val_mask": va[4]}
    os.makedirs("results", exist_ok=True)
    torch.save(out, CACHE)
    return out


def main():
    d = load_features()
    xtr, ztr = d["train_feats"][d["train_mask"]], d["train_pii"][d["train_mask"]]
    xva, zva = d["val_feats"][d["val_mask"]], d["val_pii"][d["val_mask"]]

    cross = fit_leace(xtr, ztr)        # fit on family A, applied to family B
    within = fit_leace(xva, zva)       # in-distribution control (upper bound)

    rows = {
        "n_train_regions": int(xtr.shape[0]),
        "n_val_regions": int(xva.shape[0]),
        "residual_cov": {
            "cross_fit_on_train_measured_on_train": cross.residual_covariance(xtr, ztr),
            "cross_fit_on_train_measured_on_val": cross.residual_covariance(xva, zva),
            "within_fit_on_val_measured_on_val": within.residual_covariance(xva, zva),
        },
        "probes_on_val": {
            "raw": probe_leakage(xva, zva),
            "erased_cross_family": probe_leakage(cross(xva), zva),
            "erased_within_family": probe_leakage(within(xva), zva),
        },
        "interpretation": (
            "If erased_within_family drops to the majority baseline while "
            "erased_cross_family does not, the LEACE implementation is correct "
            "and the guarantee simply does not transfer to an unseen PII "
            "distribution — which is the condition any deployment faces."),
    }
    with open("results/leace_transfer_control.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    rc = rows["residual_cov"]
    print(f"residual cov  train/train {rc['cross_fit_on_train_measured_on_train']:.3e}"
          f"  train/val {rc['cross_fit_on_train_measured_on_val']:.3e}"
          f"  val/val {rc['within_fit_on_val_measured_on_val']:.3e}")
    for name, p in rows["probes_on_val"].items():
        print(f"{name:<22} lin {p['linear']['accuracy']:.3f}  "
              f"mlp {p['mlp']['accuracy']:.3f}  "
              f"(majority {p['majority_baseline']:.3f})")


if __name__ == "__main__":
    main()
