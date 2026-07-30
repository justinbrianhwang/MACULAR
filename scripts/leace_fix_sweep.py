"""Can the LEACE transfer failure be fixed?

The control established the implementation is correct: fitting on validation and
probing validation drives the linear probe to exactly the majority baseline
(0.847 vs 0.847). Fitting on train and probing validation leaves it at 0.932,
because our splits use disjoint PII generator families and the concept subspace
carrying family A's values is not the one carrying family B's.

Before concluding "closed-form erasure does not work here", test the cheap
remedies. All of them reuse the cached features, so this runs in seconds on CPU
and can be iterated freely:

  coarser concept   erase the binary is-PII distinction rather than the 8-way
                    type. A coarser target should be less family-specific.
  shrinkage         a heavier ridge on the covariance estimate, trading exact
                    in-distribution erasure for a more stable subspace.
  rank truncation   keep only the top-k concept directions, on the theory that
                    the tail directions are family-specific noise.
  whitened-only     drop the un-whitening step (a blunter, less minimal edit).

Reported against the two reference points that matter: the majority baseline
(perfect erasure) and the cross-family LEACE number (the thing to beat).
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macular.privacy import probe_leakage                            # noqa: E402
from macular.privacy.leace import fit_leace, LeaceEraser, _inv_sqrt_psd  # noqa: E402

CACHE = "results/_features_paddleocr_vl.pt"


def fit_leace_rank_k(x, z, k, eps=1e-6):
    """LEACE with the concept subspace truncated to its top-k directions."""
    x = x.detach().to(torch.float64)
    if z.dim() == 1:
        z = torch.nn.functional.one_hot(z.long(), num_classes=int(z.max()) + 1)
    z = z.detach().to(torch.float64)
    n = x.shape[0]
    mu = x.mean(0)
    xc, zc = x - mu, z - z.mean(0)
    w, w_pinv = _inv_sqrt_psd(xc.T @ xc / (n - 1), eps)
    u, s, _ = torch.linalg.svd(w @ (xc.T @ zc / (n - 1)), full_matrices=False)
    u = u[:, :k]
    a = torch.eye(x.shape[1], dtype=x.dtype) - w_pinv @ (u @ u.T) @ w
    return LeaceEraser(a.to(torch.float32), mu.to(torch.float32))


def main():
    d = torch.load(CACHE)
    xtr, ztr = d["train_feats"][d["train_mask"]], d["train_pii"][d["train_mask"]]
    xva, zva = d["val_feats"][d["val_mask"]], d["val_pii"][d["val_mask"]]
    btr, bva = (ztr > 0).long(), (zva > 0).long()

    variants = {}
    variants["baseline_type_8way"] = fit_leace(xtr, ztr)
    variants["coarser_binary_is_pii"] = fit_leace(xtr, btr)
    for eps in (1e-4, 1e-2, 1e-1):
        variants[f"shrinkage_eps_{eps:g}"] = fit_leace(xtr, ztr, eps=eps)
    for k in (1, 2, 4):
        variants[f"rank_truncated_k{k}"] = fit_leace_rank_k(xtr, ztr, k)

    ref_raw = probe_leakage(xva, zva, seed=0)
    rows = {"reference": {
        "raw_linear": ref_raw["linear"]["accuracy"],
        "raw_mlp": ref_raw["mlp"]["accuracy"],
        "majority_baseline": ref_raw["majority_baseline"],
    }, "variants": {}}

    print(f"raw                        lin {ref_raw['linear']['accuracy']:.3f}  "
          f"mlp {ref_raw['mlp']['accuracy']:.3f}   "
          f"(majority {ref_raw['majority_baseline']:.3f})")
    for name, er in variants.items():
        p = probe_leakage(er(xva), zva, seed=0)
        resid = er.residual_covariance(xva, zva)
        rows["variants"][name] = {
            "linear": p["linear"]["accuracy"], "mlp": p["mlp"]["accuracy"],
            "linear_above_majority": p["linear"]["accuracy_above_majority"],
            "val_residual_covariance": resid,
        }
        print(f"{name:<26} lin {p['linear']['accuracy']:.3f}  "
              f"mlp {p['mlp']['accuracy']:.3f}   resid {resid:.3e}")

    with open("results/leace_fix_sweep.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
