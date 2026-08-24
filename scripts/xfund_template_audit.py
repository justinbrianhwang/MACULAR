"""Near-duplicate form-template audit across the XFUND train/eval halves.

Reviewer condition 9: the self-split is document-level, but two pages of the
same blank template (different fill-ins) could sit on opposite sides. This
computes a 16x16 difference hash per page and reports cross-half pairs whose
Hamming distance is small, plus the same statistic within halves for scale.

Usage: python scripts/xfund_template_audit.py  -> results/xfund_template_audit.json
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macular.models.ocr_adapt import halve_by_language  # noqa: E402
from macular.schema import read_jsonl  # noqa: E402


def dhash(path, size=16):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def main():
    out = {}
    for corpus in ("data/xfund", "data/xfund_cjk"):
        docs = read_jsonl(os.path.join(corpus, "test.jsonl"))
        train, evald = halve_by_language(docs)
        h = {d.doc_id: dhash(os.path.join(corpus, d.image_path)) for d in docs}
        lang = {d.doc_id: d.language for d in docs}

        def dists(A, B, same=False):
            res = []
            for i, a in enumerate(A):
                for j, b in enumerate(B):
                    if same and j <= i:
                        continue
                    if lang[a] != lang[b]:
                        continue
                    res.append((int((h[a] != h[b]).sum()), a, b))
            return sorted(res)

        cross = dists([d.doc_id for d in train], [d.doc_id for d in evald])
        within_tr = dists([d.doc_id for d in train], [d.doc_id for d in train], same=True)
        within_ev = dists([d.doc_id for d in evald], [d.doc_id for d in evald], same=True)
        thr = 24  # of 256 bits: <10% differing bits = same template family
        out[corpus] = {
            "n_train": len(train), "n_eval": len(evald), "hash_bits": 256,
            "threshold_bits": thr,
            "cross_half_pairs_below_threshold": [p for p in cross if p[0] < thr],
            "cross_half_min_distance": cross[0][0],
            "cross_half_distance_percentiles": {
                q: int(np.percentile([p[0] for p in cross], q)) for q in (1, 5, 25, 50)},
            "within_train_pairs_below_threshold": len([p for p in within_tr if p[0] < thr]),
            "within_eval_pairs_below_threshold": len([p for p in within_ev if p[0] < thr]),
            "closest_cross_pairs": cross[:10],
        }
        print(corpus, {k: v for k, v in out[corpus].items() if k != "closest_cross_pairs"})
    json.dump(out, open("results/xfund_template_audit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
