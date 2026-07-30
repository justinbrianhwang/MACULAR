"""Coordinate-only shortcut baseline (proposal 14.4 / gate 5 in section 25).

Trains a PII classifier using ONLY geometric features (box position, size,
block-type one-hot) — no text, no pixels. If this classifier scores highly,
the synthetic generator is leaking a positional shortcut and must be fixed
(e.g. enable counterfactual_layout).

This is a genuine, GPU-free experiment that returns a verdict.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..schema import Document, BLOCK_TYPES
from ..evaluation.metrics import binary_metrics, document_zero_leakage_rate


def _features(doc: Document):
    X, y, ids = [], [], []
    for c in doc.candidates:
        b = c.bbox
        onehot = [1.0 if c.block_type == bt else 0.0 for bt in BLOCK_TYPES]
        X.append([b.x0, b.y0, b.x1, b.y1, b.width, b.height, *onehot])
        y.append(1 if c.is_pii else 0)
        ids.append(doc.doc_id)
    return X, y, ids


def _collect(docs):
    X, y, ids = [], [], []
    for d in docs:
        fx, fy, fi = _features(d)
        X.extend(fx)
        y.extend(fy)
        ids.extend(fi)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), ids


def run(train_docs, test_docs, shortcut_threshold: float = 0.75) -> dict:
    Xtr, ytr, _ = _collect(train_docs)
    Xte, yte, ids_te = _collect(test_docs)

    if len(set(ytr.tolist())) < 2:
        return {"error": "training labels are single-class; cannot fit"}

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte).astype(int).tolist()

    m = binary_metrics(yte.tolist(), pred)
    dz = document_zero_leakage_rate(ids_te, yte.tolist(), pred)
    shortcut_detected = m["f1"] >= shortcut_threshold
    return {
        "experiment": "coordinate_only_shortcut_audit",
        "n_train_candidates": int(len(ytr)),
        "n_test_candidates": int(len(yte)),
        "metrics": m,
        "document_privacy": dz,
        "shortcut_threshold": shortcut_threshold,
        "shortcut_detected": bool(shortcut_detected),
        "verdict": (
            "SHORTCUT PRESENT — geometry alone predicts PII. Fix the generator "
            "(enable counterfactual_layout) before trusting safety numbers."
            if shortcut_detected else
            "OK — geometry alone is a weak PII predictor."
        ),
    }
